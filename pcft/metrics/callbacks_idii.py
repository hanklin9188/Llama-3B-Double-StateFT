from ..common_imports import *
from ..io.state import save_control_state
from ..wrapping.wrap import _find_llama_like_layers_container, _get_base_model_after_peft
from .metrics import (
    _center_gram, _gram_rbf, _matrix_based_entropy,
    _effective_rank_from_ctrl, _last_non_eos_index, _gather_last_token, _to_numpy,
    _compute_pairwise_ranks, _delta_a_to_b, _safe_import_dadapy, _id_twonN_and_gride
)

class EvalIDIIRecorderCallback(TrainerCallback):
    def __init__(
        self,
        trainer: Trainer,
        *,
        sample_size: int = 256,
        gride_range_max: int = 64,
        gride_k_lo: int = 8,
        gride_k_hi: int = 32,
        k_for_delta: int = 1,
        results_dir: str = "",
        # ===== NEW: MBE knobs =====
        ent_enable: bool = True,
        ent_alpha: float = 2.0,
        ent_sigma: float = 0.0,     # <=0 → 使用 median heuristic
        ent_center: bool = True,
        ent_log_base: float = 2.0,
        save_ckpt_on_identropy_peak: bool = True,
        save_ckpt_on_id_best: bool = True,
        save_ckpt_on_entropy_best: bool = True,
        peak_window: int = 3,                # 用近幾次 eval 的斜率變化來判斷局部高峰
        min_rel_improve: float = 0.002,      # 至少比歷史最佳提升 0.2% 才當成新高峰
        control_cfg: Optional[Dict[str, Any]] = None,  # 讓我們能呼叫 save_control_state
    ):
        self.trainer = trainer
        self.sample_size = int(sample_size)
        self.gride_range_max = int(gride_range_max)
        self.gride_k_lo = int(gride_k_lo)
        self.gride_k_hi = int(gride_k_hi)
        self.results_dir = results_dir or os.path.join(trainer.args.output_dir, "metrics")
        os.makedirs(self.results_dir, exist_ok=True)

        # NEW (MBE)
        self.ent_enable = bool(ent_enable)
        self.ent_alpha = float(ent_alpha)
        self.ent_sigma = float(ent_sigma)
        self.ent_center = bool(ent_center)
        self.ent_log_base = float(ent_log_base)


        self.save_ckpt_on_identropy_peak = bool(save_ckpt_on_identropy_peak)
        self.save_ckpt_on_id_best = bool(save_ckpt_on_id_best)
        self.save_ckpt_on_entropy_best = bool(save_ckpt_on_entropy_best)

        self.peak_window = int(max(2, peak_window))
        self.min_rel_improve = float(min_rel_improve)
        self.control_cfg = control_cfg or {}

        from collections import deque
        self._hist_global_id = deque(maxlen=16)    # [(step, mean_id)]
        self._hist_global_ent = deque(maxlen=16)   # [(step, mean_entropy)]
        self._best_id = -float("inf")
        self._best_ent = -float("inf")
        self._saved_ident_steps = set()            # 避免重複存同一步
        self._saved_idbest_steps = set()
        self._saved_entbest_steps = set()

    def _slope(self, arr):
        # arr: list of (step, value)
        if len(arr) < 2:
            return 0.0
        xs = np.array([x for x, _ in arr], dtype=float)
        ys = np.array([y for _, y in arr], dtype=float)
        xs = xs - xs.mean()
        denom = (xs**2).sum()
        if denom <= 0:
            return 0.0
        return float(((xs * (ys - ys.mean())).sum()) / denom)

    @staticmethod
    def _peak_span(ids: List[float]) -> Tuple[int, int, int]:
        L = len(ids) - 1
        peak = 1 + int(np.argmax(ids[1:]))
        end = L
        for k in range(peak + 1, L):
            if ids[k] <= ids[k - 1] and ids[k] <= ids[k + 1]:
                end = k
                break
        end_val = ids[end]
        start = 1
        for k in range(peak - 1, 0, -1):
            if ids[k] >= end_val:
                start = k
            else:
                break
        return start, peak, end

    def _get_eval_batch_iter(self):
        dl = self.trainer.get_eval_dataloader()
        for batch in dl:
            yield self.trainer._prepare_inputs(batch)

    @torch.no_grad()
    def _collect_layer_last_token_reps(self, model: nn.Module) -> List[torch.Tensor]:
        model_was_training = model.training
        model.eval()
        need = self.sample_size
        hs_lists: Optional[List[List[torch.Tensor]]] = None
        collected = 0

        def _capture_with_hooks(_model, input_ids, attention_mask):
            container, attr, layers, base = _find_llama_like_layers_container(_model)
            if layers is None:
                return None
            base_root = _get_base_model_after_peft(_model)
            embed = None
            try:
                embed = getattr(getattr(base_root, "model", base_root), "embed_tokens", None)
            except Exception:
                embed = None
            if embed is None:
                embed = getattr(getattr(base_root, "model", base_root), "tok_embeddings", None)

            captured_emb: List[torch.Tensor] = []
            captured_layers: List[torch.Tensor] = []
            hooks = []
            if embed is not None:
                hooks.append(embed.register_forward_hook(
                    lambda m, inp, out: captured_emb.append(out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else out))
                ))
            for lay in layers:
                hooks.append(lay.register_forward_hook(
                    lambda m, inp, out: captured_layers.append(out if isinstance(out, torch.Tensor) else (out[0] if isinstance(out, (tuple, list)) else out))
                ))
            _ = _model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            for h in hooks:
                try: h.remove()
                except Exception: pass
            if not captured_layers:
                return None
            emb = captured_emb[0] if captured_emb else captured_layers[0]
            return [emb] + captured_layers[: len(layers)]

        for batch in self._get_eval_batch_iter():
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = getattr(outputs, "hidden_states", None)

            if hidden_states is None:
                try:
                    old_flag = getattr(model.config, "output_hidden_states", False)
                    model.config.output_hidden_states = True
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                    hidden_states = getattr(outputs, "hidden_states", None)
                    model.config.output_hidden_states = old_flag
                except Exception:
                    hidden_states = None

            if hidden_states is None:
                hidden_states = _capture_with_hooks(model, input_ids, attention_mask)
                if hidden_states is None:
                    raise RuntimeError("hidden_states is None 且 hook fallback 失敗，無法計算 ID/II/MBE。")

            if hs_lists is None:
                hs_lists = [[] for _ in range(len(hidden_states))]

            processor = self.trainer.processing_class
            eos_id = getattr(processor, "eos_token_id", None)
            if eos_id is None:
                last_idx = attention_mask.long().sum(dim=1) - 1
            else:
                last_idx = _last_non_eos_index(input_ids, attention_mask, eos_id)
            for li, h in enumerate(hidden_states):
                h_t = h if isinstance(h, torch.Tensor) else (h[0] if isinstance(h, (tuple, list)) else None)
                if h_t is None:
                    raise RuntimeError(f"hidden_states[{li}] 不是 tensor（拿到 {type(h)}）。")
                last_vecs = _gather_last_token(h_t, last_idx)  # (B, D)
                hs_lists[li].append(last_vecs)

            collected += input_ids.size(0)
            if collected >= need:
                break

        if hs_lists is None:
            raise RuntimeError("沒有處理到任何 batch，無法蒐集 hidden states。")

        layer_arrays = [torch.cat(lst, dim=0)[:need] for lst in hs_lists]
        if model_was_training:
            model.train()
        return layer_arrays  # [L+1]

    def _load_baseline_step200(self) -> Optional[Dict[int, Dict[str, float]]]:
        path_all = os.path.join(self.results_dir, "layer_id_ii_all.csv")
        if not os.path.exists(path_all):
            return None
        import csv
        baseline: Dict[int, Dict[str, float]] = {}
        with open(path_all, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    if int(row["step"]) != 200:
                        continue
                    li = int(row["layer"])
                    # 這裡同時讀兩個 delta 欄位；舊檔若沒 delta_to_first 就先用 delta_to_last 頂一下
                    d1 = float(row.get("delta_to_first", row.get("delta_to_last", "nan")))
                    dL = float(row.get("delta_to_last", "nan"))
                    baseline[li] = {
                        "id_gride_med": float(row["id_gride_med"]),
                        "delta_to_first": d1,
                        "delta_to_last": dL,
                    }
                except Exception:
                    continue
        return baseline if baseline else None


    def _percent(self, now: float, base: float) -> Optional[float]:
        if base == 0.0 or not math.isfinite(base) or not math.isfinite(now):
            return None
        return 100.0 * (now - base) / abs(base)

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model", self.trainer.model)
        try:
            reps_per_layer = self._collect_layer_last_token_reps(model)  # list length L+1
        except RuntimeError as e:
            print("[ID/II] 評估時收集 hidden_states 失敗：", e)
            return

        Lp1 = len(reps_per_layer)
        if Lp1 < 2:
            print("[ID/II] hidden_states 少於 2 層，略過。")
            return
        id_values = [0.0] * Lp1
        first = reps_per_layer[1]
        last  = reps_per_layer[-1]

        device = first.device
        ranks_first = _compute_pairwise_ranks(first)
        ranks_last  = _compute_pairwise_ranks(last)

        header = ["step", "layer", "id_gride_med", "delta_to_first", "delta_to_last"]
        baseline = self._load_baseline_step200() if state.global_step >= 400 else None
        show_change = baseline is not None

        agg_csv = os.path.join(self.results_dir, "layer_id_ii_all.csv")
        first_time_agg = not os.path.exists(agg_csv)

        print("\n[ID/II] Layer-wise metrics @ step", state.global_step)
        print("-" * 86)
        if show_change:
            print(f"{'layer':>5} | {'GRIDE_med':>10} {'Δ%':>6} | {'Δ→first':>8} {'Δ%':>6} | {'Δ→last':>8} {'Δ%':>6}")
        else:
            print(f"{'layer':>5} | {'GRIDE_med':>10} | {'Δ→first':>8} | {'Δ→last':>8}")
        print("-" * 86)

        import csv
        with open(agg_csv, "a", newline="") as f_all:
            w_all = csv.writer(f_all)
            if first_time_agg:
                w_all.writerow(header)

            for li in range(1, Lp1):
                X = reps_per_layer[li]
                # Info Imbalance
                dist_a = torch.cdist(X, X)
                dist_a.fill_diagonal_(float("inf"))
                nn_j = dist_a.argmin(dim=1)
                N = X.size(0)
                r_first = ranks_first[torch.arange(N, device=device), nn_j]
                r_last  = ranks_last [torch.arange(N, device=device), nn_j]
                delta_first = float(((r_first - 1).float() / (N - 1)).mean().item())
                delta_last  = float(((r_last  - 1).float() / (N - 1)).mean().item())

                # ID (GRIDEmed over [k_lo, k_hi])
                X_np = _to_numpy(X)
                try:
                    _, idg = _id_twonN_and_gride(
                        X_np,
                        range_max=self.gride_range_max,
                        k_lo=self.gride_k_lo,
                        k_hi=self.gride_k_hi,
                    )
                except Exception as e:
                    print(f"[ID/II] 計算 ID 失敗（layer {li}）：{e}")
                    idg = float("nan")
                id_values[li] = idg

                row_csv = [state.global_step, li, idg, delta_first, delta_last]
                if show_change and (baseline is not None) and (li in baseline):
                    b = baseline[li]
                    p_idg = self._percent(idg,  b["id_gride_med"])
                    p_d1  = self._percent(delta_first, b["delta_to_first"])
                    p_dL  = self._percent(delta_last,  b["delta_to_last"])

                    def fmt(v):
                        return ("  —  " if (v is None or (isinstance(v, float) and np.isnan(v)))
                                else f"{v:+.2f}%")

                    print(f"{li:5d} | {idg:10.3f} {fmt(p_idg):>6} | "
                          f"{delta_first:8.3f} {fmt(p_d1):>6} | "
                          f"{delta_last:8.3f} {fmt(p_dL):>6}")
                else:
                    print(f"{li:5d} | {idg:10.3f} | {delta_first:8.3f} | {delta_last:8.3f}")
                w_all.writerow(row_csv)

        print(f"[ID/II] 累積彙總：{agg_csv}")
        print("-" * 86 + "\n")
        span_s, span_p, span_e = self._peak_span(id_values)
        span_w = span_e - span_s + 1
        print(f"[ID-SPAN] start={span_s:2d}, peak={span_p:2d}, end={span_e:2d}, width={span_w:2d}")

        span_csv = os.path.join(self.results_dir, "id_peak_span.csv")
        first_span = not os.path.exists(span_csv)
        with open(span_csv, "a", newline="") as fspan:
            wspan = csv.writer(fspan)
            if first_span:
                wspan.writerow(["step", "start_layer", "peak_layer", "end_layer", "width"])
            wspan.writerow([state.global_step, span_s, span_p, span_e, span_w])

        # ===== NEW: 每層 Matrix-based Entropy（MBE）計算與輸出 =====
        ent_values = []  # ★ 收集每層 entropy（忽略 embedding 層 0），供後面 mean_entropy 與峰值偵測

        if self.ent_enable:
            ent_csv = os.path.join(self.results_dir, "layer_entropy_all.csv")
            first_time_ent = not os.path.exists(ent_csv)
            with open(ent_csv, "a", newline="") as fe:
                we = csv.writer(fe)
                if first_time_ent:
                    we.writerow(["step", "layer", f"mbe_alpha{self.ent_alpha}"])
                print(f"[MBE] α={self.ent_alpha} | sigma={'median' if self.ent_sigma<=0 else self.ent_sigma} "
                      f"| center={self.ent_center} | log_base={self.ent_log_base}")
                print(f"{'layer':>5} | {'MBE':>10}")
                print("-" * 22)
                for li in range(1, Lp1):
                    X = reps_per_layer[li]  # (B, D)
                    mbe = _matrix_based_entropy(
                        X,
                        alpha=self.ent_alpha,
                        sigma=(self.ent_sigma if self.ent_sigma > 0 else None),
                        center=self.ent_center,
                        log_base=self.ent_log_base
                    )
                    we.writerow([state.global_step, li, mbe])
                    print(f"{li:5d} | {mbe:10.4f}")
                    if li != 0 and np.isfinite(mbe):
                        ent_values.append(float(mbe))  # ★

            print(f"[MBE] 累積彙總：{ent_csv}")
                # ===== NEW: 每層 r_eff 與啟用 rank（capacity）=====
        cap_csv = os.path.join(self.results_dir, "layer_capacity_all.csv")
        first_time_cap = not os.path.exists(cap_csv)
        with open(cap_csv, "a", newline="") as fc:
            wc = csv.writer(fc)
            if first_time_cap:
                wc.writerow(["step", "layer", "reff_mlp", "r_active_mlp", "reff_attn", "r_active_attn"])
            container, attr, layers, _ = _find_llama_like_layers_container(model)
            if layers is not None:
                print(f"[CAP] r_eff per layer (step={state.global_step})")
                print(f"{'layer':>5} | {'reff_mlp':>9} | {'reff_attn':>9}")
                print("-" * 36)
                for li, lay in enumerate(layers, start=1):
                    reff_mlp = float("nan"); ract_mlp = float("nan")
                    reff_attn = float("nan"); ract_attn = float("nan")

                    ctrl_m = getattr(lay, "ctrl_mlp", None)
                    if ctrl_m is not None:
                        reff_mlp, ract_mlp = _effective_rank_from_ctrl(ctrl_m)

                    ctrl_a = getattr(lay, "ctrl_attn", None)
                    if ctrl_a is not None:
                        reff_attn, ract_attn = _effective_rank_from_ctrl(ctrl_a)

                    wc.writerow([state.global_step, li, reff_mlp, ract_mlp, reff_attn, ract_attn])
                    rm = (f"{reff_mlp:9.3f}" if math.isfinite(reff_mlp) else f"{'—':>9}")
                    ra = (f"{reff_attn:9.3f}" if math.isfinite(reff_attn) else f"{'—':>9}")
                    print(f"{li:5d} | {rm} | {ra}")
        print(f"[CAP] 累積彙總：{cap_csv}")

        # === 產生全層平均（忽略 index 0 的 embedding 層） ===
        valid_ids = [float(v) for v in (id_values[1:] if len(id_values) > 1 else id_values) if np.isfinite(v)]
        mean_id = float(np.mean(valid_ids)) if len(valid_ids) else float("nan")

        # Entropy：若未開啟 ent_enable 或本輪沒成功計算，也能安全運作
        if "ent_values" not in locals() or ent_values is None:
            ent_values = []
        valid_ents = [float(v) for v in ent_values if np.isfinite(v)]
        mean_ent = float(np.mean(valid_ents)) if len(valid_ents) else float("nan")


        # === 記錄到全域 metrics CSV（方便之後對齊） ===
        gcsv = os.path.join(self.results_dir, "global_metrics.csv")
        is_new = not os.path.exists(gcsv)
        with open(gcsv, "a", newline="") as fg:
            import csv
            wg = csv.writer(fg)
            if is_new:
                wg.writerow(["step", "mean_id", "mean_entropy"])
            wg.writerow([state.global_step, mean_id, mean_ent])
        # 若 entropy 這輪沒算到（mean_ent 不是 finite），跳過同步高峰檢測
        if not np.isfinite(mean_ent):
            # 仍然會把 global_metrics.csv 寫入（上面已寫），但不做 IDENTPEAK 存檔
            return

        # === 更新歷史緩衝並檢測同步高峰（ID 與 Entropy 同時由升轉降 or 達歷史新高） ===
        self._hist_global_id.append((state.global_step, mean_id))
        self._hist_global_ent.append((state.global_step, mean_ent))

        # 達「歷史新高」的門檻（避免雜訊觸發）
        id_ok = (mean_id >= self._best_id * (1.0 + self.min_rel_improve)) or (mean_id > self._best_id and self._best_id == -float("inf"))
        ent_ok = (mean_ent >= self._best_ent * (1.0 + self.min_rel_improve)) or (mean_ent > self._best_ent and self._best_ent == -float("inf"))

        # 也同時看近 window 的斜率是否由正轉負（局部峰值）
        id_slope = self._slope(list(self._hist_global_id)[-self.peak_window:])
        ent_slope = self._slope(list(self._hist_global_ent)[-self.peak_window:])

        simul_peak = (id_slope <= 0.0 and ent_slope <= 0.0) and (id_ok and ent_ok)

        if self.save_ckpt_on_identropy_peak and simul_peak and state.global_step not in self._saved_ident_steps:
            # 更新歷史最佳
            self._best_id = max(self._best_id, mean_id)
            self._best_ent = max(self._best_ent, mean_ent)

            # 額外存 checkpoint 在 output_dir/peaks/，避免被 HF 的 save_total_limit 清掉
            peak_dir = os.path.join(self.trainer.args.output_dir, "peaks", f"checkpoint-IDENTPEAK-{state.global_step}")
            os.makedirs(peak_dir, exist_ok=True)

            # 1) 存 HuggingFace 模型檔（權重、config、tokenizer…）
            self.trainer.save_model(peak_dir)

            # Save the Double Control adapter state alongside the peak metadata.
            try:
                save_control_state(self.trainer.model, peak_dir, self.control_cfg)
            except Exception as e:
                print("[IDENT-PEAK SAVE] save_control_state failed:", e)

            self._saved_ident_steps.add(state.global_step)
            print(f"[IDENT-PEAK SAVE] Saved extra checkpoint → {peak_dir}  "
                f"(mean_id={mean_id:.4f}, mean_entropy={mean_ent:.4f})")
                # === 單獨「ID 創歷史新高」→ 存 IDBEST ===
        if self.save_ckpt_on_id_best:
            # 允許極小抖動：至少比舊最佳提升 min_rel_improve 比例
            id_improved = (
                (not np.isfinite(self._best_id)) or
                (np.isfinite(mean_id) and mean_id >= self._best_id * (1.0 + self.min_rel_improve))
            )
            if id_improved and state.global_step not in self._saved_idbest_steps:
                self._best_id = max(self._best_id, mean_id)
                id_dir = os.path.join(self.trainer.args.output_dir, "peaks", f"checkpoint-IDBEST-{state.global_step}")
                os.makedirs(id_dir, exist_ok=True)
                self.trainer.save_model(id_dir)
                try:
                    save_control_state(self.trainer.model, id_dir, self.control_cfg)
                except Exception as e:
                    print("[ID-BEST SAVE] save_control_state failed:", e)
                self._saved_idbest_steps.add(state.global_step)
                print(f"[ID-BEST SAVE] Saved extra checkpoint → {id_dir}  (mean_id={mean_id:.4f})")

        # === 單獨「Entropy 創歷史新高」→ 存 ENTBEST ===
        if self.save_ckpt_on_entropy_best and np.isfinite(mean_ent):
            ent_improved = (
                (not np.isfinite(self._best_ent)) or
                (mean_ent >= self._best_ent * (1.0 + self.min_rel_improve))
            )
            if ent_improved and state.global_step not in self._saved_entbest_steps:
                self._best_ent = max(self._best_ent, mean_ent)
                ent_dir = os.path.join(self.trainer.args.output_dir, "peaks", f"checkpoint-ENTBEST-{state.global_step}")
                os.makedirs(ent_dir, exist_ok=True)
                self.trainer.save_model(ent_dir)
                try:
                    save_control_state(self.trainer.model, ent_dir, self.control_cfg)
                except Exception as e:
                    print("[ENT-BEST SAVE] save_control_state failed:", e)
                self._saved_entbest_steps.add(state.global_step)
                print(f"[ENT-BEST SAVE] Saved extra checkpoint → {ent_dir}  (mean_entropy={mean_ent:.4f})")
