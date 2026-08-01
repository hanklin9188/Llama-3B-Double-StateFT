import csv
import math
import os
from collections import defaultdict

import numpy as np
import torch
from transformers import TrainerCallback

from ..wrapping.wrap import _find_llama_like_layers_container


class AdaptiveRankAllocatorCallback(TrainerCallback):
    """Redistribute a fixed per-branch rank budget using representation geometry."""

    def __init__(
        self,
        results_dir,
        total_budget,
        rank_min,
        rank_max,
        grad_recorder=None,
        loss_recorder=None,
        warmup_evals=3,
        ema_alpha=0.3,
        max_rank_step=2,
        cooldown_evals=3,
        weight_id=0.7,
        weight_delta=0.3,
        weight_gradient=0.3,
        weight_effective_rank=0.4,
    ):
        self.results_dir = results_dir
        self.total_budget = int(total_budget)
        self.rank_min = int(rank_min)
        self.rank_max = int(rank_max)
        self.grad_recorder = grad_recorder
        self.loss_recorder = loss_recorder
        self.warmup_evals = int(warmup_evals)
        self.ema_alpha = float(ema_alpha)
        self.max_rank_step = int(max_rank_step)
        self.cooldown_evals = int(cooldown_evals)
        self.weight_id = float(weight_id)
        self.weight_delta = float(weight_delta)
        self.weight_gradient = float(weight_gradient)
        self.weight_effective_rank = float(weight_effective_rank)
        self.eval_count = 0
        self.last_update_eval = -10**9
        self.ema_id = {}
        self.ema_delta = {}
        self.sleep_count = defaultdict(int)

    @staticmethod
    def _normalize(values):
        array = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(array)
        if not finite.any():
            return np.full_like(array, 0.5)
        replacement = np.nanmedian(array[finite])
        array = np.where(finite, array, replacement)
        median = np.median(array)
        mad = np.median(np.abs(array - median))
        if mad < 1e-12:
            return np.full_like(array, 0.5)
        robust = np.clip((array - median) / (1.4826 * mad), -2.5, 2.5)
        span = robust.max() - robust.min()
        return np.full_like(array, 0.5) if span < 1e-12 else (robust - robust.min()) / span

    def _read_latest_geometry(self):
        path = os.path.join(self.results_dir, "layer_id_ii_all.csv")
        if not os.path.isfile(path):
            return []
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []
        latest_step = max(int(row["step"]) for row in rows)
        latest = []
        for row in rows:
            if int(row["step"]) != latest_step:
                continue
            latest.append(
                {
                    "layer": int(row["layer"]),
                    "id": float(row["id_gride_med"]),
                    "delta": 0.5 * (
                        float(row["delta_to_first"]) + float(row["delta_to_last"])
                    ),
                }
            )
        return sorted(latest, key=lambda row: row["layer"])

    def _read_effective_rank_saturation(self, step, layers):
        path = os.path.join(self.results_dir, "layer_capacity_all.csv")
        saturation = {layer: 0.0 for layer in layers}
        if not os.path.isfile(path):
            return saturation
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        eligible = [int(row["step"]) for row in rows if int(row["step"]) <= step]
        if not eligible:
            return saturation
        selected_step = max(eligible)
        for row in rows:
            if int(row["step"]) != selected_step:
                continue
            values = []
            for branch in ("mlp", "attn"):
                effective = float(row[f"reff_{branch}"])
                active = float(row[f"r_active_{branch}"])
                if math.isfinite(effective) and active > 0:
                    values.append(min(1.0, effective / active))
            if values:
                saturation[int(row["layer"])] = sum(values) / len(values)
        return saturation

    def _allocate(self, scores):
        count = len(scores)
        minimum_budget = count * self.rank_min
        maximum_budget = count * self.rank_max
        budget = min(max(self.total_budget, minimum_budget), maximum_budget)
        ranks = [self.rank_min] * count
        remaining = budget - minimum_budget
        while remaining > 0:
            candidates = [
                (scores[index] / (1.0 + ranks[index] - self.rank_min), index)
                for index in range(count)
                if ranks[index] < self.rank_max
            ]
            if not candidates:
                break
            _, selected = max(candidates)
            ranks[selected] += 1
            remaining -= 1
        return ranks

    def _apply(self, model, layer_ids, targets):
        _, _, layers, _ = _find_llama_like_layers_container(model)
        applied = []
        for layer_id, target in zip(layer_ids, targets):
            layer = layers[layer_id - 1]
            current = layer.ctrl_mlp.active_rank
            lower = max(self.rank_min, current - self.max_rank_step)
            upper = min(self.rank_max, current + self.max_rank_step)
            next_rank = max(lower, min(upper, int(target)))
            layer.ctrl_attn.set_active_rank(next_rank)
            layer.ctrl_mlp.set_active_rank(next_rank)
            applied.append(next_rank)
        return applied

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        self.eval_count += 1
        rows = self._read_latest_geometry()
        if not rows:
            return

        layer_ids = [row["layer"] for row in rows]
        ids = []
        deltas = []
        for row in rows:
            layer = row["layer"]
            old_id = self.ema_id.get(layer, row["id"])
            old_delta = self.ema_delta.get(layer, row["delta"])
            self.ema_id[layer] = self.ema_alpha * row["id"] + (1 - self.ema_alpha) * old_id
            self.ema_delta[layer] = self.ema_alpha * row["delta"] + (1 - self.ema_alpha) * old_delta
            ids.append(self.ema_id[layer])
            deltas.append(self.ema_delta[layer])

        if self.eval_count <= self.warmup_evals:
            print(f"[AdaptiveRank] warmup {self.eval_count}/{self.warmup_evals}; ranks unchanged")
            return
        if self.eval_count - self.last_update_eval < self.cooldown_evals:
            return

        scores = self.weight_id * self._normalize(ids)
        scores += self.weight_delta * self._normalize(deltas)
        if self.grad_recorder and self.grad_recorder.g_ema:
            gradients = [self.grad_recorder.g_ema.get(layer, 0.0) for layer in layer_ids]
            scores *= 1.0 + self.weight_gradient * self._normalize(gradients)

        step = max(int(row.get("step", state.global_step)) for row in rows)
        saturation = self._read_effective_rank_saturation(step, layer_ids)
        scores *= 1.0 + self.weight_effective_rank * np.asarray(
            [saturation[layer] for layer in layer_ids]
        )

        if self.loss_recorder and self.loss_recorder.delta_loss is not None:
            if self.loss_recorder.delta_loss > 0:
                scores = 0.8 * scores + 0.2 * np.mean(scores)

        targets = self._allocate(scores.tolist())
        applied = self._apply(model or kwargs.get("model"), layer_ids, targets)
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "rank_all.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["step", "layer", "target_rank", "applied_rank"])
            for layer, target, actual in zip(layer_ids, targets, applied):
                writer.writerow([state.global_step, layer, target, actual])
        self.last_update_eval = self.eval_count
        print(
            f"[AdaptiveRank] step={state.global_step} per-branch budget={self.total_budget} "
            f"applied={applied}"
        )
