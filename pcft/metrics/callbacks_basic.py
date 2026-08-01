from ..common_imports import *
from ..wrapping.wrap import _find_llama_like_layers_container

class GradNormRecorderCallback(TrainerCallback):
    """
    收集每層（可能含 ctrl_mlp / ctrl_attn）的 LoRA/ControlLoRA 梯度能量，做 EMA 平滑。
    提供：
      - g_ema[layer_idx] : float
    也不在此決定 recycle，僅提供訊號。
    """
    def __init__(self, beta: float = 0.9, eps: float = 1e-12):
        self.beta = float(beta)
        self.eps = float(eps)
        self.g_ema: Dict[int, float] = defaultdict(float)
        self.step = 0

    def on_pre_optimizer_step(self, args, state, control, **kw):
        model = kw.get("model", None)
        if model is None:
            return
        container, attr, layers, _ = _find_llama_like_layers_container(model)
        if layers is None:
            return
        for li, base in enumerate(layers, 1):
            g2 = 0.0
            for ctrl in (getattr(base, "ctrl_mlp", None), getattr(base, "ctrl_attn", None)):
                if ctrl is None:
                    continue

                params = []
                if hasattr(ctrl, "lora_A") and hasattr(ctrl, "lora_B"):
                    params = [ctrl.lora_A, ctrl.lora_B]
                else:
                    params = [p for p in ctrl.parameters() if p.requires_grad]

                for p in params:
                    if p.grad is not None:
                        g2 += p.grad.pow(2).sum().item()

            g = math.sqrt(g2 + self.eps)
            old = self.g_ema.get(li, g)
            self.g_ema[li] = self.beta * old + (1.0 - self.beta) * g
        self.step += 1


class EvalLossRecorderCallback(TrainerCallback):
    """
    從 on_log 擷取 eval_loss，維護
      - hist: [(step, eval_loss)]
      - delta_loss: 最近一次 eval 與前一次 eval 的差（>0 變差，<0 變好）
      - slope_ma: 移動平均斜率（近 window 計最小二乘線性斜率）
    """
    def __init__(self, window: int = 5):
        self.window = int(max(2, window))
        self.hist: deque = deque(maxlen=64)
        self.delta_loss: Optional[float] = None
        self.slope_ma: Optional[float] = None

    def _compute_slope(self, xs: List[float], ys: List[float]) -> float:
        # 最小二乘直線斜率
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        x = (x - x.mean())
        denom = (x**2).sum()
        if denom <= 0:
            return 0.0
        return float((x * (y - y.mean())).sum() / denom)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "eval_loss" in logs:
            step = int(state.global_step)
            loss = float(logs["eval_loss"])
            if self.hist:
                self.delta_loss = loss - self.hist[-1][1]
            self.hist.append((step, loss))
            # slope over recent window
            if len(self.hist) >= 2:
                k = min(self.window, len(self.hist))
                xs = [self.hist[-k+i][0] for i in range(k)]
                ys = [self.hist[-k+i][1] for i in range(k)]
                self.slope_ma = self._compute_slope(xs, ys)
