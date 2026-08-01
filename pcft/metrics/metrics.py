from ..common_imports import *
from ..adapters.control import ControlLoRA
from ..utils.misc import _silence_stdout_stderr

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    J = I - torch.full((n, n), 1.0 / n, device=K.device, dtype=K.dtype)
    return J @ K @ J


def _gram_rbf(X: torch.Tensor, sigma: Optional[float] = None, center: bool = True, eps: float = 1e-8) -> torch.Tensor:
    # X: (N, D) float32/float16/bfloat16
    X = X.to(torch.float32)
    N = X.size(0)
    d2 = torch.cdist(X, X, p=2.0) ** 2  # (N, N)
    if sigma is None or sigma <= 0:
        # median heuristic（排除對角）
        med = torch.median(d2[~torch.eye(N, dtype=torch.bool, device=d2.device)])
        med = med.clamp_min(eps)
        # 常見做法：σ² = med；你也可改成 med / log(N+1)
        sigma2 = med
    else:
        sigma2 = torch.tensor(float(sigma) ** 2, device=d2.device, dtype=d2.dtype)
    K = torch.exp(-d2 / (2.0 * sigma2 + eps))
    if center:
        K = _center_gram(K)
    # 讓 trace=1，並做對稱化避免數值小誤差
    K = 0.5 * (K + K.t())
    tr = torch.trace(K).clamp_min(eps)
    A = K / tr
    # 強化數值穩定
    A = A + eps * torch.eye(N, device=A.device, dtype=A.dtype)
    A = 0.5 * (A + A.t())
    return A


def _matrix_based_entropy(X: torch.Tensor, alpha: float = 2.0, sigma: Optional[float] = None,
                          center: bool = True, eps: float = 1e-8, log_base: float = 2.0) -> float:
    """
    H_α(X) = (1/(1-α)) * log_{base} ( trace( Â^α ) ),  其中 Â = K / trace(K)
    等價於把 Â 的特徵值 λ_i 視為「機率」：trace(Â^α)=∑ λ_i^α。
    """
    A = _gram_rbf(X, sigma=sigma, center=center, eps=eps)  # (N,N), trace=1
    # 用對稱特徵分解（PSD → eigvalsh）
    w = torch.linalg.eigvalsh(A)  # (N,)
    w = w.clamp_min(0.0)
    w = w / (w.sum().clamp_min(eps))  # 安全歸一化
    if abs(alpha - 1.0) < 1e-6:
        # Shannon 極限：-∑ p log p
        H = -(w * torch.log(w.clamp_min(eps))).sum() / math.log(log_base)
    else:
        H = (torch.log((w ** alpha).sum().clamp_min(eps)) / math.log(log_base)) / (1.0 - alpha)
    return float(H.item())


def _effective_rank_from_ctrl(ctrl: nn.Module, eps: float = 1e-12) -> tuple[float, int]:
    A = ctrl.lora_A.detach().to(torch.float32)
    B = ctrl.lora_B.detach().to(torch.float32)
    m = ctrl.rank_mask.detach().to(A.device, dtype=A.dtype)
    A = A * m.view(-1, 1)
    B = B * m.view(1, -1)
    # BA and this r x r core have the same non-zero singular values.
    _, r_b = torch.linalg.qr(B, mode="reduced")
    _, r_a = torch.linalg.qr(A.t(), mode="reduced")
    C = r_b @ r_a.t()
    act = int(ctrl.active_rank)
    s = torch.linalg.svdvals(C)


    e = s.square(); e = e[e > eps]
    if e.numel() == 0:
        return 0.0, act
    p = e / e.sum()
    H = -(p * torch.log(p + eps)).sum()
    return float(torch.exp(H).item()), act


def _last_non_eos_index(input_ids: torch.Tensor, attention_mask: torch.Tensor, eos_id: int) -> torch.Tensor:
    # input_ids / attention_mask: (B, T)
    # 回傳每個樣本最後一個 (mask==1 且 token!=eos) 的位置；若找不到就退回最後一個有效位置
    B, T = input_ids.size()
    is_valid = (attention_mask > 0) & (input_ids != eos_id)
    # 轉成索引：把無效位置設為 -1，取每行最大索引
    idx = torch.arange(T, device=input_ids.device).view(1, T).expand(B, T)
    masked_idx = torch.where(is_valid, idx, torch.full_like(idx, -1))
    last_idx = masked_idx.max(dim=1).values
    # 若整行沒有非 EOS，就退回原本最後有效位
    fallback = attention_mask.long().sum(dim=1) - 1
    last_idx = torch.where(last_idx >= 0, last_idx, fallback)
    return last_idx


def _gather_last_token(h: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
    return h[torch.arange(h.size(0), device=h.device), last_idx]  # (B, D)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().to("cpu").float().numpy()


def _compute_pairwise_ranks(x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        dist = torch.cdist(x, x)  # (N, N)
        order = dist.argsort(dim=1, stable=True)
        inv = torch.empty_like(order)
        inv.scatter_(1, order, torch.arange(1, order.size(1)+1, device=order.device).unsqueeze(0).expand_as(order))
        return inv  # (N, N), 1..N


def _delta_a_to_b(x_a: torch.Tensor, x_b: torch.Tensor) -> float:
    with torch.no_grad():
        N = x_a.size(0)
        dist_a = torch.cdist(x_a, x_a)
        dist_a.fill_diagonal_(float("inf"))
        nn_j = dist_a.argmin(dim=1)  # (N,)
        ranks_b = _compute_pairwise_ranks(x_b)  # 1..N
        r = ranks_b[torch.arange(N, device=x_b.device), nn_j]  # 1..N
        r_norm = (r - 1).float() / (N - 1)
        return float(r_norm.mean().item())


def _safe_import_dadapy():
    try:
        from dadapy.data import Data as DadaData
        return DadaData
    except Exception as e:
        raise RuntimeError(
            "DADApy 尚未安裝或無法匯入。請先安裝： pip install dadapy\n"
            f"原始錯誤：{e}"
        )


def _id_twonN_and_gride(
    X_np: np.ndarray,
    range_max: int = 64,
    k_lo: int = 8,
    k_hi: int = 32,
) -> Tuple[float, float]:
    DadaData = _safe_import_dadapy()

    # 1) 先做「近似去重」：量化到 1e-6 再 unique，避免浮點微差
    q = np.round(X_np.astype(np.float64), 6)
    _, uniq_idx = np.unique(q, axis=0, return_index=True)
    Xu = X_np[sorted(uniq_idx)]
    if Xu.shape[0] < X_np.shape[0]:
        # 去重之後如果樣本太少，直接回退到原資料以防崩
        X_use = Xu if Xu.shape[0] >= max(k_hi + 2, 32) else X_np
    else:
        X_use = X_np

    data = DadaData(X_use)

    # 2) 去除完全相同點（dadapy 自帶）
    try:
        with _silence_stdout_stderr():
            data.remove_identical_points()  # 基於原始座標
    except Exception:
        pass

    # 3) 計算距離，之後如仍發現 0 距離，則 smear
    maxk = max(64, range_max, k_hi + 2)
    data.compute_distances(maxk=maxk)
    try:
        # 如果第 1、2 近鄰仍有 0 距離就 smear（加極小隨機噪聲）
        if np.any(data.distances[:, 1] == 0.0) or np.any(data.distances[:, 2] == 0.0):
            data.remove_zero_dists(smear=True, noise=1e-8, seed=0)
    except Exception:
        pass

    # 4) 正常計算
    id2, _, _ = data.compute_id_2NN()
    id_list, _, _ = data.return_id_scaling_gride(range_max=range_max)

    k_lo = max(1, int(k_lo))
    k_hi = min(max(k_lo, int(k_hi)), int(range_max))
    id_arr = np.asarray(id_list)
    if id_arr.ndim == 1:
        sel = id_arr[k_lo-1:k_hi]
        id_med = float(np.median(sel)) if sel.size > 0 else float(id2)
    else:
        id_med = float(id2)
    return float(id2), float(id_med)
