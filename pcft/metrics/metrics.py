import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from ..utils.misc import _silence_stdout_stderr


def _effective_rank_from_ctrl(ctrl: nn.Module, eps: float = 1e-12) -> tuple[float, int]:
    matrix_a, matrix_b = ctrl.active_matrices()
    matrix_a = matrix_a.detach().float()
    matrix_b = matrix_b.detach().float()
    # BA and this r x r core have the same non-zero singular values.
    _, triangular_b = torch.linalg.qr(matrix_b, mode="reduced")
    _, triangular_a = torch.linalg.qr(matrix_a.t(), mode="reduced")
    singular_values = torch.linalg.svdvals(triangular_b @ triangular_a.t())
    energy = singular_values.square()
    energy = energy[energy > eps]
    active_rank = int(ctrl.active_rank)
    if energy.numel() == 0:
        return 0.0, active_rank
    probabilities = energy / energy.sum()
    entropy = -(probabilities * torch.log(probabilities + eps)).sum()
    return float(torch.exp(entropy).item()), active_rank


def _safe_import_dadapy():
    try:
        from dadapy.data import Data as DadaData

        return DadaData
    except Exception as error:
        raise RuntimeError(
            "DADApy is required for intrinsic-dimension estimation. "
            "Install it with: python -m pip install dadapy"
        ) from error


def _id_twonN_and_gride(
    states: np.ndarray,
    range_max: int = 64,
    k_lo: int = 8,
    k_hi: int = 32,
) -> Tuple[float, float]:
    rounded = np.round(states.astype(np.float64), 6)
    _, unique_indices = np.unique(rounded, axis=0, return_index=True)
    unique_states = states[sorted(unique_indices)]
    minimum_unique = max(k_hi + 2, 32)
    selected = unique_states if len(unique_states) >= minimum_unique else states
    if len(selected) < 10:
        raise ValueError("Too few states for ID estimation")

    DadaData = _safe_import_dadapy()
    data = DadaData(selected)
    try:
        with _silence_stdout_stderr():
            data.remove_identical_points()
    except Exception:
        pass

    maximum_neighbors = min(max(64, range_max, k_hi + 2), len(selected) - 1)
    data.compute_distances(maxk=maximum_neighbors)
    try:
        if np.any(data.distances[:, 1] == 0.0) or np.any(data.distances[:, 2] == 0.0):
            data.remove_zero_dists(smear=True, noise=1e-8, seed=0)
    except Exception:
        pass

    with _silence_stdout_stderr():
        id_2nn, _, _ = data.compute_id_2NN()
        usable_range = min(int(range_max), maximum_neighbors)
        id_values, _, _ = data.return_id_scaling_gride(range_max=usable_range)
    values = np.asarray(id_values)
    lower = max(1, int(k_lo))
    upper = min(max(lower, int(k_hi)), usable_range)
    selected_values = values[lower - 1 : upper] if values.ndim == 1 else np.asarray([])
    gride_median = float(np.median(selected_values)) if selected_values.size else float(id_2nn)
    if not math.isfinite(gride_median):
        gride_median = float(id_2nn)
    return float(id_2nn), gride_median
