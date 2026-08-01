import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlLoRA(nn.Module):
    """Low-rank residual control: delta(x) = alpha * B(A(dropout(x)))."""

    def __init__(self, hidden_size: int, rank: int, alpha: float, dropout: float = 0.05):
        super().__init__()
        self.d = int(hidden_size)
        self.r = int(rank)
        self.alpha_scalar = float(alpha)
        self.lora_A = nn.Parameter(torch.empty(self.r, self.d))
        self.lora_B = nn.Parameter(torch.empty(self.d, self.r))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("rank_mask", torch.ones(self.r))
        self._active_rank = self.r
        self._last_g_mean = None
        self._last_g_var = None
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        reduced = F.linear(self.drop(hidden_states), self.lora_A)
        reduced = reduced * self.rank_mask.to(reduced.dtype)
        return F.linear(reduced, self.lora_B) * self.alpha_scalar

    @property
    def active_rank(self) -> int:
        return int(self.rank_mask.sum().item())

    @torch.no_grad()
    def set_active_rank(self, rank: int, importance: Optional[torch.Tensor] = None):
        rank = max(1, min(self.r, int(rank)))
        mask = torch.zeros(self.r, device=self.rank_mask.device)
        if importance is None:
            mask[:rank] = 1.0
        else:
            indices = torch.topk(importance.to(mask.device), k=rank).indices
            mask[indices] = 1.0
        self.rank_mask.copy_(mask)
        self._active_rank = rank
