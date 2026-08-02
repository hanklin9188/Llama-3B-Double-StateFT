import math
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SCALING_MODES = {"alpha", "alpha_over_sqrt_rank", "alpha_over_rank"}


class LowRankControl(nn.Module):
    """Ordered prefix-rank residual Control used by the ID-DR supernet."""

    def __init__(
        self,
        hidden_size: int,
        rank_max: int,
        active_rank: int,
        alpha: float,
        dropout: float = 0.05,
        rank_quantum: int = 8,
        scaling_mode: str = "alpha_over_sqrt_rank",
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.rank_max = int(rank_max)
        self.rank_quantum = int(rank_quantum)
        self.alpha = float(alpha)
        self.scaling_mode = str(scaling_mode)
        if self.scaling_mode not in SCALING_MODES:
            raise ValueError(f"Unknown scaling mode: {self.scaling_mode}")
        if self.rank_max % self.rank_quantum:
            raise ValueError("rank_max must be divisible by rank_quantum")

        self.lora_A = nn.Parameter(torch.empty(self.rank_max, self.hidden_size))
        self.lora_B = nn.Parameter(torch.empty(self.hidden_size, self.rank_max))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("active_rank_tensor", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("rank_gates", torch.zeros(self.rank_max))
        block_count = self.rank_max // self.rank_quantum
        self.register_buffer("block_activation_count", torch.zeros(block_count, dtype=torch.long))
        self.register_buffer("block_last_active_step", torch.full((block_count,), -1, dtype=torch.long))
        self._rank_override: Optional[int] = None
        self._transition_target: Optional[torch.Tensor] = None
        self._transition_delta: Optional[torch.Tensor] = None
        self._transition_steps_left = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.set_active_rank(active_rank, step=0)

    def _validate_rank(self, rank: int) -> int:
        rank = int(rank)
        if rank < self.rank_quantum or rank > self.rank_max:
            raise ValueError(f"rank must be in [{self.rank_quantum}, {self.rank_max}]")
        if rank % self.rank_quantum:
            raise ValueError(f"rank must be divisible by {self.rank_quantum}")
        return rank

    @property
    def active_rank(self) -> int:
        return int(self.active_rank_tensor.item())

    @property
    def effective_forward_rank(self) -> int:
        if self._rank_override is not None:
            return self._rank_override
        nonzero = torch.nonzero(self.rank_gates > 0, as_tuple=False)
        return int(nonzero[-1].item() + 1) if nonzero.numel() else self.rank_quantum

    def scaling(self, rank: int) -> float:
        if self.scaling_mode == "alpha":
            return self.alpha
        if self.scaling_mode == "alpha_over_rank":
            return self.alpha / float(rank)
        return self.alpha / math.sqrt(float(rank))

    @torch.no_grad()
    def set_active_rank(self, rank: int, step: int = 0) -> tuple[int, int]:
        rank = self._validate_rank(rank)
        old_rank = self.active_rank
        self.active_rank_tensor.fill_(rank)
        self.rank_gates.zero_()
        self.rank_gates[:rank] = 1.0
        old_blocks = old_rank // self.rank_quantum
        new_blocks = rank // self.rank_quantum
        if new_blocks > old_blocks:
            self.block_activation_count[old_blocks:new_blocks] += 1
            self.block_last_active_step[old_blocks:new_blocks] = int(step)
        self._transition_steps_left = 0
        self._transition_target = None
        self._transition_delta = None
        return old_rank, rank

    @torch.no_grad()
    def begin_rank_transition(self, rank: int, steps: int, step: int = 0):
        rank = self._validate_rank(rank)
        if steps <= 0:
            return self.set_active_rank(rank, step=step)
        old_rank = self.active_rank
        target = torch.zeros_like(self.rank_gates)
        target[:rank] = 1.0
        self.active_rank_tensor.fill_(rank)
        self._transition_target = target
        self._transition_delta = (target - self.rank_gates) / float(steps)
        self._transition_steps_left = int(steps)
        if rank > old_rank:
            lo = old_rank // self.rank_quantum
            hi = rank // self.rank_quantum
            self.block_activation_count[lo:hi] += 1
            self.block_last_active_step[lo:hi] = int(step)
        return old_rank, rank

    @torch.no_grad()
    def advance_transition(self):
        if self._transition_steps_left <= 0:
            return
        self.rank_gates.add_(self._transition_delta)
        self._transition_steps_left -= 1
        if self._transition_steps_left == 0:
            self.rank_gates.copy_(self._transition_target)
            self._transition_target = None
            self._transition_delta = None

    @contextmanager
    def override_rank(self, rank: Optional[int]):
        previous = self._rank_override
        self._rank_override = None if rank is None else self._validate_rank(rank)
        try:
            yield self
        finally:
            self._rank_override = previous

    def active_parameter_count(self, rank: Optional[int] = None) -> int:
        rank = self.active_rank if rank is None else self._validate_rank(rank)
        return 2 * self.hidden_size * rank

    def active_matrices(self, rank: Optional[int] = None):
        rank = self.active_rank if rank is None else self._validate_rank(rank)
        return self.lora_A[:rank], self.lora_B[:, :rank]

    def forward(self, hidden_states: torch.Tensor, active_rank_override: Optional[int] = None):
        rank = active_rank_override or self._rank_override
        if rank is not None:
            rank = self._validate_rank(rank)
            gates = None
        else:
            rank = self.effective_forward_rank
            gates = self.rank_gates[:rank].to(hidden_states.dtype)
        matrix_a = self.lora_A[:rank]
        matrix_b = self.lora_B[:, :rank]
        reduced = F.linear(self.drop(hidden_states), matrix_a)
        if gates is not None:
            reduced = reduced * gates
        return F.linear(reduced, matrix_b) * self.scaling(rank)


# Backward-compatible name used by existing checkpoints and imports.
ControlLoRA = LowRankControl
