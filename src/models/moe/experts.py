"""
Expert weight storage and computation.

Two representations, for different dispatch memory layouts:

  ModuleListExperts: a plain nn.ModuleList of independent FFN modules.
    Needed by strategies that index into an actual list of sub-modules
    (sort_and_slice, dense_masked).

  StackedExperts: all expert weights as single 3D parameter tensors
    (num_experts, ...), like Mixtral. Needed by grouped-GEMM-style
    dispatch, which batch-matmuls across the expert dimension instead
    of looping python-side over nn.Linear calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single FFN expert: up-projection -> GELU -> down-projection.

    bias=False on both linears to match StackedExperts (Mixtral-style,
    no bias). Keeping both representations bias-free keeps dispatch
    strategies numerically comparable, see
    benchmarks/check_dispatch_equivalence.py.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gelu = nn.GELU(approximate="tanh")
        self.f_u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.f_d = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_d(self.gelu(self.f_u(x)))


class ModuleListExperts(nn.Module):
    """A list of independent Expert modules, indexed by expert id."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_experts: int) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)]
        )
        self.num_experts = num_experts

    def __getitem__(self, idx: int) -> Expert:
        return self.experts[idx]

    def __len__(self) -> int:
        return self.num_experts


class StackedExperts(nn.Module):
    """All expert weights as single 3D tensors: (num_experts, ...).
    Uses SwiGLU (GLU Variants Improve Transformer: https://arxiv.org/pdf/2002.05202).

      swiglu=True: gate_up_proj packs gate and up projections together
        (2 * hidden_dim), one matmul per expert produces both halves via
        .chunk(2, dim=-1). forward_one_expert computes GELU(gate) * up @ down.

      swiglu=False: plain FFN, GELU(up(x)) @ down, same function as
        experts.Expert, just stored as stacked tensors. Exists so
        dense_masked can be checked for numerical equivalence against
        dense_all_experts and sort_and_slice (both plain-FFN Expert).
        With swiglu=True the two architectures compute different
        functions and are not expected to match.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        swiglu: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.swiglu = swiglu

        up_width = 2 * hidden_dim if swiglu else hidden_dim
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, up_width, input_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, output_dim, hidden_dim))
        self.gelu = nn.GELU(approximate="tanh")

        self._init_weights()

    def _init_weights(self) -> None:
        for e in range(self.num_experts):
            nn.init.normal_(self.gate_up_proj[e], mean=0.0, std=0.02)
            nn.init.normal_(self.down_proj[e], mean=0.0, std=0.02)

    def forward_one_expert(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Apply a single expert (by id) to a batch of tokens already
        gathered for that expert. Used by dispatch strategies that loop
        python-side over which experts were actually hit.
        """
        if self.swiglu:
            gate, up = F.linear(x, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            hidden = self.gelu(gate) * up
        else:
            hidden = self.gelu(F.linear(x, self.gate_up_proj[expert_idx]))
        return F.linear(hidden, self.down_proj[expert_idx])