"""
MoE: the nn.Module that wires router.py + experts.py + dispatch.py together.

This file does NOT implement dispatch strategies itself -- those live in
dispatch.py's DISPATCH_REGISTRY. This file owns:
  - MOEConfig: the config dataclass gpt.py builds and passes in.
  - MOE: holds the gating network, both expert-container layouts
    (ModuleListExperts / StackedExperts), and picks whichever container
    the selected dispatch strategy needs, then calls that strategy from
    the registry.

Strategy choice is a config knob (MOEConfig.dispatch), so any registered
dispatch strategy in dispatch.DISPATCH_REGISTRY can be selected without
touching this file. Default is "sort_pad_bucket".
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.moe.dispatch import DEFAULT_BUCKET, DISPATCH_REGISTRY
from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import GatingNetwork, RoutingResult, route

# Dispatch strategies that need the batched-matmul (StackedExperts) layout.
# Everything else uses the python-loop (ModuleListExperts) layout.
_STACKED_EXPERT_STRATEGIES = {"dense_masked", "sort_and_pad", "sort_pad_bucket"}


@dataclass
class MOEConfig:
    n_embd: int
    n_experts: int
    top_k: int
    dispatch: str = "sort_pad_bucket"
    alpha_moe: float = 0.01
    hidden_dim: int | None = None  # defaults to 4 * n_embd, see __post_init__
    capacity_factor: float = 1.25  # used by sort_and_pad
    bucket: list = field(default_factory=lambda: list(DEFAULT_BUCKET))  # used by sort_pad_bucket
    swiglu: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.n_embd


class MOE(nn.Module):
    """MoE feed-forward sublayer. Router -> dispatch(config.dispatch) -> output.

    Builds both expert container layouts (ModuleListExperts, StackedExperts)
    sharing the same underlying weights are NOT tied between the two --
    each strategy needs its own container because the tensor layouts differ
    (list of nn.Linear vs. one stacked 3D parameter). Whichever container the
    selected strategy needs is the one actually used in forward(); the other
    sits unused unless you switch config.dispatch at runtime.
    """

    def __init__(self, config: MOEConfig) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.top_k

        if config.dispatch not in DISPATCH_REGISTRY:
            raise ValueError(
                f"Unknown dispatch strategy {config.dispatch!r}. "
                f"Available: {sorted(DISPATCH_REGISTRY.keys())}"
            )
        self.dispatch_fn = DISPATCH_REGISTRY[config.dispatch]
        self._uses_stacked = config.dispatch in _STACKED_EXPERT_STRATEGIES

        self.gate = GatingNetwork(config.n_embd, config.n_experts)

        if self._uses_stacked:
            self.experts = StackedExperts(
                config.n_embd,
                config.hidden_dim,
                config.n_embd,
                config.n_experts,
                swiglu=config.swiglu,
            )
        else:
            self.experts = ModuleListExperts(
                config.n_embd,
                config.hidden_dim,
                config.n_embd,
                config.n_experts,
                swiglu=config.swiglu,
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingResult]:
        """x: (B, T, C). Flattened to (T, C) for routing/dispatch (dispatch's
        concern per router.py's docstring), reshaped back before return.
        """
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)

        routing = route(x_flat, self.gate, self.top_k)

        kwargs = {}
        if self.config.dispatch == "sort_and_pad":
            kwargs["capacity_factor"] = self.config.capacity_factor
        elif self.config.dispatch == "sort_pad_bucket":
            kwargs["BUCKET"] = self.config.bucket

        out_flat = self.dispatch_fn(x_flat, routing, self.experts, self.top_k, **kwargs)
        out = out_flat.reshape(B, T, C)
        return out, routing


# Backwards-compat alias: some earlier code/notes refer to this class in
# lowercase. Keep both names pointing at the same class so nothing importing
# either spelling breaks.
MoE = MOE
