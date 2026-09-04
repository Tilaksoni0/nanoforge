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

**LATEST_UPDATE** : Added optional use_segmentation version(defaults to True) motivated from the paper Deepseek Moe 
-- Segmenting available expert ,into finer expert shows better expert specialisation(dai. et al. 2024) as it helps to figh 
knowldge hybridity as discussed in the paper . 
-- keeping a pool of shared expert which is routed to every token(hard choice dose'nt pass thorugh the router)
helps it fight knowledge redundancy 
Note: we dont let shared experts contribute in loss which kills the reason why its is there for 

"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.moe.dispatch import DEFAULT_BUCKET, DISPATCH_REGISTRY
from src.models.moe.experts import  ModuleListExperts, StackedExperts,Expert
from src.models.moe.router import GatingNetwork, RoutingResult, route

# Dispatch strategies that need the batched-matmul (StackedExperts) layout.
# Everything else uses the python-loop (ModuleListExperts) layout.
_STACKED_EXPERT_STRATEGIES = {"dense_masked", "sort_and_pad", "sort_pad_bucket"}


@dataclass
class MOEConfig:
    n_embd: int
    n_experts: int
    top_k: int
    granularity_factor: int  
    num_shared_expert: int  
    dispatch: str = "sort_pad_bucket"
    alpha_moe: float = 0.01
    hidden_dim: int | None = None  # defaults to 4 * n_embd, see __post_init__
    capacity_factor: float = 1.25  # used by sort_and_pad
    bucket: list = field(default_factory=lambda: list(DEFAULT_BUCKET))  # used by sort_pad_bucket
    swiglu: bool = True
    use_segmentation: bool = True 
    use_lossFreeBalancing : bool = False

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
        self.use_segmentation = config.use_segmentation
        self.granularity_factor = config.granularity_factor
        self.num_shared_expert = config.num_shared_expert
        self.use_lossFreeBalancing = config.use_lossFreeBalancing

        

  

        self.top_k = (config.top_k -config.num_shared_expert) if config.use_segmentation  else config.top_k #


        if config.dispatch not in DISPATCH_REGISTRY:
            raise ValueError(
                f"Unknown dispatch strategy {config.dispatch!r}. "
                f"Available: {sorted(DISPATCH_REGISTRY.keys())}"
            )
        
        self.dispatch_fn = DISPATCH_REGISTRY[config.dispatch]
        self._uses_stacked = config.dispatch in _STACKED_EXPERT_STRATEGIES

        self.gate = GatingNetwork(config.n_embd, config.n_experts)
        #if segmentation then hidden dim becomes
        #  h_dim ---> h_dim/m     -(m being granularity factor)

        # shared_experts
        if self.use_segmentation:              
            self.shared_expert_stacked = StackedExperts(
                  config.n_embd, 
                  (config.hidden_dim//self.granularity_factor),
                  config.n_embd,
                  self.num_shared_expert,
                  swiglu=config.swiglu,
                  )
        
        #if segmentation then num_routing experts
        # N ---> mN - ks       -(ks is the num_shared_expert)
        #routing_experts    
        if self._uses_stacked:
            self.experts = StackedExperts(
                config.n_embd,
                (config.hidden_dim//self.granularity_factor) if self.use_segmentation else config.hidden_dim,
                config.n_embd,
                (self.granularity_factor*config.n_experts - config.num_shared_expert) if self.use_segmentation else config.n_experts ,
                swiglu=config.swiglu,
            )
        else:
            self.experts = ModuleListExperts(
                config.n_embd,
                (config.hidden_dim//self.granularity_factor) if self.use_segmentation else config.hidden_dim,
                config.n_embd,
                (self.granularity_factor*config.n_experts - config.num_shared_expert) if self.use_segmentation else config.n_experts,
                swiglu=config.swiglu,
            )

    def forward(self, x: torch.Tensor, expert_biases = None) -> tuple[torch.Tensor, RoutingResult]:
        """x: (B, T, C). Flattened to (T, C) for routing/dispatch (dispatch's
        concern per router.py's docstring), reshaped back before return.
        """
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        if self.use_lossFreeBalancing: 
            if expert_biases is None : 
                raise ValueError(
                                f"!lossFreeBalancing is {self.use_lossFreeBalancing!r}. "
                                f"lossFreeBalancing requires expert biases, cannot be None"
                            )

        routing = route(x_flat,  self.gate,self.top_k, self.use_lossFreeBalancing ,expert_biases)

        kwargs = {}
        if self.config.dispatch == "sort_and_pad":
            kwargs["capacity_factor"] = self.config.capacity_factor
        elif self.config.dispatch == "sort_pad_bucket":
            kwargs["BUCKET"] = self.config.bucket

        out_flat = self.dispatch_fn(x_flat, routing, self.experts, self.top_k, **kwargs)
        out = out_flat.reshape(B, T, C)

        if self.use_segmentation: 
            out+= (
                (
                    self.shared_expert_stacked.forward_batched(x_flat)
                ).sum(0)).reshape(B,T,C)


        return out, routing


# Backwards-compat alias: some earlier code/notes refer to this class in
# lowercase. Keep both names pointing at the same class so nothing importing
# either spelling breaks.
MoE = MOE