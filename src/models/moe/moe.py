"""
Dispatch strategies for MoE token routing and expert computation.

All strategies share (x, routing, experts) -> output (T, C):

1. dense_all_experts
   Every expert processes every token. Simple reference, but wasteful.

2. dense_masked
   Only selected experts run. Gathers selected tokens with masks and
   scatters results back. Skips unused experts but uses Python loops.

3. sort_and_slice
   Sorts token-expert pairs by expert, then processes contiguous slices.
   Avoids masking and per-expert gathering. Fastest in practice.

All three are mathematically equivalent and should produce matching
outputs and gradients. See benchmarks/check_dispatch_equivalence.py.
"""


import torch

from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import RoutingResult


def dense_all_experts(
    x: torch.Tensor,
    routing: RoutingResult,
    experts: ModuleListExperts,
    top_k: int,
) -> torch.Tensor:
    """Naive reference dispatch: every expert sees every token.

    O(num_experts) full-width matmuls regardless of how sparse the actual
    routing is. Never use this for anything beyond a correctness check or
    a small toy demo -- it defeats the entire point of MoE sparsity.
    """
    T, C = x.shape     # T -> total number of tokens and C--> embedding/hidden dim 

    num_experts = len(experts)

    expert_out_list = [expert(x).unsqueeze(1) for expert in experts.experts]  # each (T, 1, C)
    expert_output = torch.cat(expert_out_list, dim=1)  # (T, num_experts, C)

    weight_full = torch.zeros(T, num_experts, device=x.device, dtype=x.dtype)
    # placing the routing values at the position specified by the index for expert 
    weight_full.scatter_(1, routing.indices, routing.values)  # (T, num_experts) 

    
    out = torch.einsum("te,tec->tc", weight_full, expert_output)
    return out


def dense_masked(
    x: torch.Tensor,
    routing: RoutingResult,
    experts: StackedExperts,
    top_k: int,
) -> torch.Tensor:
    """dispatch: one-hot mask, loop only over hit experts.

    Builds a (num_experts, top_k, T) mask, finds which experts were
    selected by at least one token, and for each such expert gathers its
    tokens, computes, and scatters the weighted result back via
    index_add_. Skips experts with zero tokens but still does an explicit
    gather/scatter per hit expert.
    """
    T, C = x.shape
    num_experts = experts.num_experts
    final_output = torch.zeros(T, C, device=x.device, dtype=x.dtype)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(routing.indices, num_classes=num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (num_experts, top_k, T)
        expert_used = expert_mask.sum(dim=(-1, -2)).nonzero()

    for expert_idx_tensor in expert_used:
        expert_idx = expert_idx_tensor[0].item()
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = x[token_idx]
        current_hidden = experts.forward_one_expert(current_state, expert_idx)
        current_hidden = current_hidden * routing.values[token_idx, top_k_pos, None]
        final_output.index_add_(0, token_idx, current_hidden.to(final_output.dtype))

    return final_output


def sort_and_slice(
    x: torch.Tensor,
    routing: RoutingResult,
    experts: ModuleListExperts,
    top_k: int,
) -> torch.Tensor:
    """Aimed to be Fastest dispatch: global sort by expert id, contiguous per-expert slices.

    Every (token, chosen-expert) pair is flattened into one list, sorted by
    expert id so all of expert i's tokens land in one contiguous block, and
    bincount gives the exact slice boundaries with no further masking. Each
    expert's forward call operates on a tightly packed (count_i, C) tensor
     no wasted computation on tokens the expert doesn't own, no gather
    via boolean indexing (argsort + contiguous slicing is cheaper than
    torch.where-based gathering).

    This is the dispatch paired with the global-LBL MoE variant, since that
    variant is meant to be the fastest end-to-end configuration.
    """
    T, C = x.shape
    num_experts = len(experts)

    flattened_indices = torch.flatten(routing.indices)
    flattened_values = torch.flatten(routing.values)
    token_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    exp_order = torch.argsort(flattened_indices)
    sorted_experts = flattened_indices[exp_order]
    sorted_token_ids = token_ids[exp_order]
    sorted_values = flattened_values[exp_order]

    counts = torch.bincount(sorted_experts, minlength=num_experts)

    final_output = torch.zeros(T, C, device=x.device, dtype=x.dtype)
    start = 0
    for expert_id in range(num_experts):
        count = counts[expert_id].item()
        if count == 0:
            continue
        end = start + count
        tok_ids = sorted_token_ids[start:end]
        weights = sorted_values[start:end]
        expert_out = experts[expert_id](x[tok_ids])
        final_output[tok_ids] += expert_out * weights.unsqueeze(-1)
        start = end

    return final_output


DISPATCH_REGISTRY = {
    "dense_all_experts": dense_all_experts,
    "dense_masked": dense_masked,
    "sort_and_slice": sort_and_slice,
}
