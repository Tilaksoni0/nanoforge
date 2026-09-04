"""
dispatch.py — token routing strategies for MoE layers.

Four strategies: dense_all_experts (reference/oracle), dense_masked
(Mixtral-style), sort_and_slice (fastest, used with global-LBL), and
sort_and_pad (capacity-bounded batched matmul, may drop tokens).

All four share the signature (x, routing, experts, top_k) -> (T, C),
except sort_and_pad which requires an extra `capacity` argument.
See DISPATCH_REGISTRY at the bottom to swap strategies by name.
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

def sort_and_pad(
    x: torch.Tensor,
    routing: RoutingResult,
    experts: StackedExperts,
    top_k: int,
    capacity_factor: float = 1.25, # fix: instead passing capcity pass capacity factor
) -> torch.Tensor:
    """Sort + fixed capacity buffer, single batched matmul across all experts.

    Same sort step as sort_and_slice, but each expert's slice is padded or
    truncated to `capacity` tokens, enabling one bmm over the full
    (num_experts, capacity, C) buffer instead of a Python loop.
    Tokens beyond capacity for their expert are silently dropped.
    Only numerically equivalent to the other strategies when capacity is large
    enough that no token is dropped for the given input.

    capacity = round((T * top_k / num_experts) * capacity_factor), same
    idea as Switch Transformer: one shared value for every expert, not
    measured from this batch's actual load. Tokens past capacity for
    their expert get dropped. Want dropless? see sort_pad_bucket
    (bucketed capacity) or sort_and_slice (no padding at all).
    """
    T, C = x.shape
    N = T * top_k
    num_experts = len(experts)
    assert capacity_factor > 0, "capacity_factor must be positive" # changed from capcity to capcity_factor
    assert routing.indices.shape == (T, top_k), routing.indices.shape

    capacity = round((N / num_experts) * capacity_factor) # calculate capacity

    flattened_indices = torch.flatten(routing.indices)
    flattened_values = torch.flatten(routing.values)
    token_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    exp_order = torch.argsort(flattened_indices, stable = True) # edit: stable = True , making it future token leakage proof. 
    sorted_experts = flattened_indices[exp_order]
    sorted_token_ids = token_ids[exp_order]
    sorted_values = flattened_values[exp_order]

    group_sizes = torch.bincount(sorted_experts, minlength=num_experts)
    group_starts = torch.cumsum(group_sizes, dim=0) - group_sizes
    local_rank = torch.arange(N, device=x.device) - torch.repeat_interleave(group_starts, group_sizes)

    keep_mask = local_rank < capacity
    kept_dest = sorted_experts[keep_mask] * capacity + local_rank[keep_mask]
    kept_token_ids = sorted_token_ids[keep_mask]

    padded = torch.zeros(num_experts * capacity, C, device=x.device, dtype=x.dtype)
    padded[kept_dest] = x[kept_token_ids]

    hidden = experts.forward_batched(padded.view(num_experts, capacity, C))
    out_flat = hidden.reshape(num_experts * capacity, -1)

    weighted = out_flat[kept_dest] * sorted_values[keep_mask].unsqueeze(-1)

    final_output = torch.zeros(T, C, device=x.device, dtype=x.dtype)
    final_output.scatter_add_(
        0, kept_token_ids.unsqueeze(-1).expand(-1, C), weighted.to(final_output.dtype)
    )
    return final_output

def sort_pad_bucket(
    x: torch.Tensor,
    routing: RoutingResult,
    experts: StackedExperts,
    top_k: int,
    BUCKET: list = None,  
) -> torch.Tensor:
    """This func works mostly similar but with a slight optimisation, aimed to be dropless.

    Optimisation: sort_and_pad above is not aware of what capacity its
    receiving. It may be greater then the max group_size, which leads
    to a wasteful amount of padding, or it can be smaller then some of
    the group sizes, which leads to dropping tokens, which is not a
    desired thing. So instead of passing a hard capacity choice, we let
    the batch choose (we could have done max(group_size) every time, but
    this would have a lot of different shapes every time which are bad
    for hardware efficiency, so instead there is a fixed number of sizes
    we can pick from, this saves the caches and pytorch need not to
    recompile every time).

    MoEConfig defines a default BUCKET list, and it is optional, one can
    pass his own. If the batch's max group_size ends up bigger then every
    bucket in the list, this will raise, so the largest bucket should
    always cover the biggest batch this is ever run on.
    """
    
    if BUCKET is None:
        BUCKET = MoEConfig.BUCKET

    T, C = x.shape
    N = T * top_k
    num_experts = len(experts)
    assert len(BUCKET) > 0, "BUCKET must be non-empty" 
    assert routing.indices.shape == (T, top_k), routing.indices.shape

    flattened_indices = torch.flatten(routing.indices)
    flattened_values = torch.flatten(routing.values)
    token_ids = torch.arange(T, device=x.device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    exp_order = torch.argsort(flattened_indices)
    sorted_experts = flattened_indices[exp_order]
    sorted_token_ids = token_ids[exp_order]
    sorted_values = flattened_values[exp_order]

    group_sizes = torch.bincount(sorted_experts, minlength=num_experts)
    max_group_size = torch.max(group_sizes).item()
    # NOTE The largest value of capcity must be >= T(Total_tokens) as if the worst case happens and we have the max(bucket) < the worst casr 
    # the next would stop iteration
    capacity = next(b for b in BUCKET if max_group_size <= b)
    group_starts = torch.cumsum(group_sizes, dim=0) - group_sizes

    local_rank = torch.arange(N, device=x.device) - torch.repeat_interleave(group_starts, group_sizes)

    keep_mask = local_rank < capacity
    kept_dest = sorted_experts[keep_mask] * capacity + local_rank[keep_mask]
    kept_token_ids = sorted_token_ids[keep_mask]

    padded = torch.zeros(num_experts * capacity, C, device=x.device, dtype=x.dtype)
    padded[kept_dest] = x[kept_token_ids]

    hidden = experts.forward_batched(padded.view(num_experts, capacity, C))
    out_flat = hidden.reshape(num_experts * capacity, -1)

    weighted = out_flat[kept_dest] * sorted_values[keep_mask].unsqueeze(-1)

    final_output = torch.zeros(T, C, device=x.device, dtype=x.dtype)
    final_output.scatter_add_(
        0, kept_token_ids.unsqueeze(-1).expand(-1, C), weighted.to(final_output.dtype)
    )
    return final_output


DISPATCH_REGISTRY = {
    "dense_all_experts": dense_all_experts,
    "dense_masked": dense_masked,
    "sort_and_slice": sort_and_slice,
    "sort_and_pad":      sort_and_pad,
    "sort_pad_bucket": sort_pad_bucket,
}
