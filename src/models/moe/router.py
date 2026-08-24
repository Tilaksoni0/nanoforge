"""
Router: gate logits -> probabilities -> top-k expert selection.
Handles routing only. Dispatch handles token grouping and losses.py handles
loss accumulation.
"""


import torch
import torch.nn as nn
from dataclasses import dataclass


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return self.gate(x)
@dataclass
class RoutingResult:
    """Output of a full routing pass for one layer, one forward call.

    logits:  (T, num_experts) raw gate logits, graph-attached
    probs:   (T, num_experts) softmax over all experts, graph-attached
    values:  (T, top_k) softmax probability of each chosen expert, graph-attached
    indices: (T, top_k) chosen expert ids, no grad (indices are not differentiable)
    """

    logits: torch.Tensor
    probs: torch.Tensor
    values: torch.Tensor
    indices: torch.Tensor


def route(
    x: torch.Tensor,
    gate: GatingNetwork,
    top_k: int,
) -> RoutingResult:
    """Run the gate and select top_k experts per token.

    x is expected already flattened to (T, C) flattening B,T into one
    token axis is dispatch's concern (different dispatch strategies want
    different shapes), not the router's.
    """
    logits = gate(x)
    probs = torch.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, top_k, dim=-1)
    return RoutingResult(logits=logits, probs=probs, values=values, indices=indices)


def extract_stats(
    routing_results: tuple[RoutingResult, ...],
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-layer RoutingResult into frequency/probability tensors.

    freq_all_layer:  (n_layer, num_experts) hard selection counts, no grad.
                    Raw increment fed into the global LBL buffer.

    probs_all_layer: (n_layer, num_experts) summed soft routing mass, graph-attached.
                    Lets gradient flow into the router via the aux loss
                    (freq has no grad, bincount, only probs does).

    Lives here, not losses.py, since it only touches RoutingResult; losses.py
    owns accumulation/loss math, not extraction from logits. Callers decide
    whether to invoke this at all (see dispatch.py); strategies that don't
    need global LBL stats (e.g. pure inference) can skip it.
    """
    if routing_results is None or len(routing_results) == 0:
        return torch.empty(0), torch.empty(0)

    n_layer = len(routing_results)
    device = routing_results[0].indices.device

    freq_all_layer = torch.zeros(n_layer, num_experts, device=device)
    probs_all_layer = torch.zeros(
        n_layer, num_experts, device=device, dtype=routing_results[0].values.dtype
    )

    for layer_id, result in enumerate(routing_results):
        flattened_indices = torch.flatten(result.indices)
        flattened_values = torch.flatten(result.values)

        counts = torch.bincount(flattened_indices, minlength=num_experts)
        freq_all_layer[layer_id, :] = counts

        layer_probs = torch.zeros(num_experts, device=device, dtype=result.values.dtype)
        layer_probs.scatter_add_(0, flattened_indices, flattened_values)
        probs_all_layer[layer_id, :] = layer_probs

    return freq_all_layer, probs_all_layer
