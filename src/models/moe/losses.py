import torch


def return_stats(gate_logits_tuple, num_experts, top_k=2):
    """
    Returns raw per-layer routing statistics for one microstep.

    freq_all_layer:
        hard routing counts, detached/non-differentiable.

    probs_all_layer:
        soft routing mass, graph-attached.
    """
    if gate_logits_tuple is None or len(gate_logits_tuple) == 0:
        return torch.empty(0), torch.empty(0)

    n_layer = len(gate_logits_tuple)
    device = gate_logits_tuple[0].device

    freq_all_layer = torch.zeros(n_layer, num_experts, device=device)
    probs_all_layer = torch.zeros(
        n_layer,
        num_experts,
        device=device,
        dtype=gate_logits_tuple[0].dtype,
    )

    for layer_id, layer_logits in enumerate(gate_logits_tuple):
        layer_probs_full = torch.softmax(layer_logits, dim=-1)
        values, indices = torch.topk(layer_probs_full, top_k, dim=-1)

        flattened_indices = torch.flatten(indices)
        flattened_values = torch.flatten(values)

        counts = torch.bincount(
            flattened_indices,
            minlength=num_experts,
        )
        freq_all_layer[layer_id, :] = counts

        layer_probs = torch.zeros(
            num_experts,
            device=device,
            dtype=values.dtype,
        )
        layer_probs.scatter_add_(
            0,
            flattened_indices,
            flattened_values,
        )
        probs_all_layer[layer_id, :] = layer_probs

    return freq_all_layer, probs_all_layer


def global_lbl(F_i, P_i, N_e, alpha):
    """
    Switch/Qwen-style global load-balancing loss.

    F_i: detached normalized hard-routing fraction.
    P_i: graph-attached normalized soft routing probability.
    """
    assert F_i.shape == P_i.shape
    return alpha * N_e * (F_i * P_i).sum()
