import torch
import torch.nn as nn

from .experts import Expert
from .router import GatingNetwork


class MoE(nn.Module):
    """
    Current top-k sparse MoE implementation.

    using a slice and cut forward instead of passing evert token to every expert 
    (ie. expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ] is wasteful
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList(
            [
                Expert(
                    config.n_embd,
                    4 * config.n_embd,
                    config.n_embd,
                )
                for _ in range(config.n_experts)
            ]
        )

        self.gate = GatingNetwork(
            config.n_embd,
            config.n_experts,
        )

        self.track_experts = []

    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        T_total = B * T

        gate_logits = self.gate(x)
        probs = torch.softmax(gate_logits, dim=-1)

        values, indices = torch.topk(
            probs,
            k=self.config.top_k,
            dim=-1,
        )

        self.track_experts.append(indices.detach())

        flattened_indices = torch.flatten(indices)
        flattened_values = torch.flatten(values)

        token_ids = (
            torch.arange(T_total, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.config.top_k)
            .reshape(-1)
        )

        exp_order = torch.argsort(flattened_indices)

        sorted_experts = flattened_indices[exp_order]
        sorted_token_ids = token_ids[exp_order]
        sorted_values = flattened_values[exp_order]

        counts = torch.bincount(
            sorted_experts,
            minlength=self.config.n_experts,
        )

        final_output = torch.zeros(
            T_total,
            C,
            device=x.device,
            dtype=x.dtype,
        )

        start = 0

        for exp_id, expert in enumerate(self.experts):
            count = counts[exp_id].item()

            if count == 0:
                continue

            end = start + count

            tok_ids = sorted_token_ids[start:end]
            weights = sorted_values[start:end]

            expert_out = expert(x[tok_ids])
            final_output[tok_ids] += (
                expert_out * weights.unsqueeze(-1)
            )

            start = end

        return final_output.reshape(B, T, C), gate_logits
