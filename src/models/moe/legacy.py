"""
Legacy: Shazeer et al. (2017) noisy top-k MoE.
https://arxiv.org/abs/1701.06538

Two versions were built and tested before landing here:
  1. retain_graph version - backward(retain_graph=True) every microstep,
     one final backward on L_importance+L_load at the end.
  2. two-pass version (kept below) - a no-grad pass first to get true
     accumulated importance/load stats, compute d(CV^2)/d(totals) on an
     isolated graph, then a live second pass injecting that upstream
     gradient per microstep and backwarding immediately. No retain_graph.

Not wired into the current router/dispatch/losses pipeline.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class ShazeerConfig:
    n_embd: int = 32
    n_experts: int = 4
    topk: int = 2
    w_importance: float = 0.01
    w_load: float = 0.01


class ShazeerMLP(nn.Module):
    def __init__(self, config: ShazeerConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class ShazeerMOE(nn.Module):
    def __init__(self, config: ShazeerConfig) -> None:
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([ShazeerMLP(config) for _ in range(config.n_experts)])
        self.gate = nn.Linear(config.n_embd, config.n_experts)
        self.noise = nn.Linear(config.n_embd, config.n_experts)
        self.track_experts = []

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        gate_values = self.gate(x)
        raw_noise = self.noise(x)
        softplus = F.softplus(raw_noise) + 1e-4
        noise = torch.randn(B, T, self.config.n_experts, device=x.device, dtype=x.dtype) * softplus
        H_x = gate_values + noise

        values, indices = torch.topk(H_x, self.config.topk, dim=-1, largest=True, sorted=True)
        if not torch.is_grad_enabled():
            self.track_experts.append(indices.detach())

        full_H = torch.full_like(H_x, float("-inf"))
        full_H.scatter_(-1, indices,values)  # fixed: unnessacary compute
        G_x = F.softmax(full_H, dim=-1)

        importance = G_x.sum(dim=1)

        loads = gate_values.clone()
        for i in range(self.config.n_experts):
            temp_x = H_x.clone()
            temp_x[..., i] = float("-inf")
            kth_values, _ = torch.topk(temp_x, self.config.topk, dim=-1, largest=True, sorted=True)
            kth_excluding_i = kth_values[..., -1]
            loads[..., i] = (gate_values[..., i] - kth_excluding_i) / softplus[..., i]

        normal = Normal(loc=0, scale=1)
        P = normal.cdf(loads)
        Load = P.sum(dim=1)

        expert_outputs = torch.stack(
            [self.experts[i](x) for i in range(self.config.n_experts)], dim=-1
        )
        out = (expert_outputs * G_x.unsqueeze(-2)).sum(dim=-1)

        return out, importance, Load

# Implmented a global no-approximation version of load balance instead 
"""
Before reading any paper about load-balancing techniques, I came up with my own approach to try to match the way papers define load balancing.

In the papers I had read, nothing was specifically defined for the case where gradient accumulation is used.
So, I thought the batch size should still be treated as the whole batch. Instead of doing loss.backward() on each micro-batch, we should therefore do it on the whole batch.

My tries:

1. Retaining the graph:
    I first tried retaining the computation graph across all gradient accumulation steps. However, this caused a major memory problem,
    as the entire autograd graph had to remain alive on the GPU until the final accumulation step.
    This made the approach very inefficient in practice.
2. Two-pass technique:
    I then tried to solve the memory bottleneck with a more practical approach.
    I implemented a two-pass technique where the first pass creates the computation graph exclusively for the load-balancing loss across the accumulated micro-batches, 
    without including the language-model loss. The second pass then connects this load-balancing loss to the ongoing loss through the chain rule.
    One important requirement is that both passes must use the same seed before starting, so that the routing results and everything dependent on them remain identical.
    Problem with this: This is also very inefficient, since performing two complete forward passes over the whole model is extremely expensive.

Edit: I found a paper that relates directly to this problem. It states that performing loss.backward() at every gradient accumulation step can degrade model performance, and proposes an approximate global load-balancing algorithm.

Paper: Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
https://arxiv.org/abs/2501.11873
"""
def run_two_pass_step(
    model,
    train_loader,
    optimizer,
    num_accum_steps: int,
    device,
    device_type: str,
    autocast_dtype,
):
    """One optimizer step, two-pass gradient injection.

    Pass 1 (no_grad): accumulate true full-batch importance/load stats.
    Compute upstream gradient of CV^2 w.r.t. those totals on an isolated graph.
    Pass 2 (live graph): recompute per microstep, inject upstream grad, backward immediately.
    """
    optimizer.zero_grad()

    accumulated_importance = [
        torch.zeros(model.config.n_experts, device=device) for _ in range(len(model.blocks))
    ]
    accumulated_load = [
        torch.zeros(model.config.n_experts, device=device) for _ in range(len(model.blocks))
    ]

    saved_position = train_loader.current_position

    with torch.no_grad():
        for _ in range(num_accum_steps):
            xb, yb = train_loader.next_batch()
            x, y = xb.to(device), yb.to(device)
            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                _, _, importance_list, load_list = model(x, y)
            for layer_idx in range(len(model.blocks)):
                accumulated_importance[layer_idx] += importance_list[layer_idx].sum(dim=0)
                accumulated_load[layer_idx] += load_list[layer_idx].sum(dim=0)

    upstream_grad_imp = []
    upstream_grad_load = []
    for layer_idx in range(len(model.blocks)):
        imp = accumulated_importance[layer_idx].detach().requires_grad_(True)
        cv2_imp = model.config.w_importance * (imp.std() / imp.mean()) ** 2
        cv2_imp.backward()
        upstream_grad_imp.append(imp.grad.clone())

        ld = accumulated_load[layer_idx].detach().requires_grad_(True)
        cv2_ld = model.config.w_load * (ld.std() / ld.mean()) ** 2
        cv2_ld.backward()
        upstream_grad_load.append(ld.grad.clone())

    train_loader.current_position = saved_position

    loss_accum = 0.0
    for _ in range(num_accum_steps):
        xb, yb = train_loader.next_batch()
        x, y = xb.to(device), yb.to(device)
        with torch.autocast(device_type=device_type, dtype=autocast_dtype):
            logits, loss, importance_list, load_list = model(x, y)

        loss = loss / num_accum_steps
        loss_accum += loss.detach()

        bal_loss = torch.tensor(0.0, device=device)
        for layer_idx in range(len(model.blocks)):
            imp_micro = importance_list[layer_idx].sum(dim=0)
            load_micro = load_list[layer_idx].sum(dim=0)
            bal_loss = bal_loss + (upstream_grad_imp[layer_idx] * imp_micro).sum()
            bal_loss = bal_loss + (upstream_grad_load[layer_idx] * load_micro).sum()

        total = loss + bal_loss
        total.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss_accum, norm
