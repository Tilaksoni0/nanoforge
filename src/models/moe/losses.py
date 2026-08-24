"""
Load-balancing loss: buffer accumulation across microsteps + the loss formula.

This module intentionally does NOT extract freq/probs from raw gate logits
-- that's router.extract_stats's job. This module only accepts already-
extracted (freq, probs) tensors and (a) accumulates freq into a global
buffer across gradient-accumulation microsteps, matching Algorithm 1 of
Qiu et al. 2025 ("Demons in the Detail"), and (b) computes the LBL formula
itself, N_e * sum(F_i * P_i).

GlobalLoadBalancingLoss is the class-based accumulator meant for the actual
training loop -- it owns the buffer's lifecycle (reset each optimizer step,
update each microstep, forward once per microstep using the buffer's
current state). load_balancing_loss below is a stateless reference
function for local/single-step LBL, useful for testing against, for the
legacy Shazeer-style module, or for anywhere a class instance is overkill.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class GlobalLoadBalancingLoss(nn.Module):
    """Accumulates expert-selection frequency across microsteps and
    computes the global-batch LBL (Qiu et al. 2025).

    Usage per optimizer step:
        lbl.reset()
        for microstep in range(num_accum_steps):
            ...forward pass, get freq_current (n_layer, n_experts), probs_current...
            lbl.update(freq_current)                  # syncs + accumulates into buffer
            aux_loss = lbl(probs_current)              # uses buffer's CURRENT state
            (main_loss + aux_loss).backward()
        optimizer.step()

    Note: forward() uses whatever is in freq_buffer at call time, so early
    microsteps see a partial buffer. This matches the paper's Algorithm 1
    exactly, not an idealized fully-global f_i; the buffer only approximates
    true global-batch frequency by the final microstep. Open question on
    early-microstep gradient quality, unaddressed by the paper too.
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        alpha: float,
        device: torch.device,
        dist_group: "dist.ProcessGroup | None" = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.alpha = alpha
        self.dist_group = dist_group
        self.register_buffer(
            "freq_buffer",
            torch.zeros(num_layers, num_experts, device=device, dtype=torch.float32),
            persistent=False,
        )

    @torch.no_grad()
    def update(self, freq_all_layer: torch.Tensor) -> None:
        """Sync this microstep's freq increment across the process group
        (if any) and fold it into the running buffer. Only the fresh
        increment is synced, never the accumulated buffer itself -- syncing
        an already-global buffer again would double-count every prior
        microstep's contribution.
        """
        if freq_all_layer.shape != self.freq_buffer.shape:
            raise ValueError(
                f"Expected frequency shape {tuple(self.freq_buffer.shape)}, "
                f"got {tuple(freq_all_layer.shape)}"
            )
        counts = freq_all_layer.to(device=self.freq_buffer.device, dtype=self.freq_buffer.dtype)
        if self.dist_group is not None and dist.is_initialized():
            counts = counts.clone()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.dist_group)
        self.freq_buffer.add_(counts)

    def forward(self, probability_mean: torch.Tensor) -> torch.Tensor:
        """Compute N_e * sum(F_i * P_i) using the buffer's current state.

        F_i is derived from freq_buffer (detached, no grad ,bincount-
        based counts never carry grad). probability_mean (P_i) must be
        graph-attached and local to the current microstep do not pass
        an accumulated probs buffer here (see the module-level note in
        the unified script this was extracted from: accumulating probs
        across microsteps and backward()-ing after the loop is the bug
        this design deliberately avoids).
        """
        if probability_mean.shape != self.freq_buffer.shape:
            raise ValueError(
                f"Expected probability shape {tuple(self.freq_buffer.shape)}, "
                f"got {tuple(probability_mean.shape)}"
            )
        totals = self.freq_buffer.sum(dim=-1, keepdim=True)
        if torch.any(totals <= 0):
            raise RuntimeError("GlobalLoadBalancingLoss.forward() called before update().")
        frequency = self.freq_buffer / totals
        return self.alpha * self.num_experts * (frequency * probability_mean).sum()

    @torch.no_grad()
    def reset(self) -> None:
        self.freq_buffer.zero_()


def load_balancing_loss(
    probability_mean: torch.Tensor,
    frequency: torch.Tensor,
    alpha: float,
    num_experts: int,
) -> torch.Tensor:
    """Stateless single-step LBL: N_e * sum((freq / freq.sum()) * probs).

    No buffer, no accumulation frequency is normalized in-place from
    whatever is passed in. Use this for local/micro-batch LBL (the
    baseline the global variant is compared against), for the legacy
    Shazeer-style module, or in tests where instantiating the full class
    is unnecessary overhead.
    """
    totals = frequency.sum(dim=-1, keepdim=True)
    if torch.any(totals <= 0):
        raise RuntimeError("frequency must contain at least one selection")
    frequency = frequency / totals
    return alpha * num_experts * (frequency * probability_mean).sum()
