"""
Correctness check for dispatch strategies.

All three dispatch functions implement the same math, a weighted sum
over each token's top_k selected experts, so given the same weights and
the same routing decision, their outputs must match to floating point
tolerance. Builds one shared set of expert weights, copies them across
a ModuleListExperts and a StackedExperts container, runs the same input
through all three dispatch functions, and asserts the outputs agree.
"""

import torch

from src.models.moe.dispatch import DISPATCH_REGISTRY, dense_all_experts
from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import GatingNetwork, route

# Strategies that use StackedExperts instead of ModuleListExperts.
STACKED_STRATEGIES = {"dense_masked", "sort_and_pad", "sort_pad_bucket"}


# Extra kwargs each strategy needs beyond (x, routing, experts, top_k).
# T is the test batch's token count - capacity_factor/BUCKET are set here
# to guarantee zero drops for T, so the comparison is apples to apples.
def _extra_kwargs(name: str, T: int) -> dict:
    if name == "sort_and_pad":
        return {"capacity_factor": 999.0}  # deliberately oversized to guarantee no drops but can grow too large so see accrdingly 
    if name == "sort_pad_bucket":
        return {"BUCKET": [T]}  # single bucket exactly at T-  covers the true worst case, dropless
    return {}


def build_matched_experts(n_embd: int, hidden: int, num_experts: int, device: torch.device):
    """Build a ModuleListExperts and a StackedExperts(swiglu=False) that
    compute the identical function: GELU(up(x)) @ down, no bias anywhere.
    swiglu=False is required for a fair comparison, see StackedExperts'
    docstring. Weights are copied directly so this is a same-math,
    same-weights comparison, isolating dispatch (token routing/grouping)
    as the only variable between strategies.
    """
    ml = ModuleListExperts(n_embd, hidden, n_embd, num_experts, swiglu=False).to(device)
    st = StackedExperts(n_embd, hidden, n_embd, num_experts, swiglu=False).to(device)
    with torch.no_grad():
        for e in range(num_experts):
            expert = ml[e]
            st.gate_up_proj[e] = expert.f_u.weight.detach().clone()
            st.down_proj[e] = expert.f_d.weight.detach().clone()
    return ml, st


@torch.no_grad()
def check_all(T=37, n_embd=16, hidden=32, num_experts=8, top_k=2, tol=1e-4, verbose=True):
    """Run every DISPATCH_REGISTRY strategy against the dense_all_experts
    oracle on one shared (x, routing) pair, return {name: max_abs_diff}.
    Raises AssertionError on the first strategy that disagrees past tol.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")

    x = torch.randn(T, n_embd, device=device)
    gate = GatingNetwork(n_embd, num_experts).to(device)
    routing = route(x, gate, top_k)

    ml, st = build_matched_experts(n_embd, hidden, num_experts, device)

    out_oracle = dense_all_experts(x, routing, ml, top_k)

    diffs = {}
    for name, fn in DISPATCH_REGISTRY.items():
        if name == "dense_all_experts":
            continue
        experts = st if name in STACKED_STRATEGIES else ml
        kwargs = _extra_kwargs(name, T)
        out = fn(x, routing, experts, top_k, **kwargs)
        diff = (out_oracle - out).abs().max().item()
        diffs[name] = diff
        if verbose:
            print(f"  max |dense_all_experts - {name:16s}| = {diff:.2e}")
        assert diff < tol, f"{name} disagrees with reference dispatch (diff={diff:.2e}, tol={tol:.2e})"

    if verbose:
        print(f"OK: all {len(diffs)} dispatch strategies agree with the oracle.")
    return diffs


def main():
    check_all()


if __name__ == "__main__":
    main()
