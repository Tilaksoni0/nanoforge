"""
Correctness check for dispatch strategies.

All three dispatch functions implement the same math, a weighted sum
over each token's top_k selected experts, so given the same weights and
the same routing decision, their outputs must match to floating point
tolerance. Builds one shared set of expert weights, copies them across
a ModuleListExperts and a StackedExperts container, runs the same input
through all three dispatch functions, and asserts the outputs agree.

Run this after touching dispatch.py, before trusting benchmark numbers,
a "fast" dispatch that disagrees with the reference is a bug, not a win.
"""

import torch

from src.models.moe.dispatch import dense_all_experts, dense_masked, sort_and_slice
from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import GatingNetwork, route


def build_matched_experts(n_embd: int, hidden: int, num_experts: int, device: torch.device):
    """Build a ModuleListExperts and a StackedExperts(swiglu=False) that
    compute the identical function: GELU(up(x)) @ down, no bias anywhere.
    swiglu=False is required for a fair comparison, see StackedExperts'
    docstring. Weights are copied directly so this is a same-math,
    same-weights comparison, isolating dispatch (token routing/grouping)
    as the only variable between dense_all_experts/sort_and_slice (which
    use ModuleListExperts) and dense_masked (which uses StackedExperts).
    """
    ml = ModuleListExperts(n_embd, hidden, n_embd, num_experts).to(device)
    st = StackedExperts(n_embd, hidden, n_embd, num_experts, swiglu=False).to(device)

    with torch.no_grad():
        for e in range(num_experts):
            expert = ml[e]
            st.gate_up_proj[e] = expert.f_u.weight.detach().clone()
            st.down_proj[e] = expert.f_d.weight.detach().clone()

    return ml, st


@torch.no_grad()
def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    T, n_embd, hidden, num_experts, top_k = 37, 16, 32, 8, 2

    x = torch.randn(T, n_embd, device=device)
    gate = GatingNetwork(n_embd, num_experts).to(device)
    routing = route(x, gate, top_k)

    ml, st = build_matched_experts(n_embd, hidden, num_experts, device)

    out_dense = dense_all_experts(x, routing, ml, top_k)
    out_sort = sort_and_slice(x, routing, ml, top_k)
    out_masked = dense_masked(x, routing, st, top_k)

    diff_sort = (out_dense - out_sort).abs().max().item()
    diff_masked = (out_dense - out_masked).abs().max().item()

    print(f"max |dense_all_experts - sort_and_slice| = {diff_sort:.2e}")
    print(f"max |dense_all_experts - dense_masked|    = {diff_masked:.2e}")

    tol = 1e-4
    assert diff_sort < tol, "sort_and_slice disagrees with reference dispatch"
    assert diff_masked < tol, "dense_masked disagrees with reference dispatch"
    print("OK: all three dispatch strategies agree.")


if __name__ == "__main__":
    main()