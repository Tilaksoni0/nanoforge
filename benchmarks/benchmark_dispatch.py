"""
Timing benchmark for the three dispatch strategies.

Run check_dispatch_equivalence.py first, these numbers are meaningless
if the strategies don't compute the same thing. Measures wall-clock
forward+backward time under matched (T, n_embd, num_experts, top_k)
settings only, not memory. CPU timings will not predict GPU behavior:
sort_and_slice's advantage comes from avoiding per-expert boolean-mask
gather, which matters far more on GPU where gather/scatter patterns
interact with memory coalescing. CPU numbers here are a smoke test,
not the real verdict, run on CUDA for numbers that actually inform the
"which dispatch is fastest" decision.
"""

import time

import torch

from src.models.moe.dispatch import dense_all_experts, dense_masked, sort_and_slice
from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import GatingNetwork, route


def build_matched_experts(n_embd, hidden, num_experts, device):
    ml = ModuleListExperts(n_embd, hidden, n_embd, num_experts).to(device)
    st = StackedExperts(n_embd, hidden, n_embd, num_experts, swiglu=False).to(device)
    with torch.no_grad():
        for e in range(num_experts):
            expert = ml[e]
            st.gate_up_proj[e] = expert.f_u.weight.detach().clone()
            st.down_proj[e] = expert.f_d.weight.detach().clone()
    return ml, st


def time_dispatch(fn, x, gate, experts, top_k, iters, device):
    """Each iteration re-runs routing fresh, since dispatch consumes (and
    frees) the routing tensors' graph on backward, reusing one RoutingResult
    object across iterations double-backwards through an already-freed graph.
    This also more realistically mirrors real training, where routing is
    recomputed every forward pass anyway, not cached across steps.
    """
    from src.models.moe.router import route

    def run_once():
        routing = route(x, gate, top_k)
        out = fn(x, routing, experts, top_k)
        out.sum().backward()
        x.grad = None
        for p in experts.parameters():
            p.grad = None
        for p in gate.parameters():
            p.grad = None

    for _ in range(5):
        run_once()

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1000  # ms/iter


def main(
    T: int = 4096,
    n_embd: int = 384,
    hidden: int = 4 * 384,
    num_experts: int = 8,
    top_k: int = 2,
    iters: int = 20,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    x = torch.randn(T, n_embd, device=device, requires_grad=True)
    gate = GatingNetwork(n_embd, num_experts).to(device)

    ml, st = build_matched_experts(n_embd, hidden, num_experts, device)

    results = {}
    results["dense_all_experts"] = time_dispatch(dense_all_experts, x, gate, ml, top_k, iters, device)
    results["sort_and_slice"] = time_dispatch(sort_and_slice, x, gate, ml, top_k, iters, device)
    results["dense_masked"] = time_dispatch(dense_masked, x, gate, st, top_k, iters, device)

    print(f"device={device}, T={T}, n_embd={n_embd}, num_experts={num_experts}, top_k={top_k}, iters={iters}\n")
    for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:20s} {ms:8.3f} ms/iter")


if __name__ == "__main__":
    main()