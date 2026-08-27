"""
Timing benchmark for every dispatch strategy in DISPATCH_REGISTRY.
every dispatch technique run through this bench telling which one is the fastest. 
Note:
1. Run the equivalence check first before this , cuz without it these numbers means nothing 
2. Sort_and_pad and Sort_pad_bucket, here may show very differnt numer even though they feel to be similar and one might expect 
sort_pad_bucket to be faster , but remeber coparing them on this bench when the capcity is differnt is not fair at the sake of time only 
"""
import argparse
import datetime
import json
import os
import time

import torch

from src.models.moe.moe import MOEConfig
from src.models.moe.dispatch import DISPATCH_REGISTRY
from src.models.moe.experts import ModuleListExperts, StackedExperts
from src.models.moe.router import GatingNetwork, route

STACKED_STRATEGIES = {"dense_masked", "sort_and_pad", "sort_pad_bucket"}
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def build_matched_experts(n_embd, hidden, num_experts, device):
    ml = ModuleListExperts(n_embd, hidden, n_embd, num_experts, swiglu=False).to(device)
    st = StackedExperts(n_embd, hidden, n_embd, num_experts, swiglu=False).to(device)
    with torch.no_grad():
        for e in range(num_experts):
            expert = ml[e]
            st.gate_up_proj[e] = expert.f_u.weight.detach().clone()
            st.down_proj[e] = expert.f_d.weight.detach().clone()
    return ml, st


def load_real_batch(B, T, n_embd, device):
    """gets the batch from the data_loader and has its own embdding to provide the final embedinng tokens.
    """
    from src.data.token_dataloader import get_batch, vocab_size

    x_ids, _ = get_batch("train", B, T, device)
    embed = torch.nn.Embedding(vocab_size(), n_embd).to(device)
    x = embed(x_ids).reshape(B * T, n_embd)
    x = x.detach().requires_grad_(True)
    return x


def time_dispatch(fn, x, gate, experts, top_k, iters, device, extra_kwargs):
    """Each iteration re-runs routing fresh, since dispatch consumes (and
    frees) the routing tensors' graph on backward, reusing one
    RoutingResult object across iterations double-backwards through an
    already-freed graph. This also more realistically mirrors real
    training, where routing is recomputed every forward pass anyway, not
    cached across steps.
    """
    def run_once():
        routing = route(x, gate, top_k)
        out = fn(x, routing, experts, top_k, **extra_kwargs)
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
    synthetic: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    if synthetic:
        x = torch.randn(T, n_embd, device=device, requires_grad=True)
    else:
        B = 8
        T_seq = max(T // B, 1)
        x = load_real_batch(B, T_seq, n_embd, device)
        T = x.shape[0]  # B * T_seq, may differ slightly from requested T due to integer division

    gate = GatingNetwork(n_embd, num_experts).to(device)
    ml, st = build_matched_experts(n_embd, hidden, num_experts, device)

    results = {}
    for name, fn in DISPATCH_REGISTRY.items():
        experts = st if name in STACKED_STRATEGIES else ml
        extra_kwargs = {}
        if name == "sort_pad_bucket":
            extra_kwargs = {"BUCKET": MOEConfig(n_embd=n_embd, n_experts=num_experts, top_k=top_k).bucket}
        ms = time_dispatch(fn, x, gate, experts, top_k, iters, device, extra_kwargs)
        results[name] = ms

    print(
        f"device={device}, T={T}, n_embd={n_embd}, num_experts={num_experts}, top_k={top_k}, "
        f"iters={iters}, data={'synthetic' if synthetic else 'shakespeare'}\n"
    )
    for name, ms in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:20s} {ms:8.3f} ms/iter")

    save_results(results, device, T, n_embd, num_experts, top_k, iters, synthetic)
    return results


def save_results(results, device, T, n_embd, num_experts, top_k, iters, synthetic):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "device": str(device),
        "T": T,
        "n_embd": n_embd,
        "num_experts": num_experts,
        "top_k": top_k,
        "iters": iters,
        "data": "synthetic" if synthetic else "shakespeare",
        "results_ms": results,
    }

    history_path = os.path.join(RESULTS_DIR, "history.jsonl")
    with open(history_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    latest_path = os.path.join(RESULTS_DIR, "latest.json")
    with open(latest_path, "w") as f:
        json.dump(record, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    main(
        T=args.T,
        n_embd=args.n_embd,
        hidden=4 * args.n_embd,
        num_experts=args.num_experts,
        top_k=args.top_k,
        iters=args.iters,
        synthetic=args.synthetic,
    )
