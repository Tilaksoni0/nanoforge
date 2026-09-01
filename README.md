# NanoForge

NanoForge is a from-scratch LLM architecture and systems learning repo. Implementations are built directly from papers, checked against a reference implementation, and kept small enough to trace end to end: routing, gradients, dispatch, all of it inspectable.

## What you can do with this repo right now

**1. Run the dispatch benchmark and get real numbers**

```bash
git clone <this repo>
cd NanoForge
pip install -r requirements.txt   # torch, tiktoken

python -m benchmarks.check_dispatch_equivalence   # correctness first
python -m benchmarks.benchmark_dispatch --T 4096 --n_embd 384 --num_experts 8 --top_k 2
```

`check_dispatch_equivalence` confirms every dispatch strategy computes the same math as the naive oracle — run this before trusting any speed number. `benchmark_dispatch` downloads tinyshakespeare automatically on first run and times every registered strategy (`dense_all_experts`, `dense_masked`, `sort_and_slice`, `sort_and_pad`, `sort_pad_bucket`) end to end, forward + backward. Results get appended to `benchmarks/results/history.jsonl`.

No GPU? `bench_eval.ipynb` runs this same benchmark on Colab and already has saved results in it, so you can see real GPU numbers without running anything first.

Full pytest suite too (`pytest tests/`) — dispatch correctness in isolation, plus the real end-to-end `GPT -> Block -> MOE -> router -> dispatch` forward/backward path for every strategy.

**2. Train an actual model — `unified/gpt_moe_unified.py`**

This is a complete, self-contained, runnable training script — GPT + MoE + Shazeer/global LBL + DDP, one file, `python unified/gpt_moe_unified.py`. The modular `src/` version is where active development happens (and is what the benchmarks above test), but it isn't fully wired end-to-end for training yet. If you want to actually train something today, `unified/` is the one that runs.

**3. Use the MoE layer directly**

```python
from src.models.gpt import GPT, GPTConfig

config = GPTConfig(n_embd=384, n_experts=8, top_k=2, dispatch="sort_pad_bucket")
model = GPT(config)
logits, loss, routing = model(idx, targets)
```

Swap `dispatch` to any key in `DISPATCH_REGISTRY` (`src/models/moe/dispatch.py`) — nothing else needs to change. `sort_pad_bucket` is the current default: dropless, bucketed capacity, GPU-shape-stable.

## What we're currently working on

**Faster dispatch — Triton grouped-GEMM.** `sort_and_slice` is the fastest correctness-verified strategy right now, but it still loops over experts in Python. Writing a grouped-GEMM kernel to remove that loop is the active thread (`src/kernels/`).

**Does dispatch choice matter more as expert count grows?**

Comparing `sort_and_slice` against `dense_masked`, with:
- `T` = tokens in the batch, `K` = top_k, `N = T*K` = total (token, expert) assignments
- `C` = hidden dim, `E` = number of experts
- `n_i` = tokens routed to expert `i`, so one expert's forward call costs `O(n_i*C)`

| | dispatch overhead | expert compute | total |
|---|---|---|---|
| `sort_and_slice` | `O(N log N)` (the sort) | `O(N*C)` | `O(N log N + N*C)` |
| `dense_masked` | `O(N*E)` (per-expert scan) | `O(N*C)` | `O(N*E + N*C)` |

Expert compute is identical either way, so the overhead term is the whole difference: `log N` vs `E`. `sort_and_slice` doesn't care how many experts you have; `dense_masked` gets linearly worse as `E` grows.

Rough scaling check, not a measured result yet: GPT-3-small-era training used ~10B tokens at a ~500K batch size. Extrapolating that same ratio to a 20T-token modern run puts `N` around `10^12`. `log2(10^12) ≈ 40`. A Kimi-scale model has ~900 experts. So at that scale the overhead multiplier is `~40` for `sort_and_slice` vs `~900` for `dense_masked` — both scale by the same `10^12`, only the multiplier differs, and it differs by more than 20x. Hypothesis, not measured — next step is actually benchmarking at high expert counts instead of extrapolating.

**Is expert specialization actually about the loss function, or something else?**

Global vs. local load-balancing (see progression below) produce visibly different specialization in early runs. Loss clearly has *some* effect — but is it the main driver, or is something else going on that the loss curve doesn't show? Planned: interpretability probes on trained experts under both LBL formulations, since loss curves alone can't answer this.

## Repository structure

```text
NanoForge/
├── src/
│   ├── models/
│   │   ├── attention/      # causal self-attention (vanilla now, FlashAttention planned)
│   │   ├── moe/            # router, dispatch (5 strategies), experts, losses
│   │   ├── norms/
│   │   └── gpt.py          # wires attention + MoE into a full GPT
│   ├── training/           # train.py, optimizer, lr schedule
│   ├── data/                # tinyshakespeare dataloader
│   ├── config/
│   ├── kernels/             # Triton kernel work (active)
│   ├── evaluation/           # model-quality eval, not yet built (needs a trained checkpoint first)
│   ├── experiments/
│   └── distributed/
├── benchmarks/
│   ├── benchmark_dispatch.py
│   ├── check_dispatch_equivalence.py
│   └── results/              # history.jsonl, latest.json
├── tests/                    # pytest: dispatch equivalence + full GPT forward/backward
├── bench_eval.ipynb          # same benchmark, Colab-runnable, saved GPU results included
└── unified/
    └── gpt_moe_unified.py   # complete runnable training script, single file
```

`src/` is the modular implementation, where active development happens. `unified/` is the one you actually run end to end today.

## Why MoE, coming from a dense Transformer

A dense Transformer couples capacity to compute: every parameter runs on every token, so the only way to make the model know more is to make every forward pass more expensive, for every token, forever. There's no way to add capacity without paying for it on every single input.

MoE breaks that coupling. Instead of one FFN per layer, you have several ("experts"), and a learned router sends each token to only a handful of them (`top_k` out of `n_experts`). Total parameter count scales with the number of experts; compute per token scales with `top_k` only. That's the entire motivation — more capacity, same per-token cost — and it's also the source of MoE's one real new failure mode that a dense model never has to deal with: nothing stops the router from collapsing onto a couple of favorite experts and starving the rest, since routing is learned and unconstrained by default. The whole progression below is the history of that one problem (how do you make the router's discrete top-k choice differentiable, and how do you penalize imbalance) getting solved, simplified, and re-solved at increasing scale.

## Progression

Roughly the order things got built, what each step was chasing, and what came out of it.

**Shazeer MoE first** — needed a working baseline: sparse top-k routing, a gate, dispatch, and an auxiliary load-balancing loss so experts don't collapse onto one or two of them. Everything after this is a variation on this core loop.(see the legacy file in src/model/)
Paper: [arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)

**Then GShard** — read as a step toward how this scales past a single machine, automatic sharding of expert weights across devices. Didn't change the local dispatch math, but shaped how we think about what "dispatch" even means once experts live on different devices.
Paper: [arxiv.org/abs/2006.16668](https://arxiv.org/abs/2006.16668)

**Then Switch Transformer** — simplified top-1 routing instead of top-k, and a simpler load-balancing formulation. Useful comparison point against the original Shazeer version once both were implemented side by side. This is simply the main current load_balancing loss idea we are using. 
Paper: [arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961)

**Then ST-MoE** — mostly about training stability and specific design choices that keep sparse expert models from misbehaving, read as a bridge before going further into modern formulations.
Paper: [arxiv.org/abs/2202.08906](https://arxiv.org/abs/2202.08906)

**Then the global load-balancing loss** —The standard LBL is computed per-microbatch, which is a weaker signal than computing it over the full global batch. This paper walks through implementing the global version properly: accumulating stats across microbatches, keeping only the graph you need, and manually wiring the gradient back through each microstep. This directly shaped the current `losses.py`/`router.py` split in this repo.
Paper: [aclanthology.org/2025.acl-long.249](https://aclanthology.org/2025.acl-long.249/)

**Then Qwen-style / global-LBL direction** — current MoE work follows this global-batch direction, cross-checked against the Qwen3 report's MoE design.
Paper: [arxiv.org/abs/2505.09388](https://arxiv.org/abs/2505.09388)

**(Recent)Deepseekmoe-style segmentation and shared expert pool** - Most recent Architectural change Is addition of segmentation of experts + shared_expert pool which is inspired from Global-Lbl direction which gives more isolated expert utilisation , this paper aims exactly to that, by tweaking the archietecure. Sot it keeps the Total_params same as well as compute size , but fine-graining the experts into smaller ones, which leads to help **Knowledge hybridity** increase in num_experts, gives them more specialised knowledge. It also states one more problem **Knowledge-Redundancy** which means every expert learn the same common knowledge which kindof waste their capacity, to fight exaclty that it introduces **Shared_experts** which excluded from routing and every token get routed to these as a hard constraint. 
Paper: [https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)


Also referenced along the way: **Mixtral** (arxiv.org/abs/2401.04088, the `dense_masked` dispatch strategy mirrors Mixtral's `MixtralExperts`), **Attention Is All You Need** (arxiv.org/abs/1706.03762), **FlashAttention / FlashAttention-2** (arxiv.org/abs/2205.14135, arxiv.org/abs/2307.08691), **GPT-2** (cdn.openai.com/better-language-models/language-models.pdf), **GPT-3** (arxiv.org/abs/2005.14165), **Adam** (arxiv.org/abs/1412.6980), **AdamW** (arxiv.org/abs/1711.05101).
