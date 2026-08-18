# NanoForge

NanoForge is a from-scratch LLM architecture and systems learning repository.
The goal is to understand the mechanisms in modern language models by implementing them directly from papers, checking them against reference implementations, and keeping the code small enough to trace.

## What has been implemented

### Transformer / GPT-style base

The current codebase contains a GPT-style decoder-only Transformer built around the original Transformer ideas and the GPT-2 lineage.

Current implementation includes:
- token and positional embeddings
- causal self-attention
- multi-head attention
- Transformer blocks
- language-model head with tied embeddings
- autoregressive cross-entropy training
- gradient accumulation
- AdamW optimizer
- learning-rate scheduling
- basic mixed-precision support
- basic DDP support

### Mixture of Experts progression

The main part of the repository so far has been the progression through sparse MoE routing and load balancing.

#### 1. Shazeer-style MoE

Implemented from:

**Shazeer et al. (2017) — Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**

The implementation explores:
- sparse top-k routing
- gating network
- expert selection
- expert dispatch
- weighted expert outputs
- expert-selection statistics
- importance/load statistics
- auxiliary load-balancing loss
- gradient flow through the router

Paper:
https://arxiv.org/abs/1701.06538

#### 2. GShard

GShard was studied as an intermediate step in understanding how sparse MoE architectures move toward large-scale distributed training and automatic sharding.

**Lepikhin et al. (2020/2021) — GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding**

Paper:
https://arxiv.org/abs/2006.16668

#### 3. Switch Transformer

The implementation was then modified toward the Switch Transformer formulation.

**Fedus, Zoph & Shazeer (2021; JMLR 2022) — Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**

The repository includes experiments with:
- top-1 routing
- Switch-style load balancing
- comparison against the earlier Shazeer-style formulation
- removing the importance component and retaining the load-based formulation

Paper:
https://arxiv.org/abs/2101.03961

#### 4. ST-MoE

ST-MoE was studied as an intermediate step in the Switch/MoE progression, particularly around stability and design choices for sparse expert models.

**Zoph et al. (2022) — ST-MoE: Designing Stable and Transferable Sparse Expert Models**

Paper:
https://arxiv.org/abs/2202.08906

#### 5. Global Load-Balancing Loss

The next major implementation step was moving from a micro-batch view of the load-balancing loss toward a global-batch formulation.

This work is based on:

**Qiu et al. (2025) — Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models**

The implementation specifically deals with:
- global expert-selection statistics
- microbatch accumulation
- gradient accumulation
- distributed synchronization of routing statistics
- computing the load-balancing signal from global statistics
- keeping the computation graph alive only where required
- manually connecting the global auxiliary-loss gradient back into each microstep through the chain rule
- avoiding retention of all microstep computation graphs

Paper:
https://aclanthology.org/2025.acl-long.249/

#### 6. Qwen-style MoE

The current MoE work follows the Qwen-style/global-load-balancing direction discussed in the above work and Qwen model reports.

Qwen3 includes both dense and Mixture-of-Experts language models.

**Yang et al. (2025) — Qwen3 Technical Report**

Paper:
https://arxiv.org/abs/2505.09388

## Attention implementation

The current attention implementation is a causal multi-head self-attention module using PyTorch's `scaled_dot_product_attention`.

The architectural lineage being studied here starts with:

**Vaswani et al. (2017) — Attention Is All You Need**

Paper:
https://arxiv.org/abs/1706.03762

The repository also uses the FlashAttention work as part of the current attention study:

**Dao et al. (2022) — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**

Paper:
https://arxiv.org/abs/2205.14135

**Dao (2023) — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**

Paper:
https://arxiv.org/abs/2307.08691

The official FlashAttention implementation is:
https://github.com/Dao-AILab/flash-attention

## GPT papers

The decoder-only language-model progression is grounded in the GPT papers:

**Radford et al. (2019) — Language Models are Unsupervised Multitask Learners (GPT-2)**

Paper:
https://cdn.openai.com/better-language-models/language-models.pdf

Reference implementation:
https://github.com/openai/gpt-2

**Brown et al. (2020) — Language Models are Few-Shot Learners (GPT-3)**

Paper:
https://arxiv.org/abs/2005.14165

## Additional MoE paper studied in the progression

**Jiang et al. (2024) — Mixtral of Experts**

Paper:
https://arxiv.org/abs/2401.04088

Mixtral is relevant to the progression from sparse top-k MoE toward modern large language model architectures.

## Optimization

The current training code uses AdamW.

The foundational optimizer paper is:

**Kingma & Ba (2014) — Adam: A Method for Stochastic Optimization**

Paper:
https://arxiv.org/abs/1412.6980

The AdamW formulation is based on:

**Loshchilov & Hutter (2019) — Decoupled Weight Decay Regularization**

Paper:
https://arxiv.org/abs/1711.05101

## Repository structure

```text
NanoForge/
├── src/
│   ├── config/
│   ├── data/
│   ├── distributed/
│   ├── evaluation/
│   ├── experiments/
│   ├── kernels/
│   ├── models/
│   │   ├── attention/
│   │   ├── moe/
│   │   └── norms/
│   └── training/
│
├── unified/
│   └── gpt_moe_unified.py
│
└── README.md
```

`src/` contains the modular implementation.

`unified/` contains a compact single-file representation of the system so the complete computation can still be followed without navigating through the entire module tree.

The repository is intentionally modular because the same Transformer can eventually be assembled from different attention, normalization, MoE, and optimization implementations.

## Implementation philosophy

The implementations are written to understand the mechanisms rather than to reproduce an industrial training framework.

The workflow for an architectural idea is:

```text
paper
  ↓
understand the mathematics
  ↓
inspect/reference an implementation when useful
  ↓
implement the mechanism from scratch
  ↓
test and inspect its behavior
  ↓
integrate it into the training system
```

The repository keeps the implementation small enough that routing, gradients, distributed statistics, and model computation can be inspected directly.

## Current status

NanoForge is an active learning/research codebase.

The current implementations are still being separated into clean, independently testable modules, and the training infrastructure is being developed alongside the architectural implementations.

Not every module is currently a complete standalone training program. The repository is being made progressively more reproducible as the implementations are separated and integrated.
