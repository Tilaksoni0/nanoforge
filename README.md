# LLM From Scratch — Learning Repository

This repository is intentionally organized as a learning/research codebase.

## Current progression

Attention Is All You Need
    -> GPT-2
    -> Shazeer-style MoE
    -> custom MoE experiments
    -> Switch Transformer
    -> global load balancing / Qwen-style experiments
    -> future MoE variants

Attention progression

Vanilla attention
    -> FlashAttention 1
    -> FlashAttention 2
    -> FlashAttention 3
    -> FlashAttention 4
    -> Linear Attention
    -> MLA / DeepSeek
    -> future variants

## Two representations

`src/`
    Modular version. Learn repository organization and component boundaries.

`unified/`
    Single-file version. Keep a short, traceable representation of the whole system.

## Design rule

A module should have a meaningful responsibility.

Do not split code into files merely to maximize the number of files.

## Current deliberate simplifications

- Attention is PyTorch scaled_dot_product_attention.
- Norms use PyTorch LayerNorm.
- DDP is kept simple.
- Optimizer creation is separated from training.
- Kernels are only a placeholder.
- Evaluation and post-training are placeholders.
- No attempt is made here to reproduce a production-scale framework.

The repository should grow as the implementation grows.
