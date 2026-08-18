from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    alpha_moe: float = 0
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 384
    n_experts: int = 8
    top_k: int = 2
