"""
GPT: decoder-only transformer, MoE feed-forward sublayer.

This file only wires components together (embeddings, attention blocks,
MOE, LM head) -- it does not implement attention or MoE internals itself.
See src/models/attention/ and src/models/moe/ for those.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import CausalSelfAttention
from src.models.moe import MOE, MOEConfig, RoutingResult


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 384
    n_experts: int = 8
    top_k: int = 2
    dispatch: str = "sort_and_slice"
    alpha_moe: float = 0.01


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MOE(
            MOEConfig(
                n_embd=config.n_embd,
                n_experts=config.n_experts,
                top_k=config.top_k,
                dispatch=config.dispatch,
            )
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, RoutingResult]:
        x = x + self.attn(self.ln_1(x))
        moe_out, routing = self.mlp(self.ln_2(x))
        x = x + moe_out
        return x, routing


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[RoutingResult, ...]]:
        B, T = idx.shape
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        all_routing: list[RoutingResult] = []
        for block in self.transformer.h:
            x, routing = block(x)
            all_routing.append(routing)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, tuple(all_routing)

    def configure_optimizers(self, weight_decay: float, learning_rate: float, device_type: str):
        import inspect

        param_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay_param = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_param = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_param, "weight_decay": weight_decay},
            {"params": nodecay_param, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
