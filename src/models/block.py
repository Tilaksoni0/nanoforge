import torch.nn as nn

from .attention import CausalSelfAttention
from .moe import MoE


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Current model uses LayerNorm.
        # Later this can become RMSNorm/custom norm.
        self.ln_1 = nn.LayerNorm(config.n_embd)

        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)

        self.mlp = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))

        moe_out, gate_logits = self.mlp(self.ln_2(x))
        x = x + moe_out

        return x, gate_logits
