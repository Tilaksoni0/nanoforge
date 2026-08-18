from dataclasses import dataclass
import inspect
import math
import os
import time

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================
# CONFIG
# ============================================================

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


# ============================================================
# ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)


# ============================================================
# MOE COMPONENTS
# ============================================================

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return self.gate(x)


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gelu = nn.GELU(approximate="tanh")
        self.f_u = nn.Linear(input_dim, hidden_dim)
        self.f_d = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.f_d(self.gelu(self.f_u(x)))


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.experts = nn.ModuleList([
            Expert(
                config.n_embd,
                4 * config.n_embd,
                config.n_embd,
            )
            for _ in range(config.n_experts)
        ])

        self.gate = GatingNetwork(
            config.n_embd,
            config.n_experts,
        )

        self.track_experts = []

    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        T_total = B * T

        gate_logits = self.gate(x)
        probs = torch.softmax(gate_logits, dim=-1)

        values, indices = torch.topk(
            probs,
            k=self.config.top_k,
            dim=-1,
        )

        self.track_experts.append(indices.detach())

        flattened_indices = indices.flatten()
        flattened_values = values.flatten()

        token_ids = (
            torch.arange(T_total, device=x.device)
            .unsqueeze(1)
            .expand(-1, self.config.top_k)
            .reshape(-1)
        )

        exp_order = torch.argsort(flattened_indices)

        sorted_experts = flattened_indices[exp_order]
        sorted_token_ids = token_ids[exp_order]
        sorted_values = flattened_values[exp_order]

        counts = torch.bincount(
            sorted_experts,
            minlength=self.config.n_experts,
        )

        final_output = torch.zeros(
            T_total,
            C,
            device=x.device,
            dtype=x.dtype,
        )

        start = 0

        for exp_id, expert in enumerate(self.experts):
            count = counts[exp_id].item()

            if count == 0:
                continue

            end = start + count
            tok_ids = sorted_token_ids[start:end]
            weights = sorted_values[start:end]

            expert_out = expert(x[tok_ids])
            final_output[tok_ids] += (
                expert_out * weights.unsqueeze(-1)
            )

            start = end

        return final_output.reshape(B, T, C), gate_logits


# ============================================================
# AUXILIARY LOAD BALANCING
# ============================================================

def return_stats(gate_logits_tuple, num_experts, top_k=2):
    if gate_logits_tuple is None or len(gate_logits_tuple) == 0:
        return torch.empty(0), torch.empty(0)

    n_layer = len(gate_logits_tuple)
    device = gate_logits_tuple[0].device

    freq_all_layer = torch.zeros(
        n_layer,
        num_experts,
        device=device,
    )

    probs_all_layer = torch.zeros(
        n_layer,
        num_experts,
        device=device,
        dtype=gate_logits_tuple[0].dtype,
    )

    for layer_id, layer_logits in enumerate(gate_logits_tuple):
        layer_probs_full = torch.softmax(layer_logits, dim=-1)
        values, indices = torch.topk(
            layer_probs_full,
            top_k,
            dim=-1,
        )

        flattened_indices = indices.flatten()
        flattened_values = values.flatten()

        counts = torch.bincount(
            flattened_indices,
            minlength=num_experts,
        )

        freq_all_layer[layer_id, :] = counts

        layer_probs = torch.zeros(
            num_experts,
            device=device,
            dtype=values.dtype,
        )

        layer_probs.scatter_add_(
            0,
            flattened_indices,
            flattened_values,
        )

        probs_all_layer[layer_id, :] = layer_probs

    return freq_all_layer, probs_all_layer


def global_lbl(F_i, P_i, N_e, alpha):
    assert F_i.shape == P_i.shape
    return alpha * N_e * (F_i * P_i).sum()


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MoE(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        moe_out, gate_logits = self.mlp(self.ln_2(x))
        x = x + moe_out

        return x, gate_logits


# ============================================================
# GPT
# ============================================================

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([
                Block(config)
                for _ in range(config.n_layer)
            ]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )

        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(
                module.weight,
                mean=0,
                std=std,
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0,
                std=0.02,
            )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        assert T <= self.config.block_size

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device,
        )

        x = (
            self.transformer.wte(idx)
            + self.transformer.wpe(pos)
        )

        all_gate_logits = []

        for block in self.transformer.h:
            x, gate_logits = block(x)
            all_gate_logits.append(gate_logits)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss, tuple(all_gate_logits)


# ============================================================
# DATA
# ============================================================

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = enc.encode(text)

        self.current_position = B * T * process_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = torch.tensor(
            self.tokens[
                self.current_position:
                self.current_position + B * T + 1
            ]
        )

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if (
            self.current_position
            + B * T * self.num_processes
            + 1
            > len(self.tokens)
        ):
            self.current_position = B * T * self.process_rank

        return x, y


# ============================================================
# OPTIMIZER / LR
# ============================================================

MAX_LR = 6e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 10
MAX_STEPS = 200


def get_lr(it):
    if it < WARMUP_STEPS:
        return MAX_LR * (it + 1) / WARMUP_STEPS

    if it > MAX_STEPS:
        return MIN_LR

    decay_ratio = (
        it - WARMUP_STEPS
    ) / (
        MAX_STEPS - WARMUP_STEPS
    )

    coeff = 0.5 * (
        1 + math.cos(math.pi * decay_ratio)
    )

    return MIN_LR + coeff * (MAX_LR - MIN_LR)


def configure_optimizers(
    model,
    weight_decay,
    learning_rate,
    device_type,
):
    param_dict = {
        n: p
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    decay_param = [
        p for p in param_dict.values()
        if p.dim() >= 2
    ]

    nodecay_param = [
        p for p in param_dict.values()
        if p.dim() < 2
    ]

    optim_groups = [
        {
            "params": decay_param,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_param,
            "weight_decay": 0.0,
        },
    ]

    fused_available = (
        "fused"
        in inspect.signature(torch.optim.AdamW).parameters
    )

    use_fused = (
        fused_available
        and device_type == "cuda"
    )

    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )


# ============================================================
# DDP SETUP
# ============================================================

def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1

    if ddp:
        assert torch.cuda.is_available()

        init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)

        return (
            True,
            rank,
            local_rank,
            world_size,
            rank == 0,
            device,
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return (
        False,
        0,
        0,
        1,
        True,
        device,
    )


def cleanup_distributed(ddp):
    if ddp:
        dist.destroy_process_group()


# ============================================================
# TRAINING
# ============================================================

def main():
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_distributed()

    device_type = (
        "cuda"
        if device.startswith("cuda")
        else "mps"
        if device == "mps"
        else "cpu"
    )

    torch.manual_seed(1337)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    if device_type == "cuda":
        B, T = 4, 1024
        total_batch_size = 32768
    else:
        B, T = 4, 256
        total_batch_size = 4096

    assert total_batch_size % (
        B * T * ddp_world_size
    ) == 0

    num_accum_steps = (
        total_batch_size
        // (B * T * ddp_world_size)
    )

    train_loader = DataLoaderLite(
        B,
        T,
        ddp_rank,
        ddp_world_size,
    )

    model = GPT(
        GPTConfig(
            vocab_size=50304,
            block_size=T,
        )
    ).to(device)

    if device_type == "cuda":
        model = torch.compile(model)

    if ddp:
        model = DDP(
            model,
            device_ids=[ddp_local_rank],
        )

    raw_model = model.module if ddp else model

    optimizer = configure_optimizers(
        raw_model,
        weight_decay=0.1,
        learning_rate=6e-4,
        device_type=device_type,
    )

    autocast_dtype = (
        torch.bfloat16
        if device_type in ("cuda", "cpu")
        else torch.float32
    )

    for step in range(MAX_STEPS):
        t0 = time.time()

        loss_accum = 0
        aux_loss_accum = 0

        optimizer.zero_grad()

        freq_buffer = torch.zeros(
            raw_model.config.n_layer,
            raw_model.config.n_experts,
            device=device,
        )

        tokens_seen = 0

        for micro_step in range(num_accum_steps):
            xb, yb = train_loader.next_batch()

            x = xb.to(device)
            y = yb.to(device)

            with torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
            ):
                logits, loss, gate_logits_tuple = model(x, y)

            freq_current, probs_current = return_stats(
                gate_logits_tuple,
                raw_model.config.n_experts,
                raw_model.config.top_k,
            )

            freq_increment = freq_current.detach()

            tokens_this_step = torch.tensor(
                x.numel()
                * raw_model.config.top_k,
                device=device,
                dtype=torch.float32,
            )

            if ddp:
                dist.all_reduce(
                    freq_increment,
                    op=dist.ReduceOp.SUM,
                )

                dist.all_reduce(
                    tokens_this_step,
                    op=dist.ReduceOp.SUM,
                )

            freq_buffer += freq_increment
            tokens_seen += tokens_this_step.item()

            F_i = (
                freq_buffer / tokens_seen
            ).detach()

            P_i_current = (
                probs_current
                / (x.numel() * raw_model.config.top_k)
            )

            aux_loss = global_lbl(
                F_i,
                P_i_current,
                raw_model.config.n_experts,
                raw_model.config.alpha_moe,
            )

            total_loss = (
                loss + aux_loss
            ) / num_accum_steps

            loss_accum += (
                loss.detach()
                / num_accum_steps
            )

            aux_loss_accum += (
                aux_loss.detach()
                / num_accum_steps
            )

            if ddp:
                model.require_backward_grad_sync = (
                    micro_step
                    == num_accum_steps - 1
                )

            total_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )

        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

        if master_process:
            print(
                f"step {step:3d} | "
                f"loss {loss_accum.item():.4f} | "
                f"aux {aux_loss_accum.item():.4f} | "
                f"norm {norm:.4f}"
            )

    cleanup_distributed(ddp)


if __name__ == "__main__":
    main()
