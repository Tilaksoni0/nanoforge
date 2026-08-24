"""
Training entry point.

Owns the microstep loop and the GlobalLoadBalancingLoss buffer lifecycle
(reset per optimizer step, update+forward per microstep) -- this is the
one place Algorithm 1 from Qiu et al. 2025 actually gets executed. Model
definition, dispatch strategy, optimizer construction, and LR schedule are
all delegated to their own modules; this file coordinates them.
"""

import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataloader import DataLoaderLite
from src.models.gpt import GPT, GPTConfig
from src.models.moe.losses import GlobalLoadBalancingLoss
from src.models.moe.router import extract_stats
from src.training.lr_schedule import get_lr


def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        dist.init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        master_process = True
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device


def main(
    max_steps: int = 200,
    total_batch_size: int = 32768,
    B: int = 4,
    T: int = 1024,
    dispatch: str = "sort_and_slice",
    alpha_moe: float = 0.01,
):
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process, device = setup_distributed()
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")

    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    assert total_batch_size % (B * T * ddp_world_size) == 0
    num_accum_steps = total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"desired batch size: {total_batch_size}")
        print(f"accumulation steps: {num_accum_steps}")

    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size)

    config = GPTConfig(vocab_size=50304, block_size=T, dispatch=dispatch, alpha_moe=alpha_moe)
    model = GPT(config).to(device)

    if device_type == "cuda":
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model

    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1, learning_rate=6e-4, device_type=device_type
    )

    lbl = GlobalLoadBalancingLoss(
        num_experts=config.n_experts,
        num_layers=config.n_layer,
        alpha=config.alpha_moe,
        device=device,
        dist_group=dist.group.WORLD if ddp else None,
    )

    autocast_dtype = torch.bfloat16 if device_type in ("cuda", "cpu") else torch.float32

    for step in range(max_steps):
        t0 = time.time()
        loss_accum = 0.0
        aux_loss_accum = 0.0

        optimizer.zero_grad()
        lbl.reset()

        for micro_step in range(num_accum_steps):
            xb, yb = train_loader.next_batch()
            x, y = xb.to(device), yb.to(device)

            with torch.autocast(device_type=device_type, dtype=autocast_dtype):
                logits, loss, routing_tuple = model(x, y)

            freq_current, probs_current = extract_stats(routing_tuple, config.n_experts)
            tokens_this_step = x.numel() * config.top_k
            probs_mean_current = probs_current / tokens_this_step

            lbl.update(freq_current)
            aux_loss = lbl(probs_mean_current)

            total_loss = (loss + aux_loss) / num_accum_steps
            loss_accum += loss.detach() / num_accum_steps
            aux_loss_accum += aux_loss.detach() / num_accum_steps

            if ddp:
                model.require_backward_grad_sync = micro_step == num_accum_steps - 1

            total_loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_steps=max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()
        elif device_type == "mps":
            torch.mps.synchronize()

        dt = (time.time() - t0) * 1000
        if master_process:
            loss_val = loss_accum.item() if torch.is_tensor(loss_accum) else loss_accum
            aux_val = aux_loss_accum.item() if torch.is_tensor(aux_loss_accum) else aux_loss_accum
            print(
                f"step {step:3d} | loss {loss_val:.4f} | aux {aux_val:.4f} "
                f"| norm {norm:.4f} | dt {dt:.1f}ms"
            )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
