import os
import torch
import torch.distributed as dist
from torch.distributed import init_process_group


def setup_distributed():
    ddp = int(os.environ.get("RANK", -1)) != -1

    if not ddp:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        return {
            "ddp": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "master_process": True,
            "device": device,
        }

    assert torch.cuda.is_available()

    init_process_group(backend="nccl")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    return {
        "ddp": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "master_process": rank == 0,
        "device": device,
    }


def cleanup_distributed(ddp):
    if ddp:
        dist.destroy_process_group()
