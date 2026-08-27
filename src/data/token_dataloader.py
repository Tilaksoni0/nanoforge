"""
Tokenizing data loader for tinyshakespeare, GPT-2 BPE via tiktoken.

this file provides a functional get_batch()/vocab_size() API used by the benchmark scripts.
Tokenization happens once, the resulting token ids are cached to
train.bin/val.bin on disk (90/10 split), and get_batch() just slices out
of the cached tensor. Re-running training/benchmarks never re-encodes.

Depends on shakespeare_dataset.py only for the raw text path this file
owns everything downstream of that (tokenizing, caching, batching).
"""
import os

import numpy as np
import tiktoken
import torch

from src.data.shakespeare_dataset import get_raw_text_path

CACHE_DIR = os.path.dirname(__file__)


def _prepare(cache_dir: str = CACHE_DIR) -> None:
    """Idempotent: tokenize + cache train.bin/val.bin if not already done."""
    train_path = os.path.join(cache_dir, "train.bin")
    val_path = os.path.join(cache_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    raw_path = get_raw_text_path(cache_dir)
    with open(raw_path, "r") as f:
        text = f.read()

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_ids = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)


def get_batch(split: str, B: int, T: int, device: torch.device):
    """split: 'train' or 'val'. Returns (x, y), each (B, T) LongTensor of
    GPT-2 token ids; y is x shifted one position (next-token targets).
    Matches the (x, y) convention of DataLoaderLite.
    """
    _prepare()
    path = os.path.join(CACHE_DIR, f"{split}.bin")
    data = np.memmap(path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def vocab_size() -> int:
    """GPT-2 BPE vocab size, fixed by the tiktoken encoding used above."""
    return tiktoken.get_encoding("gpt2").n_vocab


if __name__ == "__main__":
    _prepare()
    device = torch.device("cpu")
    x, y = get_batch("train", B=4, T=32, device=device)
    print(f"vocab_size={vocab_size()}, x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}")
