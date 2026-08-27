"""
Pure I/O for the tinyshakespeare dataset: download the raw text and cache
it to disk. No tokenizing here that's dataloader.py's job. This file
only ever hands back a path to the raw .txt file, downloading it once if
it isn't already present.
This dataset is very small need no sharding at this stage ~ 33k tokens (tiktoken('gpt2'))
"""
import os
import urllib.request

RAW_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CACHE_DIR = os.path.dirname(__file__)


def get_raw_text_path(cache_dir: str = CACHE_DIR) -> str:
    """Return the path to input.txt, downloading it first if missing.
    Idempotent -- safe to call every run, only hits the network once.
    """
    os.makedirs(cache_dir, exist_ok=True)
    raw_path = os.path.join(cache_dir, "input.txt")
    if not os.path.exists(raw_path):
        urllib.request.urlretrieve(RAW_URL, raw_path)
    return raw_path


if __name__ == "__main__":
    path = get_raw_text_path()
    with open(path, "r") as f:
        text = f.read()
    print(f"input.txt cached at {path} ({len(text):,} chars)")
