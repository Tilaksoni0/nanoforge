import os
import sys
import json
import time
import shutil
import argparse

import requests
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    EntryNotFoundError,
    GatedRepoError,
)

from src.shared import get_base_dir


""" Downloads the dataset from hugginng face, domain wise files are seperated for later mixabillity 
    CLI helps in specifying the domain and the number of shards to download(for respective domains)
"""

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

BASE_DIR = get_base_dir()
DATA_DIR = os.path.join(BASE_DIR, "data")
api = HfApi()

DATASETS_CHOICES = {
    "fineweb-edu/sample-10BT": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "path_in_repo": "sample/10BT",
        "repo_type": "dataset",
    }
}


def show_available_choices():
    """Shows available domains. Fixed set for now, will be made editable."""
    return list(DATASETS_CHOICES.keys())


def idx_to_path(idx):
    return f"shard_{idx:05d}.parquet"


def domain_to_dirname(domain):
    """'fineweb-edu/sample-10BT' -> 'fineweb-edu__sample-10BT' (one flat folder)."""
    return domain.replace("/", "__")


def list_remote_shards(domain):
    info = DATASETS_CHOICES[domain]
    paths = [
        p.path
        for p in api.list_repo_tree(
            repo_id=info["repo_id"],
            path_in_repo=info["path_in_repo"],
            repo_type=info["repo_type"],
            recursive=True,
        )
        if p.path.endswith(".parquet")
    ]
    return sorted(paths)


def is_valid_parquet(path):
    """True if file exists, opens as parquet, and has a 'text' column."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    try:
        return "text" in pq.ParquetFile(path).schema_arrow.names
    except Exception:
        return False


def build_domain_file_index(data_dir=None):
    """Map each downloaded domain folder to its sorted parquet shard paths."""
    data_dir = data_dir or DATA_DIR
    parquet_dir = os.path.join(data_dir, "parquet")
    assert os.path.isdir(parquet_dir), f"{parquet_dir} does not exist, download something first"

    path_dict = {}
    for name in sorted(os.listdir(parquet_dir)):
        folder = os.path.join(parquet_dir, name)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
        path_dict[name] = {"parquet_paths": [os.path.join(folder, f) for f in files]}
    return path_dict


def download_shard_with_retry(repo_id, filename, save_path, staging_dir, max_attempts=5):
    """Download one shard to staging, then atomically move it to save_path.
    Returns save_path on success, None after exhausting retries."""
    for attempt in range(1, max_attempts + 1):
        try:
            tmp = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=staging_dir,
                token=HF_TOKEN,
            )
            if not is_valid_parquet(tmp):
                raise OSError(f"downloaded file {filename} is not valid parquet with a 'text' column")
            os.replace(tmp, save_path)
            return save_path

        except RepositoryNotFoundError as e:
            raise RuntimeError(f"Repo '{repo_id}' not found. Check DATASETS_CHOICES for a typo or a removed dataset.") from e
        except EntryNotFoundError as e:
            raise RuntimeError(f"File '{filename}' not found in '{repo_id}'. Shard list may be stale.") from e
        except GatedRepoError as e:
            raise RuntimeError(f"Access denied to '{repo_id}'. Set HF_TOKEN and accept the dataset terms on huggingface.co.") from e
        except (HfHubHTTPError, requests.exceptions.RequestException, OSError) as e:
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  attempt {attempt}/{max_attempts} failed for {filename}: {e}. retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  FAILED {repo_id}/{filename} after {max_attempts} attempts: {e}")
                return None


def download(domain, num_shards):
    assert domain in DATASETS_CHOICES, f"Choose a domain from {show_available_choices()}"
    assert num_shards > 0, "num_shards must be positive"
    info = DATASETS_CHOICES[domain]

    domain_dir = os.path.join(DATA_DIR, "parquet", domain_to_dirname(domain))
    staging_dir = os.path.join(domain_dir, "_staging")
    os.makedirs(domain_dir, exist_ok=True)
    os.makedirs(staging_dir, exist_ok=True)

    remote = list_remote_shards(domain)
    if not remote:
        raise RuntimeError(f"No parquet files found for '{domain}'")
    if num_shards > len(remote):
        print(f"  asked for {num_shards} shards but only {len(remote)} exist, using {len(remote)}")
        num_shards = len(remote)

    done, skipped, failed = [], [], []
    try:
        for k in range(num_shards):
            save_path = os.path.join(domain_dir, idx_to_path(k))
            if is_valid_parquet(save_path):
                print(f"[{k + 1}/{num_shards}] exists, skipping {idx_to_path(k)}")
                skipped.append(save_path)
                continue
            print(f"[{k + 1}/{num_shards}] downloading {remote[k]}")
            result = download_shard_with_retry(info["repo_id"], remote[k], save_path, staging_dir)
            (done if result else failed).append(save_path)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    # manifest lists only shards that really exist; build script must read this, not range(n)
    shards = []
    for k in range(num_shards):
        p = os.path.join(domain_dir, idx_to_path(k))
        if is_valid_parquet(p):
            shards.append({
                "file": idx_to_path(k),
                "remote_path": remote[k],
                "rows": pq.ParquetFile(p).metadata.num_rows,
                "bytes": os.path.getsize(p),
            })
    manifest = {"domain": domain, "repo_id": info["repo_id"], "shards": shards}
    with open(os.path.join(domain_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "domain": domain,
        "downloaded": len(done),
        "skipped_existing": len(skipped),
        "failed": [os.path.basename(p) for p in failed],
        "total_rows": sum(s["rows"] for s in shards),
        "dir": domain_dir,
    }


def main(domains, num_shards_list):
    out = {}
    for domain, num_shards in zip(domains, num_shards_list):
        print(f"\n=== {domain} ({num_shards} shards) ===")
        out[domain] = download(domain, num_shards)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dataset downloader")
    parser.add_argument("--list", action="store_true", help="Show available domains and exit.")
    parser.add_argument("--domains", nargs="+", type=str, default=None, help="Domain names, space-separated.")
    parser.add_argument("--num-shards", nargs="+", type=int, default=None,
                        help="Shards to download per domain, same order as --domains.")
    args = parser.parse_args()

    if args.list:
        print("Available domains:")
        for choice in show_available_choices():
            print(f"  {choice}")
        raise SystemExit(0)

    assert args.domains is not None and args.num_shards is not None, "Both --domains and --num-shards are required"
    assert len(args.domains) == len(args.num_shards), "Give one shard count per domain"

    result = main(args.domains, args.num_shards)
    print("\nDone:")
    print(json.dumps(result, indent=2))

    if any(r["failed"] for r in result.values()):
        print("\nSome shards failed. Rerun the same command to retry only those.")
        sys.exit(1)