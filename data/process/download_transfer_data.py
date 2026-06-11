"""Download Epilepsy and Gesture transfer targets from Figshare.

Article-level ndownloader URLs (e.g. .../articles/19930199/versions/2) return
HTTP 202 with an empty body. This script uses the Figshare API to fetch per-file
download URLs, then writes processed numpy splits under data/{epilepsy,gesture}/processed/.
"""

import os
from pathlib import Path

import numpy as np
import requests
import torch
from einops import rearrange
from data.process.utils import download_file

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "epilepsy": 19930199,
    "gesture": 19930247,
}


def figshare_files(article_id: int) -> dict[str, str]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return {f["name"]: f["download_url"] for f in response.json()["files"]}


def download_dataset(name: str, article_id: int, redownload: bool = False):
    raw_dir = REPO_ROOT / "data" / name
    processed_dir = raw_dir / "processed"
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    files = figshare_files(article_id)
    for split in ("train", "val", "test"):
        pt_name = f"{split}.pt"
        if pt_name not in files:
            raise KeyError(f"{name}: missing {pt_name} in Figshare article {article_id}")

        pt_path = raw_dir / pt_name
        if not pt_path.exists() or redownload:
            print(f"Downloading {name}/{pt_name} ...")
            download_file(files[pt_name], str(pt_path))
        else:
            print(f"Using existing {pt_path}")

        data = torch.load(pt_path, weights_only=False)
        subseq = rearrange(data["samples"], "b c t -> b t c").numpy()
        labels = data["labels"].numpy()

        np.save(processed_dir / f"{split}_data_subseq.npy", subseq)
        np.save(processed_dir / f"{split}_labels_subseq.npy", labels)

    print(f"Wrote processed splits to {processed_dir}")


def main(redownload: bool = False):
    for name, article_id in DATASETS.items():
        download_dataset(name, article_id, redownload=redownload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Re-fetch .pt files even if they already exist",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default=None,
        help="Download one dataset only (default: both)",
    )
    args = parser.parse_args()

    items = (
        {args.dataset: DATASETS[args.dataset]}
        if args.dataset
        else DATASETS
    )
    for name, article_id in items.items():
        download_dataset(name, article_id, redownload=args.redownload)
