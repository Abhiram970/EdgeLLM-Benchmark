"""
Pre-download the four 1.5B model weights so the first eval run doesn't stall.
All four repos are openly downloadable (no gating). `huggingface-cli login` is
optional but helps with rate limits.

    python download.py            # download all four
    python download.py --model deepscaler-1.5b
"""
import argparse

from huggingface_hub import snapshot_download

from models import MODELS, get


def fetch(name: str):
    spec = get(name)
    print(f"downloading {spec.name}  <-  {spec.repo}")
    path = snapshot_download(repo_id=spec.repo)
    print(f"  -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS))
    args = ap.parse_args()
    names = [args.model] if args.model else list(MODELS)
    for n in names:
        fetch(n)
    print("done.")


if __name__ == "__main__":
    main()
