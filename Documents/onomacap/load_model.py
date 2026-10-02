"""Download the model; use --then-data to download the dataset afterwards."""

import argparse
from pathlib import Path
import subprocess
import sys

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--then-data", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    model_dir = root / "models" / "onoma_ende"
    print(f"Downloading model to: {model_dir}", flush=True)
    snapshot_download(
        repo_id="myamyam/onoma_ende",
        revision="main",
        local_dir=model_dir,
    )
    print(f"Model download complete: {model_dir}", flush=True)
    if args.then_data:
        print("Starting dataset download...", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(root / "load_dataset.py")], check=True
        )
