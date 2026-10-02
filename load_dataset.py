"""Download the dataset repository locally: python3 load_dataset.py."""

from pathlib import Path

from huggingface_hub import snapshot_download


if __name__ == "__main__":
    data_dir = Path(__file__).resolve().parent / "data" / "onoma"
    print(f"Downloading myamyam/onoma to: {data_dir}", flush=True)
    downloaded_path = snapshot_download(
        repo_id="myamyam/onoma",
        repo_type="dataset",
        revision="main",
        local_dir=data_dir,
        max_workers=32,
    )
    print(f"Dataset downloaded to: {downloaded_path}")
