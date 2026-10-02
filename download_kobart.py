"""Download the Korean decoder and tokenizer into the project."""
from huggingface_hub import snapshot_download

from onomacap.config import ROOT


if __name__ == "__main__":
    snapshot_download("gogamza/kobart-base-v2", local_dir=ROOT / "models/kobart-base-v2",
                      allow_patterns=["*.json", "*.txt", "*.safetensors"])
