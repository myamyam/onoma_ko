"""Evaluate one prediction per clip against all five Korean references."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from onomacap.config import get_device, load_config
from onomacap.data import Collator, OnomaDataset, read_records
from onomacap.runtime import evaluate, load_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--config", help="Optional dataset location override for another machine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.jsonl"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    if args.config:
        config["data"] = load_config(args.config)["data"]
    records = read_records(config, args.split)
    loader = DataLoader(OnomaDataset(records, config["audio"]),
                        batch_size=config["training"]["eval_batch_size"],
                        collate_fn=Collator(tokenizer, config["model"]["max_target_tokens"]))
    scores = evaluate(model, loader, tokenizer, device, config["generation"], args.output)
    args.output.with_suffix(".metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))
