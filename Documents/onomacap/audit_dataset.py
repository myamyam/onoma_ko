"""Audit Korean labels, audio metadata and optional exact file duplicates."""
import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf

from onomacap.config import load_config
from onomacap.data import check_split_overlap, read_records
from onomacap.tokenizer import build_tokenizer, validate_labels


def audit(config, hash_audio=False):
    splits = {s: read_records(config, s) for s in ["train", "valid", "test"]}
    check_split_overlap(splits)
    tokenizer = build_tokenizer(config["model"]["text_model"])
    result, digests, duplicate_groups = {}, {}, {}
    for split, records in splits.items():
        lengths, rates = [], set()
        for record in records:
            info = sf.info(record["path"])
            if not info.frames or not info.samplerate or info.duration > config["audio"]["max_seconds"]:
                raise ValueError(f"Invalid duration: {record['path']} ({info.duration}s)")
            lengths.append(info.duration)
            rates.add(info.samplerate)
            if hash_audio:
                digest = hashlib.sha256()
                with open(record["path"], "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                key = digest.hexdigest()
                if key in digests and digests[key][0] != split:
                    duplicate_groups.setdefault(key, [digests[key]]).append((split, record["audio_file"]))
                else:
                    digests[key] = (split, record["audio_file"])
        result[split] = {"audio_count": len(records), "min_seconds": min(lengths),
                         "max_seconds": max(lengths), "sample_rates": sorted(rates),
                         **validate_labels(tokenizer, records, config["model"]["max_target_tokens"])}
        print(f"Audited {split}: {len(records)} audio files", flush=True)
    result["cross_split_identical_files"] = list(duplicate_groups.values()) if hash_audio else None
    result["notes"] = "File hashes detect exact duplicates only; recording provenance/near duplicates need separate review."
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/korean.yaml")
    parser.add_argument("--hash-audio", action="store_true")
    parser.add_argument("--output", default="outputs/dataset_audit.json")
    args = parser.parse_args()
    report = audit(load_config(args.config), args.hash_audio)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["cross_split_identical_files"]:
        raise SystemExit("Cross-split duplicate audio detected; resolve before training.")
