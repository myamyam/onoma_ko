"""Write separate split CSVs without byte-identical cross-split audio leakage."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path

from onomacap.config import load_config
from onomacap.data import check_split_overlap, read_records


def prepare(config):
    source_config = copy.deepcopy(config)
    source_dir = Path(config["data"]["source_csv_dir"])
    target_dir = Path(config["data"]["csv_dir"])
    if source_dir.resolve() == target_dir.resolve():
        raise ValueError("Prepared CSV directory must differ from the original directory")
    source_config["data"]["csv_dir"] = str(source_dir)
    splits = {s: read_records(source_config, s) for s in ["test", "valid", "train"]}
    check_split_overlap(splits)
    owners, kept, removed = {}, {}, []
    # Preserve test examples first, then validation; never move held-out audio into training.
    for split, records in splits.items():
        kept[split] = set()
        for record in records:
            digest = hashlib.sha256()
            with open(record["path"], "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            key = digest.hexdigest()
            if key in owners and owners[key]["split"] != split:
                removed.append({"split": split, "audio_file": record["audio_file"],
                                "sha256": key, "retained": owners[key]})
            else:
                owners.setdefault(key, {"split": split, "audio_file": record["audio_file"]})
                kept[split].add(record["audio_file"])
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        filename = config["data"]["csv_pattern"].format(split=split)
        with (source_dir / filename).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        with (target_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row["audio_file"] in kept[split])
    report = {"policy": "Byte-identical cross-split files removed; priority test > valid > train.",
              "source_counts": {s: len(v) for s, v in splits.items()},
              "prepared_counts": {s: len(v) for s, v in kept.items()}, "removed": removed,
              "limitation": "Different encodings or segments of the same recording need provenance review."}
    (target_dir / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/korean.yaml")
    args = parser.parse_args()
    result = prepare(load_config(args.config))
    print(json.dumps({"prepared_counts": result["prepared_counts"], "removed_count": len(result["removed"])}, indent=2))
