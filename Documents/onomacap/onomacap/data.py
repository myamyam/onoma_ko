import csv
import math
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset


def normalize_text(text):
    # Internal spaces, punctuation and repeated syllables carry annotation meaning.
    return unicodedata.normalize("NFC", text).strip()


def read_records(config, split):
    data = config["data"]
    path = Path(data["csv_dir"]) / data["csv_pattern"].format(split=split)
    records, seen = [], set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"audio_file", *[f"B{i}_text" for i in range(1, 6)]}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing Korean label columns in {path}")
        root = Path(data["audio_dir"]).resolve()
        for row in reader:
            name = row["audio_file"]
            audio_path = (root / name).resolve()
            if root not in audio_path.parents:
                raise ValueError(f"Audio path outside dataset: {name}")
            if name in seen:
                raise ValueError(f"Duplicate audio row in {split}: {name}")
            seen.add(name)
            refs = [normalize_text(row[f"B{i}_text"] or "") for i in range(1, 6)]
            if not all(refs):
                raise ValueError(f"Empty Korean reference: {name}")
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            records.append({"audio_file": name, "path": str(audio_path),
                            "references": refs, "class": row.get("class", "")})
    if not records:
        raise ValueError(f"Empty split: {split}")
    return records


def check_split_overlap(splits):
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = {r["audio_file"] for r in splits[left]} & {r["audio_file"] for r in splits[right]}
            if overlap:
                raise ValueError(f"Audio leakage between {left}/{right}: {sorted(overlap)[:5]}")


def load_audio(path, config):
    wave, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if len(wave) == 0 or not np.isfinite(wave).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    wave = wave.mean(axis=1)
    target_rate = config["sr"]
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        wave = resample_poly(wave, target_rate // divisor, sample_rate // divisor)
    # Never silently discard events whose labels describe the complete clip.
    max_samples = round(target_rate * config["max_seconds"])
    if len(wave) > max_samples:
        raise ValueError(f"Audio exceeds {config['max_seconds']} seconds: {path}. "
                         "Create aligned segments before training; automatic cropping is disabled.")
    length = len(wave)
    wave = np.pad(wave, (0, max_samples - length))
    return torch.from_numpy(wave.astype(np.float32)), length


class OnomaDataset(Dataset):
    def __init__(self, records, audio_config, training=False):
        self.records = records
        self.audio_config = audio_config
        self.training = training

    def __len__(self):
        return len(self.records) * (5 if self.training else 1)

    def __getitem__(self, index):
        record = self.records[index // 5 if self.training else index]
        wave, length = load_audio(record["path"], self.audio_config)
        return {**record, "waveform": wave, "length": length,
                "text": record["references"][index % 5 if self.training else 0]}


def encode_targets(tokenizer, texts, max_tokens):
    sequences = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.unk_token_id is not None and tokenizer.unk_token_id in ids:
            raise ValueError(f"Unknown token in Korean label: {text!r}")
        ids = ids + [tokenizer.eos_token_id]
        if len(ids) > max_tokens:
            raise ValueError(f"Label needs {len(ids)} tokens (limit {max_tokens}): {text!r}")
        sequences.append(ids)
    labels = torch.full((len(sequences), max(map(len, sequences))), -100, dtype=torch.long)
    for i, ids in enumerate(sequences):
        labels[i, :len(ids)] = torch.tensor(ids)
    return labels


class Collator:
    def __init__(self, tokenizer, max_tokens, training=False):
        self.tokenizer, self.max_tokens, self.training = tokenizer, max_tokens, training

    def __call__(self, samples):
        batch = {"waveforms": torch.stack([s["waveform"] for s in samples]),
                 "lengths": torch.tensor([s["length"] for s in samples]),
                 "audio_files": [s["audio_file"] for s in samples],
                 "references": [s["references"] for s in samples]}
        if self.training:
            batch["labels"] = encode_targets(self.tokenizer, [s["text"] for s in samples], self.max_tokens)
        return batch
