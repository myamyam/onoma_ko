from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    # Paths are relative to this project, independent of the caller's cwd.
    for section, key in [("data", "csv_dir"), ("data", "audio_dir"),
                         ("model", "htsat_checkpoint")]:
        config[section][key] = str(resolve_path(config[section][key]))
    if "source_csv_dir" in config["data"]:
        config["data"]["source_csv_dir"] = str(resolve_path(config["data"]["source_csv_dir"]))
    config["output_dir"] = str(resolve_path(config["output_dir"]))
    local_model = resolve_path(config["model"]["text_model"])
    if local_model.is_dir():
        config["model"]["text_model"] = str(local_model)
    return config


def resolve_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def get_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)
