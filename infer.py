"""Generate Korean onomatopoeia from one or more audio files."""
import argparse
import json
from pathlib import Path

import torch

from onomacap.config import get_device
from onomacap.data import load_audio
from onomacap.runtime import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/korean_htsat_kobart/best.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    generation = dict(config["generation"])
    if args.num_beams is not None:
        generation["num_beams"] = args.num_beams
    for path in args.audio:
        waveform, length = load_audio(path, config["audio"])
        ids = model.generate(waveform[None].to(device), torch.tensor([length], device=device), **generation)
        text = tokenizer.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(json.dumps({"audio_file": str(path), "prediction": text}, ensure_ascii=False))


if __name__ == "__main__":
    main()
