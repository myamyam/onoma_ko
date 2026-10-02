import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .metrics import aggregate_scores, score_prediction
from .model import KoreanOnomaModel


def load_checkpoint(path, device):
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path.parent / "tokenizer", local_files_only=True)
    model = KoreanOnomaModel(checkpoint["config"], tokenizer=tokenizer,
                             decoder_config=checkpoint["decoder_config"], pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), tokenizer, checkpoint


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, generation, output_path=None):
    was_training = model.training
    model.eval()
    rows = []
    for batch in loader:
        tokens = model.generate(batch["waveforms"].to(device), batch["lengths"].to(device), **generation)
        predictions = tokenizer.batch_decode(tokens, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)
        for name, prediction, refs in zip(batch["audio_files"], predictions, batch["references"]):
            rows.append({"audio_file": name, "prediction": prediction, "references": refs,
                         **score_prediction(prediction, refs)})
    model.train(was_training)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return aggregate_scores(rows)
