"""Train HTSAT -> KoBART on original Korean onomatopoeia labels."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from onomacap.config import get_device, load_config
from onomacap.data import Collator, OnomaDataset, check_split_overlap, read_records
from onomacap.model import KoreanOnomaModel
from onomacap.runtime import evaluate, load_checkpoint
from onomacap.tokenizer import build_tokenizer, validate_labels


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/korean.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overfit", type=int, default=0, metavar="N",
                        help="Train/evaluate the first N training clips in a separate output directory")
    parser.add_argument("--smoke-test", action="store_true", help="One optimizer step and checkpoint verification")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.overfit < 0 or (args.overfit and args.smoke_test):
        parser.error("Choose either --overfit N (N > 0) or --smoke-test")
    if args.resume and (args.overfit or args.smoke_test or args.epochs or args.output_dir):
        parser.error("Resume uses saved settings; do not combine with experiment overrides")
    torch.set_num_threads(args.threads)
    config = load_config(args.config)
    device = get_device(args.device or config["device"])
    checkpoint = None
    if args.resume:
        model, tokenizer, checkpoint = load_checkpoint(args.resume, device)
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires last.pt; best.pt is for inference")
        config = checkpoint["config"]
        output = args.resume.resolve().parent
    else:
        if args.epochs is not None:
            config["training"]["epochs"] = args.epochs
        output = args.output_dir or Path(config["output_dir"])
        if args.smoke_test or args.overfit:
            output = output / ("smoke" if args.smoke_test else f"overfit_{args.overfit}")
            config["training"]["freeze_encoder_epochs"] = config["training"]["epochs"]
        if args.smoke_test:
            config["training"].update(epochs=1, batch_size=1, accumulation_steps=1)
            config["generation"].update(max_new_tokens=8, num_beams=1)
        config["experiment"] = {"smoke_test": args.smoke_test, "overfit": args.overfit}
        config["output_dir"] = str(output.resolve())
        if (output / "last.pt").exists() or (output / "best.pt").exists():
            raise FileExistsError(f"Existing run in {output}; use --resume or a new --output-dir")
        tokenizer = build_tokenizer(config["model"]["text_model"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        model = KoreanOnomaModel(config, tokenizer).to(device)
    training = config["training"]
    if min(training[k] for k in ["epochs", "batch_size", "accumulation_steps"]) < 1:
        raise ValueError("Epochs, batch size and accumulation steps must be positive")
    splits = {s: read_records(config, s) for s in ["train", "valid", "test"]}
    check_split_overlap(splits)
    for records in splits.values():
        validate_labels(tokenizer, records, config["model"]["max_target_tokens"])
    smoke = config.get("experiment", {}).get("smoke_test", False)
    overfit = config.get("experiment", {}).get("overfit", 0)
    if smoke or overfit:
        splits["train"] = splits["train"][:overfit or 1]
        splits["valid"] = splits["train"]
    generator = torch.Generator().manual_seed(config["seed"])
    loader_args = {"num_workers": training["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(OnomaDataset(splits["train"], config["audio"], training=True),
                              batch_size=training["batch_size"], shuffle=True, generator=generator,
                              collate_fn=Collator(tokenizer, config["model"]["max_target_tokens"], True),
                              **loader_args)
    valid_loader = DataLoader(OnomaDataset(splits["valid"], config["audio"]),
                              batch_size=training["eval_batch_size"],
                              collate_fn=Collator(tokenizer, config["model"]["max_target_tokens"]), **loader_args)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": training["encoder_lr"]},
        {"params": model.projection.parameters(), "lr": training["projection_lr"]},
        {"params": model.language.parameters(), "lr": training["decoder_lr"]},
    ], weight_decay=training["weight_decay"])
    accumulation = training["accumulation_steps"]
    steps_per_epoch = 1 if smoke else math.ceil(len(train_loader) / accumulation)
    total_steps = steps_per_epoch * training["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * training["warmup_ratio"]), total_steps)
    amp = training["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, stale, step = 0, float("inf"), 0, 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best, stale, step = (checkpoint[k] for k in ["next_epoch", "best_cer", "stale", "step"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        del checkpoint
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output / "tokenizer")
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"device={device}, train_clips={len(splits['train'])}, output={output}", flush=True)
    if start >= training["epochs"]:
        print("This run has already completed its configured epochs.")
        return
    if stale >= training["patience"]:
        print("This run has already reached its early-stopping patience.")
        return
    for epoch in range(start, training["epochs"]):
        blocks = 0 if epoch < training["freeze_encoder_epochs"] else training["unfreeze_last_blocks"]
        model.set_encoder_stage(blocks)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, count, group_tokens = 0.0, 0, 0
        for batch_index, batch in enumerate(train_loader):
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = model(batch["waveforms"].to(device), batch["lengths"].to(device), batch["labels"].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {batch_index}")
            # Token-weighted accumulation handles variable label lengths and the final partial group.
            tokens = int(batch["labels"].ne(-100).sum())
            scaler.scale(loss * tokens).backward()
            loss_sum += loss.item() * tokens
            count += tokens
            group_tokens += tokens
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader) or smoke:
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
                step += 1
                if step % 20 == 0 or smoke:
                    print(f"epoch={epoch + 1} step={step} token_loss={loss_sum / count:.4f}", flush=True)
            if smoke:
                break
        metrics = evaluate(model, valid_loader, tokenizer, device, config["generation"],
                           output / f"valid_epoch_{epoch + 1}.jsonl")
        improved = metrics["min_cer"] < best
        best = min(best, metrics["min_cer"])
        stale = 0 if improved else stale + 1
        log = {"epoch": epoch + 1, "step": step, "train_loss": loss_sum / count,
               "unfrozen_blocks": blocks, **metrics}
        print(json.dumps(log, ensure_ascii=False), flush=True)
        with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log) + "\n")
        payload = {"format_version": 1, "config": config, "decoder_config": model.language.config.to_dict(),
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                   "next_epoch": epoch + 1, "best_cer": best, "stale": stale, "step": step,
                   "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                   "loader_rng": generator.get_state(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        save_checkpoint(output / "last.pt", payload)
        if improved:
            save_checkpoint(output / "best.pt", {k: payload[k] for k in
                            ["format_version", "config", "decoder_config", "model"]})
        if stale >= training["patience"]:
            break
    if smoke:
        del optimizer, scheduler, payload, model
        restored, _, _ = load_checkpoint(output / "best.pt", device)
        batch = next(iter(valid_loader))
        result = restored.generate(batch["waveforms"].to(device), batch["lengths"].to(device), **config["generation"])
        print("Checkpoint reload/generation passed:", tokenizer.batch_decode(result, skip_special_tokens=True), flush=True)


if __name__ == "__main__":
    main()
