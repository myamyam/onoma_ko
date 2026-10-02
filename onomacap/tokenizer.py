"""Extend KoBART's BPE alphabet without learning vocabulary from held-out labels."""
import json

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def build_tokenizer(model_path):
    original = AutoTokenizer.from_pretrained(model_path)
    backend = json.loads(original.backend_tokenizer.to_str())
    if backend["model"]["type"] != "BPE":
        raise ValueError("Expected a KoBART BPE tokenizer")
    # NFKC changes compatibility jamo (e.g. ㅡ). NFC preserves those annotations.
    backend["normalizer"] = {"type": "NFC"}
    vocab = backend["model"]["vocab"]
    # Fixed Unicode ranges, independent of train/validation/test contents.
    # Extend BPE's base alphabet rather than AddedToken matching, preserving spaces/merges.
    for start, stop in [(0xAC00, 0xD7A4), (0x1100, 0x1200), (0x3131, 0x318F),
                        (0xA960, 0xA97D), (0xD7B0, 0xD7FC)]:
        for codepoint in range(start, stop):
            char = chr(codepoint)
            if char not in vocab:
                vocab[char] = len(vocab)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(backend, ensure_ascii=False)),
        **original.special_tokens_map,
        model_max_length=original.model_max_length,
        clean_up_tokenization_spaces=False,
    )
    return tokenizer


def validate_labels(tokenizer, records, max_tokens):
    texts = [text for record in records for text in record["references"]]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    decoded = tokenizer.batch_decode(encoded, clean_up_tokenization_spaces=False)
    unknown = sum(tokenizer.unk_token_id in ids for ids in encoded)
    failures = [(a, b) for a, b in zip(texts, decoded) if a != b]
    maximum = max(len(ids) + 1 for ids in encoded)
    if unknown or failures or maximum > max_tokens:
        raise ValueError(f"Tokenizer audit failed: unknown={unknown}, roundtrip={len(failures)}, "
                         f"max_tokens={maximum}/{max_tokens}, examples={failures[:3]}")
    return {"labels": len(texts), "unknown_labels": unknown, "roundtrip_failures": len(failures),
            "max_tokens_including_eos": maximum, "vocab_size": len(tokenizer)}
