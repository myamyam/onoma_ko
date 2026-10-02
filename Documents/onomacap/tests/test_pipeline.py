import copy
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import BartConfig, AutoTokenizer

from onomacap.config import ROOT, load_config
from onomacap.data import encode_targets, load_audio, normalize_text, check_split_overlap
from onomacap.metrics import score_prediction
from onomacap.model import KoreanOnomaModel
from onomacap.tokenizer import build_tokenizer
from prepare_dataset import prepare


class DataTests(unittest.TestCase):
    def test_resample_stereo_and_padding(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tone.wav"
            t = np.arange(8000) / 16000
            wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)
            sf.write(path, np.stack([wave, wave], axis=1), 16000, subtype="FLOAT")
            result, length = load_audio(path, {"sr": 32000, "max_seconds": 1})
            self.assertEqual(length, 16000)
            self.assertEqual(result.shape, (32000,))
            self.assertTrue(torch.all(result[length:] == 0))
            # Resampling preserves frequency rather than compressing/stretching playback.
            peak = np.argmax(np.abs(np.fft.rfft(result[:length].numpy()))) * 32000 / length
            self.assertAlmostEqual(peak, 440)

    def test_long_audio_is_not_silently_cropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "long.wav"
            sf.write(path, np.zeros(32001), 32000)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_audio(path, {"sr": 32000, "max_seconds": 1})

    def test_normalization_preserves_repetition_and_jamo(self):
        self.assertEqual(normalize_text("  드 르 르 ㅡ  "), "드 르 르 ㅡ")
        self.assertEqual(normalize_text("드"), "드")

    def test_multi_reference_metrics(self):
        result = score_prediction("드르륵", ["탕", "드 르 륵"])
        self.assertEqual(result["min_cer"], 0)
        self.assertEqual(result["exact_match"], 0)
        self.assertEqual(result["exact_match_no_spaces"], 1)
        self.assertGreater(result["min_cer_with_spaces"], 0)
        self.assertGreater(score_prediction("드 르", ["드 르 르"])["min_cer"], 0)

    def test_split_leakage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            check_split_overlap({"train": [{"audio_file": "a"}], "test": [{"audio_file": "a"}]})

    def test_preparation_keeps_test_and_preserves_originals(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, target = root / "source", root / "prepared"
            source.mkdir()
            for split in ["train", "valid", "test"]:
                (root / f"{split}.wav").write_bytes(b"identical contents")
                (root / f"{split}_unique.wav").write_bytes(split.encode())
                with (source / f"{split}.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["audio_file"] + [f"B{i}_text" for i in range(1, 6)])
                    writer.writeheader()
                    for name in [f"{split}.wav", f"{split}_unique.wav"]:
                        writer.writerow({"audio_file": name, **{f"B{i}_text": "쿵" for i in range(1, 6)}})
            original = (source / "train.csv").read_bytes()
            config = {"data": {"source_csv_dir": str(source), "csv_dir": str(target),
                                "audio_dir": str(root), "csv_pattern": "{split}.csv"}}
            result = prepare(config)
            self.assertEqual(result["prepared_counts"], {"test": 2, "valid": 1, "train": 1})
            self.assertEqual((source / "train.csv").read_bytes(), original)
            self.assertEqual(len(result["removed"]), 2)


@unittest.skipUnless((ROOT / "models/kobart-base-v2/tokenizer.json").exists(), "Download KoBART first")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = load_config(ROOT / "configs/korean.yaml")
        cls.tokenizer = build_tokenizer(cls.config["model"]["text_model"])

    def test_rare_hangul_and_jamo_survive_save_reload(self):
        text = "즹 즹 즤 잉 뚀 콬 ㅡ 드 르 르 르"
        with tempfile.TemporaryDirectory() as folder:
            self.tokenizer.save_pretrained(folder)
            restored = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            ids = restored.encode(text, add_special_tokens=False)
            self.assertNotIn(restored.unk_token_id, ids)
            self.assertEqual(restored.decode(ids, clean_up_tokenization_spaces=False), text)
            self.assertEqual(ids, self.tokenizer.encode(text, add_special_tokens=False))

    def test_target_eos_padding_and_overflow(self):
        labels = encode_targets(self.tokenizer, ["쿵", "드 르 르 르 르 륵"], 64)
        self.assertEqual(labels[0, labels[0].ne(-100).sum() - 1], self.tokenizer.eos_token_id)
        self.assertTrue(labels[0].eq(-100).any())
        with self.assertRaisesRegex(ValueError, "limit"):
            encode_targets(self.tokenizer, ["드 르 르 르 륵"], 2)

    def test_cross_attention_gradients_freezing_and_generation(self):
        # Real HTSAT plus a small random BART keeps the integration check inexpensive.
        decoder = BartConfig(vocab_size=len(self.tokenizer), d_model=32,
                             encoder_layers=1, decoder_layers=1,
                             encoder_attention_heads=2, decoder_attention_heads=2,
                             encoder_ffn_dim=64, decoder_ffn_dim=64,
                             max_position_embeddings=64).to_dict()
        model = KoreanOnomaModel(copy.deepcopy(self.config), self.tokenizer,
                                 decoder_config=decoder, pretrained=False)
        model.set_encoder_stage(1)
        model.train()
        self.assertFalse(model.encoder.bn0.training)
        self.assertFalse(model.encoder.layers[0].training)
        self.assertTrue(model.encoder.layers[-1].training)
        waveforms = torch.randn(1, 320000) * 0.1
        lengths = torch.tensor([64000])
        labels = encode_targets(self.tokenizer, ["즹 드 르 르"], 64)
        loss = model(waveforms, lengths, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.projection[0].weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.encoder.layers[-1].parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.layers[0].parameters()))
        self.assertGreater(model.language.model.decoder.layers[0].encoder_attn.k_proj.weight.grad.abs().sum().item(), 0)
        model.eval()
        with torch.no_grad():
            base_loss = model(waveforms, lengths, labels)
            padded_loss = model(waveforms, lengths, torch.cat([labels, torch.full((1, 3), -100)], dim=1))
        self.assertTrue(torch.allclose(base_loss, padded_loss, atol=1e-5))
        tokens = model.generate(waveforms, lengths, max_new_tokens=4, num_beams=2)
        self.assertEqual(tokens.shape[0], 1)
        self.assertLessEqual(tokens.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
