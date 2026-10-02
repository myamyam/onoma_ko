import math
from types import SimpleNamespace

import torch
from torch import nn
from transformers import BartConfig, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from .vendor.htsat import HTSAT_Swin_Transformer


class KoreanOnomaModel(nn.Module):
    def __init__(self, config, tokenizer=None, decoder_config=None, pretrained=True):
        super().__init__()
        self.config = config
        audio = config["audio"]
        if audio["n_mels"] != 64 or audio["max_seconds"] > 10:
            raise ValueError("This HTSAT checkpoint requires 64 mel bins and clips <= 10 seconds")
        htsat_config = SimpleNamespace(to_dict=lambda: {"audio_args": audio, "spec_augment": False})
        self.encoder = HTSAT_Swin_Transformer(config=htsat_config)
        if pretrained:
            self.load_audio_checkpoint(config["model"]["htsat_checkpoint"])
            self.language, info = BartForConditionalGeneration.from_pretrained(
                config["model"]["text_model"], output_loading_info=True)
            missing_decoder = [k for k in info["missing_keys"] if "decoder.layers" in k]
            if missing_decoder:
                raise ValueError(f"Missing pretrained decoder weights: {missing_decoder}")
        else:
            self.language = BartForConditionalGeneration(BartConfig.from_dict(decoder_config))
        if tokenizer is not None:
            if any(x is None for x in [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id]):
                raise ValueError("Tokenizer must define BOS/EOS/PAD")
            if len(tokenizer) != self.language.config.vocab_size:
                if not pretrained:
                    raise ValueError("Saved tokenizer and decoder vocabularies do not match")
                self.language.resize_token_embeddings(len(tokenizer))
            for attr, value in {
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "pad_token_id": tokenizer.pad_token_id,
                "decoder_start_token_id": tokenizer.bos_token_id,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
            }.items():
                setattr(self.language.config, attr, value)
                setattr(self.language.generation_config, attr, value)
        # Audio features directly supply cross-attention memory. Remove the unused text encoder.
        self.language.model.encoder = None
        width = self.language.config.d_model
        self.projection = nn.Sequential(nn.Linear(768, width), nn.LayerNorm(width))
        self.unfrozen_blocks = 0
        self.set_encoder_stage(0)

    def load_audio_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        state = {k.removeprefix("sed_model."): v for k, v in state.items()}
        own = self.encoder.state_dict()
        selected = {k: v for k, v in state.items() if k in own and v.shape == own[k].shape}
        # Only the local, deterministic spectrogram frontend may be absent.
        missing = [k for k in own if k not in selected and not k.startswith("audio_feats_extractor.")]
        if missing:
            raise ValueError(f"Incomplete HTSAT checkpoint: {missing[:10]}")
        self.encoder.load_state_dict(selected, strict=False)

    def set_encoder_stage(self, last_blocks):
        if not 0 <= last_blocks <= len(self.encoder.layers):
            raise ValueError("Invalid number of HTSAT blocks to unfreeze")
        self.unfrozen_blocks = last_blocks
        self.encoder.requires_grad_(False)
        if last_blocks:
            for layer in self.encoder.layers[-last_blocks:]:
                layer.requires_grad_(True)
            self.encoder.norm.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Frozen HTSAT blocks and batch statistics must not drift during decoder training.
        self.encoder.eval()
        if mode and self.unfrozen_blocks:
            for layer in self.encoder.layers[-self.unfrozen_blocks:]:
                layer.train()
            self.encoder.norm.train()
        return self

    def encode_audio(self, waveforms, lengths):
        # Keep log-mel in FP32: its 1e-10 floor underflows in FP16 on padded silence.
        with torch.autocast(device_type=waveforms.device.type, enabled=False):
            mel = self.encoder.audio_feats_extractor(waveforms.float())
        features = self.encoder(mel)
        # Legacy HTSAT repeats each temporal feature 32 times. Remove exact copies.
        features = features[:, ::32, :]
        memory = self.projection(features)
        # Supply explicit temporal order after bypassing BART's positional encoder.
        steps, width = memory.shape[1:]
        positions = torch.arange(steps, device=memory.device).float()[:, None]
        scale = torch.exp(torch.arange(0, width, 2, device=memory.device).float()
                          * (-math.log(10000.0) / width))
        positional = torch.zeros(steps, width, device=memory.device)
        positional[:, 0::2] = torch.sin(positions * scale)
        positional[:, 1::2] = torch.cos(positions * scale[:width // 2])
        memory = memory + positional.to(memory.dtype)
        valid = torch.ceil(lengths.float() / waveforms.shape[1] * steps).long().clamp(1, steps)
        mask = torch.arange(steps, device=memory.device)[None, :] < valid[:, None]
        return BaseModelOutput(last_hidden_state=memory), mask.long()

    def forward(self, waveforms, lengths, labels):
        memory, mask = self.encode_audio(waveforms, lengths)
        decoder_ids = self.language.prepare_decoder_input_ids_from_labels(labels)
        decoder_mask = decoder_ids.ne(self.language.config.pad_token_id).long()
        decoder_mask[:, 0] = 1
        return self.language(encoder_outputs=memory, attention_mask=mask,
                             decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
                             labels=labels, use_cache=False).loss

    @torch.no_grad()
    def generate(self, waveforms, lengths, **kwargs):
        memory, mask = self.encode_audio(waveforms, lengths)
        return self.language.generate(encoder_outputs=memory, attention_mask=mask, **kwargs)
