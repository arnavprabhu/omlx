# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR loading respects quantized and floating-point audio weights."""

import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.engine.stt import STTEngine


def test_qwen3_asr_quantization_patch_is_idempotent(monkeypatch):
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

    from omlx.patches.mlx_audio_compat import ensure_qwen3_asr_audio_quantization

    monkeypatch.setattr(Qwen3ASRModel, "model_quant_predicate", lambda *args: False)
    assert ensure_qwen3_asr_audio_quantization()
    predicate = Qwen3ASRModel.model_quant_predicate
    assert ensure_qwen3_asr_audio_quantization()
    assert Qwen3ASRModel.model_quant_predicate is predicate
    assert predicate(None, "audio_tower.conv_out", None)
    assert not predicate(None, "model.embed_tokens", None)


@pytest.mark.parametrize("quantization_key", ["quantization", "quantization_config"])
@pytest.mark.parametrize("quantized_audio", [False, True])
async def test_stt_loads_qwen3_asr_audio_quantization(
    tmp_path, monkeypatch, quantization_key, quantized_audio
):
    from mlx_audio.stt.models.qwen3_asr.config import ModelConfig
    from mlx_audio.stt.models.qwen3_asr.qwen3_asr import Qwen3ASRModel

    # Restore the predicate after the engine installs its compatibility patch.
    monkeypatch.setattr(
        Qwen3ASRModel, "model_quant_predicate", Qwen3ASRModel.model_quant_predicate
    )
    monkeypatch.setattr(
        Qwen3ASRModel,
        "post_load_hook",
        classmethod(lambda cls, model, path: model),
    )
    config = {
        "model_type": "qwen3_asr",
        "audio_config": {
            "num_mel_bins": 8,
            "d_model": 32,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 32,
            "output_dim": 32,
            "max_source_positions": 16,
        },
        "text_config": {
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 16,
        },
    }
    quantization = {"group_size": 32, "bits": 4}
    if quantized_audio:
        quantization["audio_tower.conv_out"] = {"group_size": 32, "bits": 8}
    source = Qwen3ASRModel(ModelConfig.from_dict(config))
    nn.quantize(
        source,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: (
            quantization["audio_tower.conv_out"]
            if quantized_audio and p == "audio_tower.conv_out"
            else p == "model.embed_tokens"
        ),
    )
    config[quantization_key] = quantization
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(source.parameters()))
    )
    inputs = mx.ones((1, 2, 32))
    expected = source.audio_tower.conv_out(inputs)
    mx.eval(expected)

    engine = STTEngine(str(tmp_path))
    try:
        await engine.start()
        projection = engine._model.audio_tower.conv_out
        assert isinstance(projection, nn.QuantizedLinear) == quantized_audio
        if quantized_audio:
            assert projection.bits == 8
        assert engine._model.model.embed_tokens.bits == 4
        assert isinstance(engine._model.audio_tower.proj1, nn.Linear)
        assert mx.allclose(projection(inputs), expected).item()
    finally:
        await engine.stop()
