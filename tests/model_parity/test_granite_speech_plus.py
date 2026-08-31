"""Mandatory deterministic Torch/MLX parity for Granite Speech Plus."""

import mlx.core as mx
import numpy as np
import tokenizers
import torch
import torchaudio
import transformers
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig
from transformers.models.granite.configuration_granite import GraniteConfig
from transformers.models.granite_speech.feature_extraction_granite_speech import (
    GraniteSpeechFeatureExtractor,
)
from transformers.models.granite_speech_plus.configuration_granite_speech_plus import (
    GraniteSpeechPlusConfig,
    GraniteSpeechPlusEncoderConfig,
)
from transformers.models.granite_speech_plus.modeling_granite_speech_plus import (
    GraniteSpeechPlusForConditionalGeneration,
)

from mlx_audio.stt.models.granite_speech.config import (
    EncoderConfig,
    ModelConfig,
    ProjectorConfig,
    TextConfig,
)
from mlx_audio.stt.models.granite_speech.granite_speech import Model

EXPECTED_TORCH = "2.13.0"
EXPECTED_TORCHAUDIO = "2.11.0"
EXPECTED_TRANSFORMERS = "5.16.1"
EXPECTED_TOKENIZERS = "0.23.1"


def _base_version(version: str) -> str:
    return version.split("+", 1)[0]


def _configs():
    encoder = dict(
        input_dim=160,
        # Preserve the published checkpoint's 16-layer/layer-3 contract while
        # keeping hidden dimensions small enough for every pull request.
        num_layers=16,
        hidden_dim=8,
        feedforward_mult=2,
        num_heads=2,
        dim_head=4,
        output_dim=4,
        context_size=8,
        max_pos_emb=16,
        dropout=0.0,
        conv_kernel_size=5,
        conv_expansion_factor=2,
        cat_hidden_layers=[3],
    )
    projector = dict(
        hidden_size=8,
        encoder_hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        cross_attention_frequency=1,
        layer_norm_eps=1e-12,
    )
    text = dict(
        model_type="granite",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        mlp_bias=False,
        attention_multiplier=0.5,
        embedding_multiplier=1.5,
        residual_multiplier=0.75,
        logits_scaling=2.0,
        tie_word_embeddings=True,
    )

    hf_config = GraniteSpeechPlusConfig(
        text_config=GraniteConfig(**text),
        encoder_config=GraniteSpeechPlusEncoderConfig(**encoder),
        projector_config=Blip2QFormerConfig(**projector),
        audio_token_index=31,
        has_lora_adapter=False,
        downsample_rate=5,
        window_size=15,
        tie_word_embeddings=True,
    )
    mlx_config = ModelConfig(
        model_type="granite_speech_plus",
        encoder_config=EncoderConfig(**encoder),
        projector_config=ProjectorConfig(**projector),
        text_config=TextConfig(**text),
        audio_token_index=31,
        has_lora_adapter=False,
        downsample_rate=5,
        window_size=15,
    )
    return hf_config, mlx_config


def _checkpoint_weights(reference_model):
    """Convert generated HF names to the published checkpoint name contract."""
    weights = {}
    for name, value in reference_model.state_dict().items():
        if name == "lm_head.weight" or name.endswith("num_batches_tracked"):
            # The published checkpoint stores tied embeddings only once, and
            # MLX BatchNorm has no PyTorch bookkeeping counter.
            continue
        if name.startswith(("model.encoder.", "model.projector.")):
            name = name.removeprefix("model.")
        elif name.startswith("model.language_model."):
            name = "language_model.model." + name.removeprefix("model.language_model.")
        else:
            raise AssertionError(f"Unhandled reference weight: {name}")
        weights[name] = mx.array(value.detach().numpy())
    return Model.sanitize(weights)


def _assert_close(stage, mlx_value, torch_value, *, atol):
    actual = np.asarray(mlx_value)
    expected = torch_value.detach().float().cpu().numpy()
    assert (
        actual.shape == expected.shape
    ), f"{stage} shape mismatch: MLX {actual.shape}, Torch {expected.shape}"
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-4, err_msg=stage)


def test_reference_dependency_versions_are_exact():
    assert _base_version(torch.__version__) == EXPECTED_TORCH
    assert _base_version(torchaudio.__version__) == EXPECTED_TORCHAUDIO
    assert _base_version(transformers.__version__) == EXPECTED_TRANSFORMERS
    assert _base_version(tokenizers.__version__) == EXPECTED_TOKENIZERS


def test_granite_plus_full_stack_matches_transformers():
    torch.manual_seed(0)
    hf_config, mlx_config = _configs()
    reference = GraniteSpeechPlusForConditionalGeneration(hf_config).eval()
    model = Model(mlx_config)

    # This strict load also gates every sanitizer rename and convolution layout.
    model.load_weights(list(_checkpoint_weights(reference).items()), strict=True)
    mx.eval(model.parameters())

    sample_rate = 16_000
    samples = np.arange(3_200, dtype=np.float32)
    audio = (
        0.2 * np.sin(2 * np.pi * 440 * samples / sample_rate)
        + 0.05 * np.cos(2 * np.pi * 97 * samples / sample_rate)
    ).astype(np.float32)

    hf_features = GraniteSpeechFeatureExtractor()(audio)["input_features"]
    mlx_features, num_audio_tokens = model._extract_features(audio)
    mx.eval(mlx_features)
    _assert_close("log-mel frontend", mlx_features, hf_features, atol=2e-4)
    assert num_audio_tokens == 3

    with torch.no_grad():
        hf_encoder = reference.model.encoder(hf_features).last_hidden_state
        hf_projected = reference.model.projector(hf_encoder)
    mlx_encoder = model.encoder(mlx_features)
    mlx_projected = model.projector(mlx_encoder)
    mx.eval(mlx_encoder, mlx_projected)

    # The output is [selected layer 3, final layer], each eight channels wide.
    _assert_close(
        "selected encoder layer 3",
        mlx_encoder[..., :8],
        hf_encoder[..., :8],
        atol=1.5e-3,
    )
    _assert_close(
        "final encoder layer", mlx_encoder[..., 8:], hf_encoder[..., 8:], atol=1.5e-3
    )
    _assert_close("concatenated encoder", mlx_encoder, hf_encoder, atol=1.5e-3)
    _assert_close("Q-Former projector", mlx_projected, hf_projected, atol=3e-4)

    input_ids = np.array([4, 31, 31, 31, 7], dtype=np.int64)
    with torch.no_grad():
        hf_merged = reference.model.get_merged_audio_embeddings(
            torch.from_numpy(input_ids[None]), hf_projected
        )
        hf_logits = reference(inputs_embeds=hf_merged, use_cache=False).logits[:, -1]

    mlx_merged = model._build_inputs_embeds(mx.array(input_ids), mlx_projected)
    mlx_logits = model(mx.array(input_ids[None]), input_embeddings=mlx_merged)[:, -1]
    mx.eval(mlx_merged, mlx_logits)
    _assert_close("multimodal embedding merge", mlx_merged, hf_merged, atol=3e-4)

    # Transformers' speech wrapper exposes the raw tied-head scores, while its
    # base Granite LM and this port apply the positive logits_scaling constant.
    # Compare the raw head values and separately enforce identical greedy choice.
    raw_mlx_logits = mlx_logits * mlx_config.text_config.logits_scaling
    _assert_close("tied output head", raw_mlx_logits, hf_logits, atol=3e-3)
    assert int(mx.argmax(mlx_logits, axis=-1).item()) == int(
        torch.argmax(hf_logits, dim=-1).item()
    )
