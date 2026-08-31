"""Manual pinned-checkpoint parity for Granite Speech Plus.

This test is intentionally excluded from the pull-request job because it
materializes the 2B reference checkpoint. Run it through the dedicated manual
workflow or set ``MLX_AUDIO_RUN_GRANITE_CHECKPOINT_PARITY=1`` locally.
"""

import gc
import importlib.metadata
import json
import os
from pathlib import Path

import accelerate
import mlx.core as mx
import numpy as np
import pytest
import tokenizers
import torch
import torchaudio
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from mlx_audio.audio_io import read as read_audio
from mlx_audio.stt.models.granite_speech.granite_speech import (
    PLUS_SYSTEM_PROMPT,
    TASK_PROMPTS,
)
from mlx_audio.stt.utils import load_model

MODEL_ID = "ibm-granite/granite-speech-4.1-2b-plus"
MODEL_REVISION = "1454e6e1e33845ca9280ff65f52cf1141ba6e6e2"
TOKENIZER_REVISION = MODEL_REVISION
REFERENCE_TRANSFORMERS = "5.8.1"
REFERENCE_TOKENIZERS = "0.23.0-rc0"
REFERENCE_ACCELERATE = "1.14.0"
FIXTURE = Path(__file__).parents[2] / "examples/voice_prompts/en_man.wav"
RICH_FIXTURE_SECONDS = 4
REFERENCE_SAMPLES = RICH_FIXTURE_SECONDS * 16_000

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX_AUDIO_RUN_GRANITE_CHECKPOINT_PARITY") != "1",
    reason="set MLX_AUDIO_RUN_GRANITE_CHECKPOINT_PARITY=1 for the 2B smoke test",
)


def _prompt_text(tokenizer, task):
    chat = [
        {"role": "system", "content": PLUS_SYSTEM_PROMPT},
        {"role": "user", "content": f"<|audio|> {TASK_PROMPTS[task]}"},
    ]
    return tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )


def _log_provenance():
    record = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "accelerate": accelerate.__version__,
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "mlx": importlib.metadata.version("mlx"),
        "fixture": str(FIXTURE.relative_to(Path(__file__).parents[2])),
        "reference_samples": REFERENCE_SAMPLES,
        "rich_fixture_seconds": RICH_FIXTURE_SECONDS,
        "conversion": {
            "source": "pinned native safetensors",
            "dtype": "bfloat16",
            "quantize": False,
            "strict_load": True,
            "reference_device_map": "mps",
        },
    }
    print("Granite Speech Plus checkpoint parity provenance:")
    print(json.dumps(record, indent=2, sort_keys=True))


def test_pinned_checkpoint_reference_and_mlx_paths():
    if transformers.__version__ != REFERENCE_TRANSFORMERS:
        raise RuntimeError(
            "The multi-window checkpoint reference requires "
            f"transformers=={REFERENCE_TRANSFORMERS}, got "
            f"{transformers.__version__}. Use the isolated reference target "
            "documented in tests/model_parity/README.md."
        )
    if tokenizers.__version__ != REFERENCE_TOKENIZERS:
        raise RuntimeError(
            "The pinned reference requires "
            f"tokenizers=={REFERENCE_TOKENIZERS}, got {tokenizers.__version__}."
        )
    if accelerate.__version__ != REFERENCE_ACCELERATE:
        raise RuntimeError(
            "The direct-to-MPS reference load requires "
            f"accelerate=={REFERENCE_ACCELERATE}, got {accelerate.__version__}."
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("The pinned reference smoke requires a real MPS device.")
    if not mx.metal.is_available():
        raise RuntimeError("The pinned MLX smoke requires a real Metal device.")

    _log_provenance()
    audio, sample_rate = read_audio(
        FIXTURE, dtype="float32", sample_rate=16_000, nchannels=1
    )
    assert sample_rate == 16_000
    reference_audio = np.asarray(audio[:REFERENCE_SAMPLES], dtype=np.float32)
    rich_audio = np.asarray(
        audio[: RICH_FIXTURE_SECONDS * sample_rate], dtype=np.float32
    )

    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=TOKENIZER_REVISION)
    reference_inputs = {}
    for task in TASK_PROMPTS:
        reference_inputs[task] = processor(
            _prompt_text(processor.tokenizer, task),
            reference_audio,
            return_tensors="pt",
        )

    device = torch.device("mps")
    reference = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        # A single-device map materializes checkpoint tensors directly on MPS.
        # Avoiding a simultaneous full CPU model copy keeps this manual gate
        # viable on memory-constrained GitHub-hosted Apple Silicon runners.
        device_map="mps",
    )
    reference.eval()
    asr_inputs = reference_inputs["asr"].to(device)
    with torch.inference_mode():
        reference_logits = (
            reference(**asr_inputs, use_cache=False).logits[:, -1].float().cpu().numpy()
        )
    reference_token = int(np.argmax(reference_logits, axis=-1)[0])

    del asr_inputs, reference
    gc.collect()
    torch.mps.empty_cache()

    model = load_model(MODEL_ID, revision=MODEL_REVISION, strict=True)
    input_features, num_audio_tokens = model._extract_features(reference_audio)
    audio_features = model.get_audio_features(input_features)
    mx.eval(audio_features)

    mlx_prompt_ids = {}
    for task in TASK_PROMPTS:
        mlx_prompt_ids[task] = model._build_prompt(
            num_audio_tokens,
            TASK_PROMPTS[task],
            system_prompt=PLUS_SYSTEM_PROMPT,
        )
        expected_ids = reference_inputs[task]["input_ids"][0].cpu().numpy()
        np.testing.assert_array_equal(np.asarray(mlx_prompt_ids[task]), expected_ids)

    inputs_embeds = model._build_inputs_embeds(mlx_prompt_ids["asr"], audio_features)
    mx.eval(inputs_embeds)
    mlx_logits = model(mlx_prompt_ids["asr"][None], input_embeddings=inputs_embeds)[
        :, -1
    ]
    mx.eval(mlx_logits)
    mlx_logits_np = np.asarray(mlx_logits.astype(mx.float32))
    absolute_error = np.abs(mlx_logits_np - reference_logits)
    cosine = float(
        np.sum(mlx_logits_np * reference_logits)
        / (np.linalg.norm(mlx_logits_np) * np.linalg.norm(reference_logits))
    )
    print(
        "First-token logits: "
        f"max_abs={absolute_error.max():.6f}, "
        f"mean_abs={absolute_error.mean():.6f}, "
        f"p99_abs={np.percentile(absolute_error, 99):.6f}, "
        f"cosine={cosine:.8f}"
    )
    # BF16 kernels quantize logits in different steps across MPS and Metal, so
    # gate aggregate and worst-case error separately.
    assert float(absolute_error.max()) <= 0.25
    assert float(absolute_error.mean()) <= 0.05
    assert float(np.percentile(absolute_error, 99)) <= 0.1
    assert cosine >= 0.9999
    assert int(np.argmax(mlx_logits_np, axis=-1)[0]) == reference_token

    saa = model.generate(
        rich_audio,
        task="saa",
        system_prompt=PLUS_SYSTEM_PROMPT,
        max_tokens=512,
    )
    assert "[Speaker " in saa.text
    assert saa.segments

    timestamps = model.generate(
        rich_audio,
        task="timestamps",
        system_prompt=PLUS_SYSTEM_PROMPT,
        max_tokens=1_024,
    )
    assert "[T:" in timestamps.text
    assert timestamps.segments
    assert timestamps.segments[0]["words"]
