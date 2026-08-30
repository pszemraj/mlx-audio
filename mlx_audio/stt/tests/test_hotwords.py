"""Uniform ``hotwords`` support across ASR backends (mlx-vlm #1781, part one).

Each supporting model folds a structured ``hotwords`` list into its own native
prompt field inside its ``generate`` forward loop; models without a hook ignore
it (silent drop). These tests stay weight-free: they exercise the shared merge
helper, inspect each model's ``generate`` signature, and observe the value at
the two prompt boundaries that regressed when the helper was called twice.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_audio.stt.utils import merge_hotwords


class TestMergeHotwords:
    def test_none_is_noop(self):
        assert merge_hotwords("base", None) == "base"
        assert merge_hotwords(None, None) is None

    def test_empty_or_blank_terms_noop(self):
        assert merge_hotwords("base", []) == "base"
        assert merge_hotwords("base", ["", "   ", None]) == "base"

    def test_terms_only(self):
        assert merge_hotwords(None, ["Nativ", "MLX"]) == "Nativ, MLX"

    def test_single_string_is_one_term(self):
        assert merge_hotwords(None, "QFormer") == "QFormer"

    def test_blank_string_is_noop(self):
        assert merge_hotwords("base", "   ") == "base"

    def test_appends_to_existing_base(self):
        assert (
            merge_hotwords("Prior text", ["Nativ", "MLX"]) == "Prior text\nNativ, MLX"
        )

    def test_strips_and_drops_blanks(self):
        assert merge_hotwords(None, [" Nativ ", "", "MLX"]) == "Nativ, MLX"


# model file -> the native prompt field hotwords must fold into
_NATIVE_FIELD = {
    "qwen3_asr/qwen3_asr.py": "system_prompt",
    "whisper/whisper.py": "initial_prompt",
    "vibevoice_asr/vibevoice_asr.py": "context",
    "moss_transcribe_diarize/moss_transcribe_diarize.py": "prompt",
    "granite_speech/granite_speech.py": "prompt",
}

_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


class TestGenerateAcceptsHotwords:
    def _generate_args(self, rel_path):
        tree = ast.parse((_MODELS_DIR / rel_path).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate":
                args = [a.arg for a in node.args.args + node.args.kwonlyargs]
                if "hotwords" in args or "audio" in args:
                    return args
        return []

    def test_each_supporting_model_exposes_hotwords_and_native_field(self):
        for rel_path, native in _NATIVE_FIELD.items():
            args = self._generate_args(rel_path)
            assert "hotwords" in args, f"{rel_path}: generate() missing hotwords"
            assert native in args, f"{rel_path}: generate() missing {native}"


def test_vibevoice_prompt_boundary_receives_each_hotword_once():
    from mlx_audio.stt.models.vibevoice_asr.vibevoice_asr import Model

    captured = {}

    def build_prompt_tokens(speech_features, audio_duration, context):
        del speech_features, audio_duration
        captured["context"] = context
        return mx.zeros((1, 1), dtype=mx.int32), mx.zeros((1, 1), dtype=mx.bool_)

    stub = SimpleNamespace(
        _preprocess_audio=lambda audio, sampling_rate: mx.zeros((1, 24_000)),
        encode_speech=lambda audio, verbose: mx.zeros((1, 1, 1)),
        _build_prompt_tokens=build_prompt_tokens,
        stream_generate=lambda **kwargs: iter(()),
        tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: ""),
        parse_transcription=lambda text: [],
    )

    Model.generate(
        stub,
        mx.zeros((1,)),
        context="Meeting transcript",
        hotwords=["Granite", "MLX"],
    )

    assert captured["context"] == "Meeting transcript\nGranite, MLX"


def test_moss_prompt_boundary_receives_each_hotword_once():
    from mlx_audio.stt.models.moss_transcribe_diarize.moss_transcribe_diarize import (
        Model,
    )

    captured = {}
    sentinel = object()

    def stream_transcribe(audio, **kwargs):
        del audio
        captured["prompt"] = kwargs["prompt"]
        return sentinel

    stub = SimpleNamespace(_stream_transcribe=stream_transcribe)

    result = Model.generate(
        stub,
        mx.zeros((1,)),
        prompt="Meeting transcript",
        hotwords=["Granite", "MLX"],
        stream=True,
    )

    assert result is sentinel
    assert captured["prompt"] == "Meeting transcript\nGranite, MLX"
