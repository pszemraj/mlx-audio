"""Weight-free tests for granite-speech-4.1-2b-plus support.

The plus checkpoint concatenates intermediate Conformer layer outputs onto the
final encoder output (``cat_hidden_layers``), adds a system turn and
``prefix_text`` to the prompt, and emits ``[Speaker N]:`` / ``[T:N]`` tags that
are parsed into the repo's ``segments`` schema. An optional parity test against
``transformers`` runs only where torch is installed.
"""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_audio.stt.generate import _get_cues, generate_transcription, parse_args
from mlx_audio.stt.models.granite_speech.config import (
    EncoderConfig,
    ModelConfig,
    ProjectorConfig,
    TextConfig,
)
from mlx_audio.stt.models.granite_speech.granite_speech import (
    PLUS_SYSTEM_PROMPT,
    TASK_PROMPTS,
    ConformerAttention,
    CTCEncoder,
    EncoderProjector,
    Model,
    StreamingResult,
    _parse_saa,
    _parse_segments,
    _parse_timestamps,
    _resolve_prompt,
)

HIDDEN_DIM = 8


def _tiny_encoder_config(**overrides):
    params = dict(
        input_dim=4,
        num_layers=2,
        hidden_dim=HIDDEN_DIM,
        num_heads=2,
        dim_head=4,
        output_dim=4,
        context_size=8,
        max_pos_emb=16,
    )
    params.update(overrides)
    return EncoderConfig(**params)


class TestEncoderCatHiddenLayers:
    def _run(self, cat_hidden_layers):
        encoder = CTCEncoder(_tiny_encoder_config(cat_hidden_layers=cat_hidden_layers))
        out = encoder(mx.zeros((1, 8, 4)))
        mx.eval(out)
        return out

    def test_none_keeps_hidden_dim(self):
        assert self._run(None).shape == (1, 8, HIDDEN_DIM)

    def test_empty_list_keeps_hidden_dim(self):
        assert self._run([]).shape == (1, 8, HIDDEN_DIM)

    def test_single_layer_doubles_dim(self):
        assert self._run([1]).shape == (1, 8, 2 * HIDDEN_DIM)

    def test_layer_zero_exports_input_linear_output(self):
        assert self._run([0, 1]).shape == (1, 8, 3 * HIDDEN_DIM)

    def test_config_from_dict_keeps_cat_hidden_layers(self):
        cfg = EncoderConfig.from_dict({"cat_hidden_layers": [3], "num_layers": 16})
        assert cfg.cat_hidden_layers == [3]

    def test_projector_consumes_concatenated_features(self):
        config = ModelConfig(
            encoder_config=_tiny_encoder_config(cat_hidden_layers=[1]),
            projector_config=ProjectorConfig(
                hidden_size=HIDDEN_DIM,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=16,
                encoder_hidden_size=2 * HIDDEN_DIM,
            ),
            text_config=TextConfig(hidden_size=12),
        )
        encoder = CTCEncoder(config.encoder_config)
        projector = EncoderProjector(config)
        out = projector(encoder(mx.zeros((1, 8, 4))))
        mx.eval(out)
        # 8 frames -> 1 window of 15 -> window_size // downsample_rate queries
        num_queries = config.window_size // config.downsample_rate
        assert out.shape == (1, num_queries, 12)


def test_non_aligned_attention_preserves_bfloat16():
    config = _tiny_encoder_config(context_size=8)
    attention = ConformerAttention(config)
    attention.set_dtype(mx.bfloat16)

    seq = mx.arange(config.context_size)
    attention_dists = (
        mx.clip(
            seq[:, None] - seq[None, :],
            -config.context_size,
            config.context_size,
        )
        + config.max_pos_emb
    )
    output = attention(
        mx.zeros((1, config.context_size + 1, config.hidden_dim), dtype=mx.bfloat16),
        attention_dists,
    )
    mx.eval(output)

    assert output.dtype == mx.bfloat16


class TestOutputParsers:
    def test_parse_saa_card_example(self):
        text = (
            "[Speaker 1]: Hello how are you "
            "[Speaker 2]: I'm fine and how are you feeling "
            "[Speaker 1]: I feel wonderful"
        )
        segments = _parse_saa(text)
        assert [s["speaker_id"] for s in segments] == [1, 2, 1]
        assert segments[0]["text"] == "Hello how are you"
        assert segments[2]["text"] == "I feel wonderful"
        assert all("start" not in s for s in segments)

    def test_parse_saa_untagged_text_yields_nothing(self):
        assert _parse_saa("plain transcription without tags") == []

    def test_parse_saa_preserves_leading_untagged_text(self):
        assert _parse_saa("intro without attribution [Speaker 1]: tagged turn") == [
            {"speaker_id": None, "text": "intro without attribution"},
            {"speaker_id": 1, "text": "tagged turn"},
        ]

    def test_parse_timestamps_rollover_and_silence(self):
        segments = _parse_timestamps(
            "hello [T:995] world [T:012] _ [T:100] again [T:150]"
        )
        assert len(segments) == 1
        words = segments[0]["words"]
        assert [w["word"] for w in words] == ["hello", "world", "again"]
        assert [w["end"] for w in words] == pytest.approx([9.95, 10.12, 11.50])
        # the dropped silence still advances the next word's start
        assert words[2]["start"] == pytest.approx(11.00)
        assert segments[0]["start"] == 0.0
        assert segments[0]["end"] == pytest.approx(11.50)
        assert segments[0]["text"] == "hello world again"

    def test_parse_timestamps_untagged_text_yields_nothing(self):
        assert _parse_timestamps("no tags here") == []

    def test_timestamp_continuation_inherits_prefix_clock(self):
        segments = _parse_segments(
            "timestamps",
            "continued [T:012]",
            prefix_text="first [T:995] second [T:012] third [T:995]",
        )
        assert segments == [
            {
                "text": "continued",
                "start": pytest.approx(19.95),
                "end": pytest.approx(20.12),
                "words": [
                    {
                        "word": "continued",
                        "start": pytest.approx(19.95),
                        "end": pytest.approx(20.12),
                    }
                ],
            }
        ]

    def test_timestamp_segments_yield_only_word_level_cues(self):
        output = SimpleNamespace(
            segments=_parse_timestamps("hello [T:50] world [T:100]")
        )
        assert _get_cues(output) == [
            {"start": 0.0, "end": 0.5, "text": "hello"},
            {"start": 0.5, "end": 1.0, "text": "world"},
        ]

    def test_parse_segments_dispatch(self):
        assert _parse_segments("asr", "[Speaker 1]: hi") == []
        assert _parse_segments("saa", "[Speaker 1]: hi") == [
            {"speaker_id": 1, "text": "hi"}
        ]
        assert _parse_segments("timestamps", "hi [T:50]")[0]["end"] == 0.5

    def test_untagged_saa_continuation_attributed_to_prefix_speaker(self):
        # Mid-turn continuations carry no tag; they belong to the prefix's
        # last speaker.
        segments = _parse_segments(
            "saa",
            "and then the meeting ended",
            prefix_text="[Speaker 1]: hello [Speaker 2]: hi there",
        )
        assert segments == [{"speaker_id": 2, "text": "and then the meeting ended"}]

    def test_tagged_saa_continuation_ignores_prefix(self):
        segments = _parse_segments(
            "saa", "[Speaker 3]: new voice", prefix_text="[Speaker 1]: hello"
        )
        assert segments == [{"speaker_id": 3, "text": "new voice"}]

    def test_saa_continuation_without_prefix_tags_unchanged(self):
        assert _parse_segments("saa", "plain text", prefix_text="no tags here") == []

    @pytest.mark.parametrize("output_format", ["srt", "vtt"])
    def test_untimed_saa_subtitles_fall_back_to_text(
        self, output_format, tmp_path, capsys
    ):
        def generate(audio, verbose=False, generation_stream=None):
            return SimpleNamespace(
                text="complete transcript",
                segments=[{"speaker_id": 1, "text": "complete transcript"}],
            )

        output_path = tmp_path / "transcript"
        generate_transcription(
            model=SimpleNamespace(generate=generate),
            audio=mx.zeros((1,)),
            output_path=str(output_path),
            format=output_format,
        )

        assert "No timed cues found" in capsys.readouterr().out
        assert output_path.with_suffix(".txt").read_text() == "complete transcript"
        assert not output_path.with_suffix(f".{output_format}").exists()


# Enough of the plus chat template to exercise the native prefix_text hook and
# the system-turn handling; the 4.0/4.1 template lacks the hook entirely.
PLUS_TEMPLATE_TAIL = (
    "{%- if add_generation_prompt %}{{- '<|start_of_role|>assistant<|end_of_role|>' }}"
    "{%- if prefix_text is defined and prefix_text %}{{- prefix_text }}{%- endif %}"
    "{%- endif %}"
)
LEGACY_TEMPLATE = "{{ messages }}<|start_of_role|>assistant<|end_of_role|>"


class StubTokenizer:
    """Records what _build_prompt renders; mimics the template contract only."""

    def __init__(self, chat_template=None):
        if chat_template is not None:
            self.chat_template = chat_template
        self.last_prompt = None

    def apply_chat_template(
        self, chat, tokenize=False, add_generation_prompt=True, **kwargs
    ):
        parts = [
            f"<|start_of_role|>{m['role']}<|end_of_role|>{m['content']}<|end_of_text|>\n"
            for m in chat
        ]
        if add_generation_prompt:
            parts.append("<|start_of_role|>assistant<|end_of_role|>")
            if "prefix_text" in self.chat_template and kwargs.get("prefix_text"):
                parts.append(kwargs["prefix_text"])
        return "".join(parts)

    def encode(self, text):
        self.last_prompt = text
        return [0]


def _build_prompt(
    tokenizer, num_audio_tokens=2, prompt="do the thing", *, is_plus=True, **kwargs
):
    stub_model = SimpleNamespace(_tokenizer=tokenizer, is_plus=is_plus)
    Model._build_prompt(stub_model, num_audio_tokens, prompt, **kwargs)
    return tokenizer.last_prompt


class TestBuildPrompt:
    def test_plus_placeholder_has_reference_space(self):
        tok = StubTokenizer(PLUS_TEMPLATE_TAIL)
        rendered = _build_prompt(tok)
        assert "<|audio|><|audio|> do the thing<|end_of_text|>" in rendered

    def test_non_plus_placeholder_has_no_space(self):
        tok = StubTokenizer(LEGACY_TEMPLATE)
        rendered = _build_prompt(tok, is_plus=False)
        assert "<|audio|><|audio|>do the thing<|end_of_text|>" in rendered

    def test_system_turn_inserted_first(self):
        tok = StubTokenizer(PLUS_TEMPLATE_TAIL)
        rendered = _build_prompt(tok, system_prompt=PLUS_SYSTEM_PROMPT)
        assert rendered.startswith(
            f"<|start_of_role|>system<|end_of_role|>{PLUS_SYSTEM_PROMPT}"
        )

    def test_native_prefix_text_not_double_appended(self):
        tok = StubTokenizer(PLUS_TEMPLATE_TAIL)
        rendered = _build_prompt(tok, prefix_text="[Speaker 1]: so far")
        assert rendered.endswith("<|end_of_role|>[Speaker 1]: so far")
        assert rendered.count("[Speaker 1]: so far") == 1

    def test_manual_prefix_append_for_legacy_template(self):
        tok = StubTokenizer(LEGACY_TEMPLATE)
        rendered = _build_prompt(tok, prefix_text="prior text")
        assert rendered.endswith("<|start_of_role|>assistant<|end_of_role|>prior text")

    def test_no_template_fallback(self):
        tok = StubTokenizer()
        rendered = _build_prompt(tok, prefix_text="prior")
        assert rendered == "USER: <|audio|><|audio|> do the thing\nASSISTANT:prior"

    def test_default_prompt_is_asr_task(self):
        tok = StubTokenizer(PLUS_TEMPLATE_TAIL)
        stub_model = SimpleNamespace(_tokenizer=tok, is_plus=True)
        Model._build_prompt(stub_model, 1, None)
        assert TASK_PROMPTS["asr"] in tok.last_prompt


def test_generate_rejects_unknown_task():
    with pytest.raises(ValueError, match="Unknown task"):
        Model.generate(SimpleNamespace(), None, task="diarize")


class TestResolvePrompt:
    def test_explicit_prompt_wins(self):
        assert _resolve_prompt("saa", "custom", "fr") == "custom"

    def test_rich_task_beats_translation_language(self):
        assert _resolve_prompt("saa", None, "en") == TASK_PROMPTS["saa"]
        assert _resolve_prompt("timestamps", None, "en") == TASK_PROMPTS["timestamps"]

    def test_asr_with_language_translates(self):
        assert _resolve_prompt("asr", None, "fr") == "Translate the speech to French."

    def test_asr_default(self):
        assert _resolve_prompt("asr", None, None) == TASK_PROMPTS["asr"]

    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            _resolve_prompt("diarize", None, None)


def test_cli_omits_language_to_preserve_model_default(tmp_path):
    args = parse_args(["--audio", "audio.wav", "--output-path", "transcript"])
    captured = {}

    def generate(
        audio, language="model-default", verbose=False, generation_stream=None
    ):
        captured["language"] = language
        return SimpleNamespace(text="transcript", segments=[])

    generate_transcription(
        model=SimpleNamespace(generate=generate),
        audio=mx.zeros((1,)),
        output_path=str(tmp_path / "transcript"),
        language=args.language,
    )

    assert args.language is None
    assert captured["language"] == "model-default"


class TestIsPlus:
    def _is_plus(self, config):
        return Model.is_plus.fget(SimpleNamespace(config=config))

    def test_by_model_type(self):
        assert self._is_plus(ModelConfig(model_type="granite_speech_plus"))

    def test_by_cat_hidden_layers_after_conversion(self):
        # mlx_audio.convert rewrites model_type to "granite_speech"; the
        # architectural fingerprint must still identify the plus variant.
        config = ModelConfig(
            model_type="granite_speech",
            encoder_config={"cat_hidden_layers": [3]},
        )
        assert self._is_plus(config)

    def test_non_plus(self):
        assert not self._is_plus(ModelConfig())


def test_plus_model_type_alias_does_not_require_repo_name_hint():
    from mlx_audio.stt.utils import MODEL_REMAPPING
    from mlx_audio.utils import get_model_class

    arch, model_type = get_model_class(
        model_type="granite_speech_plus",
        model_name=["1454e6e1e33845ca9280ff65f52cf1141ba6e6e2"],
        category="stt",
        model_remapping=MODEL_REMAPPING,
    )

    assert model_type == "granite_speech"
    assert arch.Model is Model


class TestSanitizeWeights:
    @pytest.mark.parametrize(
        ("name", "pytorch_shape", "mlx_shape"),
        [
            ("up_conv", (16, 8, 1), (16, 1, 8)),
            ("down_conv", (8, 16, 1), (8, 1, 16)),
            ("depth_conv", (16, 1, 5), (16, 5, 1)),
        ],
    )
    def test_convolution_conversion_is_idempotent(self, name, pytorch_shape, mlx_shape):
        key = f"encoder.layers.0.conv.{name}.weight"
        source = {key: mx.zeros(pytorch_shape)}

        converted = Model.sanitize(source)
        reloaded = Model.sanitize(converted)

        assert converted[key].shape == mlx_shape
        assert reloaded[key].shape == mlx_shape


@pytest.mark.parametrize("encoder_dtype", [mx.float32, mx.bfloat16])
def test_audio_features_match_loaded_encoder_dtype(encoder_dtype):
    class RecordingEncoder:
        def __init__(self):
            self.input_linear = SimpleNamespace(
                weight=mx.zeros((1,), dtype=encoder_dtype)
            )
            self.input_dtype = None

        def __call__(self, features):
            self.input_dtype = features.dtype
            return features

    encoder = RecordingEncoder()
    stub_model = SimpleNamespace(encoder=encoder, projector=lambda features: features)

    output = Model.get_audio_features(stub_model, mx.zeros((1, 2, 4), dtype=mx.float32))

    assert encoder.input_dtype == encoder_dtype
    assert output.dtype == encoder_dtype


def test_in_memory_audio_is_resampled_to_model_rate():
    audio = np.zeros(48000, dtype=np.float32)

    loaded = Model._load_audio(SimpleNamespace(), audio, sample_rate=48000)
    mx.eval(loaded)

    assert loaded.dtype == mx.float32
    assert loaded.shape == (16000,)


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros(16000, dtype=np.int16),
        mx.zeros((16000,), dtype=mx.int16),
    ],
)
def test_in_memory_audio_rejects_integer_pcm(audio):
    with pytest.raises(ValueError, match="normalized floating-point"):
        Model._load_audio(SimpleNamespace(), audio)


@pytest.mark.parametrize("shape", [(16000, 2), (2, 16000)])
def test_in_memory_audio_rejects_multichannel_arrays(shape):
    with pytest.raises(ValueError, match="mono 1-D waveform"):
        Model._load_audio(SimpleNamespace(), np.zeros(shape, dtype=np.float32))


def test_load_audio_rejects_invalid_sample_rate():
    with pytest.raises(ValueError, match="sample_rate must be a positive integer"):
        Model._load_audio(SimpleNamespace(), np.zeros(1), sample_rate=0)


def test_stream_generation_forwards_sample_rate():
    forwarded = {}

    def stream_generate(audio, **kwargs):
        forwarded.update(kwargs)
        return iter(())

    stub_model = SimpleNamespace(
        is_plus=False,
        config=SimpleNamespace(model_type="granite_speech"),
        _stream_generate=stream_generate,
    )

    result = Model.generate(stub_model, mx.zeros(48000), stream=True, sample_rate=48000)

    assert list(result) == []
    assert forwarded["sample_rate"] == 48000
    assert forwarded["task"] == "asr"


def test_word_timestamps_alias_selects_rich_timestamp_task():
    forwarded = {}

    def stream_generate(audio, **kwargs):
        forwarded.update(kwargs)
        return iter(())

    stub_model = SimpleNamespace(
        is_plus=True,
        config=SimpleNamespace(model_type="granite_speech_plus"),
        _stream_generate=stream_generate,
    )

    result = Model.generate(
        stub_model,
        mx.zeros((16000,)),
        stream=True,
        word_timestamps=True,
    )

    assert list(result) == []
    assert forwarded["task"] == "timestamps"
    assert forwarded["prompt"] == TASK_PROMPTS["timestamps"]


def test_word_timestamps_alias_rejects_saa_conflict():
    with pytest.raises(ValueError, match="conflicts"):
        Model.generate(
            SimpleNamespace(),
            mx.zeros((16000,)),
            task="saa",
            word_timestamps=True,
        )


@pytest.mark.parametrize(
    ("task", "pieces", "expected_segments"),
    [
        (
            "saa",
            {10: "[Speaker 1]: ", 11: "hello"},
            [{"speaker_id": 1, "text": "hello"}],
        ),
        (
            "timestamps",
            {10: "hello ", 11: "[T:50]"},
            [
                {
                    "text": "hello",
                    "start": 0.0,
                    "end": 0.5,
                    "words": [{"word": "hello", "start": 0.0, "end": 0.5}],
                }
            ],
        ),
    ],
)
def test_stream_final_result_carries_parsed_segments(
    monkeypatch, task, pieces, expected_segments
):
    import mlx_audio.lm.generate as lm_generate

    class SequenceTokenizer:
        eos_token_id = 99
        clean_up_tokenization_spaces = False

        def decode(self, token_ids, **kwargs):
            del kwargs
            return "".join(pieces[token_id] for token_id in token_ids)

    def fake_generate_step(**kwargs):
        del kwargs
        yield 10, None
        yield 11, None
        yield 99, None

    monkeypatch.setattr(lm_generate, "generate_step", fake_generate_step)
    stub_model = SimpleNamespace(
        _tokenizer=SequenceTokenizer(),
        _load_audio=lambda audio, sample_rate: audio,
        _extract_features=lambda audio: (mx.zeros((1, 1, 160)), 1),
        get_audio_features=lambda features: mx.zeros((1, 1, 4)),
        _build_prompt=lambda *args, **kwargs: mx.array([0]),
        _build_inputs_embeds=lambda *args, **kwargs: mx.zeros((1, 1, 4)),
    )

    results = list(Model._stream_generate(stub_model, mx.zeros((1,)), task=task))

    assert "".join(result.text for result in results) == "".join(pieces.values())
    assert all(result.start_time is None for result in results)
    assert all(result.end_time is None for result in results)
    assert results[-1].is_final
    assert results[-1].segments == expected_segments


def test_streamed_timestamp_cli_uses_final_structured_segments(tmp_path):
    parsed = _parse_timestamps("hello [T:50]")

    def generate(audio, *, stream=False, verbose=False):
        del audio, stream, verbose
        yield StreamingResult("hello ", False, None, None)
        yield StreamingResult("[T:50]", False, None, None)
        yield StreamingResult("", True, None, None, segments=parsed)

    output_path = tmp_path / "timestamped"
    generate_transcription(
        model=SimpleNamespace(generate=generate),
        audio=mx.zeros((1,)),
        output_path=str(output_path),
        format="srt",
        stream=True,
    )

    subtitle = output_path.with_suffix(".srt").read_text()
    assert subtitle.count("-->") == 1
    assert "00:00:00,000 --> 00:00:00,500" in subtitle
    assert "hello" in subtitle
    assert "[T:50]" not in subtitle


def test_untimed_stream_does_not_create_zero_duration_cues(tmp_path, capsys):
    def generate(audio, *, stream=False, verbose=False):
        del audio, stream, verbose
        yield StreamingResult("hello", False, 0.0, 0.0)
        yield StreamingResult("", True, 0.0, 0.0)

    output_path = tmp_path / "untimed"
    generate_transcription(
        model=SimpleNamespace(generate=generate),
        audio=mx.zeros((1,)),
        output_path=str(output_path),
        format="srt",
        stream=True,
    )

    assert "No timed cues found" in capsys.readouterr().out
    assert output_path.with_suffix(".txt").read_text() == "hello"
    assert not output_path.with_suffix(".srt").exists()


def test_encoder_parity_with_transformers_plus():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from transformers.models.granite_speech_plus import (
            modeling_granite_speech_plus as hf_mod,
        )
    except ImportError:
        pytest.skip("transformers build lacks granite_speech_plus")

    hf_config_cls = getattr(hf_mod, "GraniteSpeechPlusEncoderConfig", None)
    if hf_config_cls is None:
        from transformers.models.granite_speech_plus import (
            configuration_granite_speech_plus as hf_cfg_mod,
        )

        hf_config_cls = hf_cfg_mod.GraniteSpeechPlusEncoderConfig

    params = dict(
        input_dim=4,
        # Match the real checkpoint's exported layer 3 of 10. In Transformers,
        # exporting the midpoint aliases an in-place update and is not a stable
        # reference behavior for the pre-injection hidden state.
        num_layers=10,
        hidden_dim=HIDDEN_DIM,
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
    torch.manual_seed(0)
    hf_encoder = hf_mod.GraniteSpeechPlusCTCEncoder(hf_config_cls(**params)).eval()
    mlx_encoder = CTCEncoder(EncoderConfig(**params))

    weights = {
        f"encoder.{k}": mx.array(v.detach().numpy())
        for k, v in hf_encoder.state_dict().items()
    }
    weights = Model.sanitize(weights)
    mlx_encoder.load_weights(
        [(k.removeprefix("encoder."), v) for k, v in weights.items()], strict=True
    )

    x = np.random.default_rng(0).standard_normal((1, 8, 4)).astype(np.float32)
    with torch.no_grad():
        hf_out = hf_encoder(torch.from_numpy(x))
    if hasattr(hf_out, "last_hidden_state"):
        hf_out = hf_out.last_hidden_state
    mlx_out = mlx_encoder(mx.array(x))
    mx.eval(mlx_out)

    diff = np.max(np.abs(np.array(mlx_out) - hf_out.numpy()))
    assert diff < 1e-3, f"encoder outputs diverge: max abs diff {diff}"
