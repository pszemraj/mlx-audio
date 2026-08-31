import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_audio.lm.models.base import create_attention_mask
from mlx_audio.lm.models.cache import KVCache
from mlx_audio.lm.models.granite import Model as GraniteLM
from mlx_audio.lm.models.granite import ModelArgs as GraniteModelArgs
from mlx_audio.stt.models.base import STTOutput

from .config import EncoderConfig, ModelConfig, ProjectorConfig

LANGUAGE_CODES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "pt": "Portuguese",
    "ja": "Japanese",
}
SAMPLE_RATE = 16000

# Verbatim from the granite-speech-4.1-2b-plus model card. IBM's reference code
# sends this system turn; without one the plus chat template substitutes a
# generic assistant message the model was not trained against.
PLUS_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\nToday's Date: December 19, 2024.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)

# Task prompts verbatim from the model card. An unfamiliar or malformed prompt
# makes the model silently fall back to plain transcription, so these must not
# be reworded.
TASK_PROMPTS = {
    "asr": "can you transcribe the speech into a written format?",
    "saa": (
        "Speaker attribution: Transcribe and denote who is speaking by adding "
        "[Speaker 1]: and [Speaker 2]: tags before speaker turns."
    ),
    "timestamps": (
        "Timestamps: Transcribe the speech. After each word, add a timestamp tag "
        "showing the end time in centiseconds, e.g. hello [T:45] world [T:82]"
    ),
}

_SPEAKER_RE = re.compile(r"\[Speaker (\d+)\]:")
_TS_RE = re.compile(r"\[T:(\d+)\]")


class UnsupportedTranscriptionTask(ValueError):
    """The loaded checkpoint cannot perform the requested transcription task."""


class StructuredTranscriptError(RuntimeError):
    """A rich transcription did not contain the requested structured syntax."""

    def __init__(self, message: str, *, raw_text: str) -> None:
        super().__init__(message)
        self.raw_text = raw_text


class IncompleteTranscription(RuntimeError):
    """Generation reached its token budget before a transcript was complete."""

    def __init__(self, *, task: str, max_tokens: int, partial_text: str) -> None:
        super().__init__(
            f"Granite {task} generation reached max_tokens={max_tokens} before "
            "EOS. The structured transcript is incomplete."
        )
        self.task = task
        self.max_tokens = max_tokens
        self.partial_text = partial_text


def _normalize_task(task: str, *, word_timestamps: bool = False) -> str:
    normalized = (task or "asr").lower().strip()
    if normalized not in TASK_PROMPTS:
        raise ValueError(
            f"Unknown task {task!r}; expected one of {sorted(TASK_PROMPTS)}"
        )
    if word_timestamps:
        if normalized not in ("asr", "timestamps"):
            raise ValueError(
                "word_timestamps=True conflicts with " f"task={normalized!r}"
            )
        normalized = "timestamps"
    return normalized


def _reject_reserved_audio_token(**text_fields: object) -> None:
    """Reject user-controlled text that could create extra audio placeholders."""
    for name, value in text_fields.items():
        values = value if isinstance(value, (list, tuple)) else (value,)
        if any(isinstance(item, str) and "<|audio|>" in item for item in values):
            raise ValueError(f"{name} must not contain the reserved <|audio|> token.")


def _speaker_state_from_prefix(
    prefix_text: Optional[str],
) -> Tuple[List[int], Optional[int]]:
    """Validate an SAA prefix and return seen speakers plus the active speaker."""
    if not prefix_text:
        return [], None

    matches = list(_SPEAKER_RE.finditer(prefix_text))
    if not matches:
        raise StructuredTranscriptError(
            "SAA prefix_text contains no [Speaker N]: tags.",
            raw_text=prefix_text,
        )
    if prefix_text[: matches[0].start()].strip():
        raise StructuredTranscriptError(
            "SAA prefix_text begins with unattributed text.",
            raw_text=prefix_text,
        )

    seen: List[int] = []
    for index, match in enumerate(matches):
        speaker_id = int(match.group(1))
        body_end = (
            matches[index + 1].start() if index + 1 < len(matches) else len(prefix_text)
        )
        if not prefix_text[match.end() : body_end].strip():
            raise StructuredTranscriptError(
                f"[Speaker {speaker_id}]: has no associated transcript text.",
                raw_text=prefix_text,
            )
        if speaker_id not in seen:
            expected = len(seen) + 1
            if speaker_id != expected:
                raise StructuredTranscriptError(
                    "SAA speakers must be introduced in order: "
                    f"expected Speaker {expected}, got Speaker {speaker_id}.",
                    raw_text=prefix_text,
                )
            seen.append(speaker_id)

    return seen, int(matches[-1].group(1))


def _parse_saa(text: str, prefix_text: Optional[str] = None) -> List[dict]:
    """Parse a complete SAA continuation without inventing speaker metadata."""
    seen, active_prefix_speaker = _speaker_state_from_prefix(prefix_text)
    matches = list(_SPEAKER_RE.finditer(text))
    segments: List[dict] = []
    leading = text[: matches[0].start()].strip() if matches else text.strip()

    if leading:
        if active_prefix_speaker is None:
            raise StructuredTranscriptError(
                "Granite SAA output begins with unattributed text before the "
                "first [Speaker N]: tag.",
                raw_text=text,
            )
        segments.append(
            {
                "speaker_id": active_prefix_speaker,
                "speaker_id_source": "inferred_from_prefix",
                "text": leading,
            }
        )

    if not matches:
        if segments:
            return segments
        raise StructuredTranscriptError(
            "Granite speaker-attribution mode produced no [Speaker N]: tags "
            "and no attributable continuation.",
            raw_text=text,
        )

    for index, match in enumerate(matches):
        speaker_id = int(match.group(1))
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : body_end].strip()
        if not body:
            raise StructuredTranscriptError(
                f"[Speaker {speaker_id}]: has no associated transcript text.",
                raw_text=text,
            )
        if speaker_id not in seen:
            expected = len(seen) + 1
            if speaker_id != expected:
                raise StructuredTranscriptError(
                    "SAA speakers must be introduced in order: "
                    f"expected Speaker {expected}, got Speaker {speaker_id}.",
                    raw_text=text,
                )
            seen.append(speaker_id)
        segments.append({"speaker_id": speaker_id, "text": body})

    return segments


def _resolve_timestamp_centiseconds(value: str, previous_cs: Optional[int]) -> int:
    """Resolve a modulo-1000 timestamp while enforcing monotonic output."""
    current_cs = int(value)
    if current_cs >= 1000:
        if previous_cs is not None and current_cs < previous_cs:
            raise ValueError(
                "Absolute Granite timestamp moved backwards: "
                f"{current_cs} < {previous_cs} centiseconds."
            )
        return current_cs
    if previous_cs is None:
        return current_cs

    current_cs += (previous_cs // 1000) * 1000
    while current_cs < previous_cs:
        current_cs += 1000
    return current_cs


def _timestamp_items(text: str, *, source: str) -> List[Tuple[str, str]]:
    """Return validated ``(word, timestamp)`` pairs for one transcript string."""
    tags = _TS_RE.findall(text)
    if not tags:
        message = (
            "Timestamp prefix_text contains no [T:N] tags."
            if source == "prefix_text"
            else "Granite timestamp mode produced no [T:N] tags. The model may "
            "have fallen back to plain ASR."
        )
        raise StructuredTranscriptError(message, raw_text=text)

    parts = _TS_RE.split(text)
    trailing = parts[-1].strip()
    if trailing:
        message = (
            "Timestamp prefix_text ends with content lacking a [T:N] tag: "
            if source == "prefix_text"
            else "Granite timestamp output ends with content lacking a [T:N] tag: "
        )
        raise StructuredTranscriptError(f"{message}{trailing!r}.", raw_text=text)

    items: List[Tuple[str, str]] = []
    for raw_token, raw_timestamp in zip(parts[0::2], tags):
        token = raw_token.strip()
        if not token:
            raise StructuredTranscriptError(
                f"[T:{raw_timestamp}] has no preceding word or '_' marker.",
                raw_text=text,
            )
        if token != "_" and len(token.split()) != 1:
            raise StructuredTranscriptError(
                "Expected exactly one word or '_' before each timestamp tag, "
                f"but {token!r} precedes [T:{raw_timestamp}]. One or more "
                "timestamp tags are missing.",
                raw_text=text,
            )
        items.append((token, raw_timestamp))
    return items


def _timestamp_cursor_from_prefix(
    prefix_text: Optional[str],
) -> Tuple[float, Optional[int]]:
    """Validate a timestamp prefix and recover its absolute time cursor."""
    if not prefix_text:
        return 0.0, None

    previous_cs: Optional[int] = None
    for _, raw_timestamp in _timestamp_items(prefix_text, source="prefix_text"):
        try:
            previous_cs = _resolve_timestamp_centiseconds(raw_timestamp, previous_cs)
        except ValueError as exc:
            raise StructuredTranscriptError(str(exc), raw_text=prefix_text) from exc
    return previous_cs / 100.0, previous_cs


def _parse_timestamps(text: str, prefix_text: Optional[str] = None) -> List[dict]:
    """Parse a complete word-timestamp sequence without fabricating alignment."""
    cursor, previous_cs = _timestamp_cursor_from_prefix(prefix_text)
    words = []

    for token, raw_timestamp in _timestamp_items(text, source="output"):
        try:
            current_cs = _resolve_timestamp_centiseconds(raw_timestamp, previous_cs)
        except ValueError as exc:
            raise StructuredTranscriptError(str(exc), raw_text=text) from exc
        end = current_cs / 100.0
        if token != "_":
            words.append({"word": token, "start": cursor, "end": end})
        cursor = end
        previous_cs = current_cs

    if not words:
        return []
    return [
        {
            "text": " ".join(word["word"] for word in words),
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "words": words,
        }
    ]


def _resolve_prompt(task: str, prompt: Optional[str], language: Optional[str]) -> str:
    # Rich tasks are prompt-controlled, so their canonical prompts are part of
    # the output-schema contract. Custom prompts remain available for ASR.
    task = _normalize_task(task)
    if prompt is not None:
        if task != "asr":
            raise ValueError(
                f"prompt cannot override task={task!r}. Use hotwords=[...] for "
                "contextual biasing, or use task='asr' for an unconstrained "
                "custom instruction."
            )
        return prompt
    if task != "asr":
        return TASK_PROMPTS[task]
    if language is not None:
        lang_name = LANGUAGE_CODES.get(language.lower(), language)
        return f"Translate the speech to {lang_name}."
    return TASK_PROMPTS["asr"]


def _validate_structured_output(
    task: str, text: str, prefix_text: Optional[str] = None
) -> None:
    """Validate rich output through the same strict parser used for results."""
    _parse_segments(task, text, prefix_text)


def _parse_segments(
    task: str, text: str, prefix_text: Optional[str] = None
) -> List[dict]:
    if task == "saa":
        return _parse_saa(text, prefix_text)
    if task == "timestamps":
        return _parse_timestamps(text, prefix_text)
    return []


@dataclass
class StreamingResult:
    text: str
    is_final: bool
    start_time: Optional[float]
    end_time: Optional[float]
    language: Optional[str] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    segments: Optional[List[dict]] = None
    finish_reason: Optional[str] = None
    complete: Optional[bool] = None
    raw_text: Optional[str] = None
    error_type: Optional[str] = None
    error: Optional[str] = None


class BatchNorm1d(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias = mx.zeros((num_features,))
        self.running_mean = mx.zeros((num_features,))
        self.running_var = mx.ones((num_features,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return (x - self.running_mean) / mx.sqrt(
            self.running_var + self.eps
        ) * self.weight + self.bias


class ConformerFeedForward(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.up_proj = nn.Linear(
            config.hidden_dim, config.hidden_dim * config.feedforward_mult
        )
        self.down_proj = nn.Linear(
            config.hidden_dim * config.feedforward_mult, config.hidden_dim
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pre_norm(x)
        x = nn.silu(self.up_proj(x))
        x = self.down_proj(x)
        return x


class ConformerAttention(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.dim_head * config.num_heads
        self.max_pos_emb = config.max_pos_emb
        self.context_size = config.context_size
        self.num_heads = config.num_heads
        self.dim_head = config.dim_head
        self.scale = config.dim_head**-0.5
        self.pre_norm = nn.LayerNorm(config.hidden_dim)
        self.to_q = nn.Linear(config.hidden_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(config.hidden_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, config.hidden_dim)
        self.rel_pos_emb = nn.Embedding(2 * self.max_pos_emb + 1, self.dim_head)

    def __call__(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = self.pre_norm(x)
        B, N, _ = x.shape

        num_blocks = math.ceil(N / self.context_size)
        remainder = N % self.context_size

        if remainder > 0:
            pad_len = self.context_size - remainder
            x = mx.pad(x, [(0, 0), (0, pad_len), (0, 0)])

        q = self.to_q(x)
        kv = self.to_kv(x)
        k, v = mx.split(kv, 2, axis=-1)

        q = q.reshape(B, num_blocks, self.context_size, self.num_heads, -1)
        k = k.reshape(B, num_blocks, self.context_size, self.num_heads, -1)
        v = v.reshape(B, num_blocks, self.context_size, self.num_heads, -1)

        q = q.transpose(0, 1, 3, 2, 4)
        k = k.transpose(0, 1, 3, 2, 4)
        v = v.transpose(0, 1, 3, 2, 4)

        rel_pos_emb = self.rel_pos_emb(attention_dists)

        C = self.context_size
        # Contract the head dimension directly.  Expanding q and rel_pos_emb
        # first creates a [B, blocks, heads, C, C, dim_head] temporary; for the
        # supported nine-minute input that single allocation can exceed Metal's
        # buffer-size limit by itself.
        pos_attn = mx.einsum("bnhcd,crd->bnhcr", q, rel_pos_emb) * self.scale

        if remainder > 0:
            row_valid = mx.arange(C)[:, None] < remainder
            col_valid = mx.arange(C)[None, :] < remainder
            mask = ~(row_valid & col_valid)
            mask_value = mx.array(mx.finfo(pos_attn.dtype).min, dtype=pos_attn.dtype)
            pos_attn_last = mx.where(
                mask[None, None, None], mask_value, pos_attn[:, -1:, :, :, :]
            )
            pos_attn = mx.concatenate(
                [pos_attn[:, :-1, :, :, :], pos_attn_last], axis=1
            )

        attn_weights = (q @ k.transpose(0, 1, 2, 4, 3)) * self.scale + pos_attn
        attn_weights = mx.softmax(attn_weights, axis=-1)

        out = attn_weights @ v
        out = out.transpose(0, 1, 3, 2, 4)
        out = out.reshape(B, -1, self.num_heads * self.dim_head)
        out = out[:, :N, :]
        out = self.to_out(out)
        return out


class DepthWiseConv1d(nn.Module):

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int):
        super().__init__()
        pad = kernel_size // 2
        pad_offset = (kernel_size + 1) % 2
        self.padding = (pad, pad - pad_offset)
        self.conv = nn.Conv1d(
            chan_in, chan_out, kernel_size, groups=chan_in, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, [(0, 0), (self.padding[0], self.padding[1]), (0, 0)])
        return self.conv(x)


class ConformerConvModule(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        inner_dim = config.hidden_dim * config.conv_expansion_factor

        self.norm = nn.LayerNorm(config.hidden_dim)
        self.up_conv = nn.Conv1d(config.hidden_dim, inner_dim * 2, 1)
        self.depth_conv = DepthWiseConv1d(inner_dim, inner_dim, config.conv_kernel_size)
        self.batch_norm = BatchNorm1d(inner_dim)
        self.down_conv = nn.Conv1d(inner_dim, config.hidden_dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.up_conv(x)
        x1, x2 = mx.split(x, 2, axis=-1)
        x = x1 * mx.sigmoid(x2)
        x = self.depth_conv(x)
        x = nn.silu(self.batch_norm(x))
        x = self.down_conv(x)
        return x


class ConformerBlock(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.ff1 = ConformerFeedForward(config)
        self.attn = ConformerAttention(config)
        self.conv = ConformerConvModule(config)
        self.ff2 = ConformerFeedForward(config)
        self.post_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(self, x: mx.array, attention_dists: mx.array) -> mx.array:
        x = 0.5 * self.ff1(x) + x
        x = self.attn(x, attention_dists) + x
        x = self.conv(x) + x
        x = 0.5 * self.ff2(x) + x
        x = self.post_norm(x)
        return x


class CTCEncoder(nn.Module):

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.input_linear = nn.Linear(config.input_dim, config.hidden_dim)
        self.layers = [ConformerBlock(config) for _ in range(config.num_layers)]
        self.out = nn.Linear(config.hidden_dim, config.output_dim)
        self.out_mid = nn.Linear(config.output_dim, config.hidden_dim)
        self.num_layers = config.num_layers
        self._attention_dists = None

        seq = mx.arange(config.context_size)
        relpos_dist = seq[:, None] - seq[None, :]
        self._attention_dists = (
            mx.clip(relpos_dist, -config.context_size, config.context_size)
            + config.max_pos_emb
        )
        mx.eval(self._attention_dists)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.input_linear(x)
        cat_layers = set(self.config.cat_hidden_layers or ())
        exported = [x] if 0 in cat_layers else []
        for idx, layer in enumerate(self.layers, start=1):
            x = layer(x, attention_dists=self._attention_dists)
            # HF exports the hidden state before the mid-layer CTC injection.
            if idx in cat_layers:
                exported.append(x)
            if idx == self.num_layers // 2:
                x_mid = self.out(x)
                x = x + self.out_mid(mx.softmax(x_mid, axis=-1))
        if exported:
            x = mx.concatenate([*exported, x], axis=-1)
        return x


class QFormerMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, kv_hidden_size: int = None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kv_dim = kv_hidden_size or hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(kv_dim, hidden_size)
        self.value = nn.Linear(kv_dim, hidden_size)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array = None
    ) -> mx.array:
        B, L, _ = hidden_states.shape

        q = self.query(hidden_states)
        kv_input = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        k = self.key(kv_input)
        v = self.value(kv_input)

        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = self.head_dim**-0.5
        attn = (q * scale) @ k.transpose(0, 1, 3, 2)
        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return out


class QFormerSelfOutput(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_hidden_size: int = None,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = QFormerMultiHeadAttention(
            hidden_size, num_heads, kv_hidden_size
        )
        self.output = QFormerSelfOutput(hidden_size, eps)

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array = None
    ) -> mx.array:
        attn_out = self.attention(hidden_states, encoder_hidden_states)
        return self.output(attn_out, hidden_states)


class QFormerIntermediate(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(self.dense(x))


class QFormerOutput(nn.Module):

    def __init__(self, intermediate_size: int, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class QFormerLayer(nn.Module):

    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.attention = QFormerAttention(
            config.hidden_size, config.num_attention_heads, eps=config.layer_norm_eps
        )
        self.crossattention = QFormerAttention(
            config.hidden_size,
            config.num_attention_heads,
            kv_hidden_size=config.encoder_hidden_size,
            eps=config.layer_norm_eps,
        )
        self.intermediate_query = QFormerIntermediate(
            config.hidden_size, config.intermediate_size
        )
        self.output_query = QFormerOutput(
            config.intermediate_size, config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        hidden_states = self.attention(hidden_states)
        hidden_states = self.crossattention(hidden_states, encoder_hidden_states)
        intermediate = self.intermediate_query(hidden_states)
        hidden_states = self.output_query(intermediate, hidden_states)
        return hidden_states


class QFormerEncoder(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layer = [QFormerLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(
        self, hidden_states: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        for layer in self.layer:
            hidden_states = layer(hidden_states, encoder_hidden_states)
        return hidden_states


class QFormerModel(nn.Module):
    def __init__(self, config: ProjectorConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = QFormerEncoder(config)

    def __call__(
        self, query_embeds: mx.array, encoder_hidden_states: mx.array
    ) -> mx.array:
        hidden_states = self.layernorm(query_embeds)
        return self.encoder(hidden_states, encoder_hidden_states)


class EncoderProjector(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.projector_config.hidden_size
        self.downsample_rate = config.downsample_rate
        self.window_size = config.window_size
        self.num_queries = config.window_size // config.downsample_rate

        self.query = mx.zeros(
            (1, self.num_queries, config.projector_config.hidden_size)
        )
        self.qformer = QFormerModel(config.projector_config)
        self.linear = nn.Linear(
            config.projector_config.hidden_size, config.text_config.hidden_size
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, L, D = hidden_states.shape
        nblocks = math.ceil(L / self.window_size)
        pad = nblocks * self.window_size - L
        if pad > 0:
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, pad), (0, 0)])

        hidden_states = hidden_states.reshape(B * nblocks, self.window_size, D)

        query = mx.broadcast_to(
            self.query, (B * nblocks, self.num_queries, self.hidden_size)
        )

        query_output = self.qformer(query, hidden_states)
        query_proj = self.linear(
            query_output.reshape(B, nblocks * self.num_queries, -1)
        )
        return query_proj


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = CTCEncoder(config.encoder_config)
        self.projector = EncoderProjector(config)
        text_args = GraniteModelArgs.from_dict(
            config.text_config.__dict__
            if hasattr(config.text_config, "__dict__")
            else config.text_config
        )
        self.language_model = GraniteLM(text_args)

        self.audio_token_id = config.audio_token_index
        self._tokenizer = None

    @property
    def layers(self):
        return self.language_model.model.layers

    @property
    def is_plus(self) -> bool:
        # mlx_audio.convert rewrites model_type to the module name
        # ("granite_speech"), so detect the plus variant by its architectural
        # fingerprint, which survives conversion.
        return self.config.model_type == "granite_speech_plus" or bool(
            self.config.encoder_config.cat_hidden_layers
        )

    def make_cache(self) -> List[KVCache]:
        return [KVCache() for _ in range(len(self.layers))]

    def __call__(
        self,
        input_ids: mx.array,
        cache: Optional[List[KVCache]] = None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.language_model.model.embed_tokens(input_ids)

        h = h * self.language_model.model.embedding_multiplier

        if cache is None:
            cache = [None] * len(self.language_model.model.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.language_model.model.layers, cache):
            h = layer(h, mask, cache=c)

        h = self.language_model.model.norm(h)

        if self.language_model.args.tie_word_embeddings:
            logits = self.language_model.model.embed_tokens.as_linear(h)
        else:
            logits = self.language_model.lm_head(h)

        return logits / self.language_model.logits_scaling

    def get_audio_features(self, input_features: mx.array) -> mx.array:
        encoder_dtype = self.encoder.input_linear.weight.dtype
        if input_features.dtype != encoder_dtype:
            input_features = input_features.astype(encoder_dtype)
        encoder_output = self.encoder(input_features)
        projected = self.projector(encoder_output)
        return projected

    def validate_generation_request(
        self,
        *,
        task: str = "asr",
        prompt: Optional[str] = None,
        language: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prefix_text: Optional[str] = None,
        hotwords: Optional[Union[str, List[str]]] = None,
        word_timestamps: bool = False,
        **_: object,
    ) -> None:
        """Validate cheap checkpoint capability constraints before generation."""
        normalized_task = _normalize_task(task, word_timestamps=word_timestamps)
        if normalized_task != "asr" and not self.is_plus:
            raise UnsupportedTranscriptionTask(
                f"task={normalized_task!r} requires a Granite Speech Plus "
                f"checkpoint, but the loaded model has "
                f"model_type={self.config.model_type!r}. Load "
                "ibm-granite/granite-speech-4.1-2b-plus or use task='asr'."
            )
        _resolve_prompt(normalized_task, prompt, language=language)
        _reject_reserved_audio_token(
            prompt=prompt,
            language=language,
            system_prompt=system_prompt,
            prefix_text=prefix_text,
            hotwords=hotwords,
        )

    def model_quant_predicate(self, p: str, m: nn.Module) -> bool:
        return not (p.startswith("encoder") or p.startswith("projector"))

    @staticmethod
    def sanitize(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
        sanitized = {}
        for k, v in weights.items():
            if "num_batches_tracked" in k:
                continue

            if (
                any(name in k for name in ["up_conv", "down_conv", "depth_conv"])
                and k.endswith("weight")
                and len(v.shape) == 3
            ):
                # MLX Conv1d expects weights in (out_channels, kernel_size, in_channels)
                # layout, while PyTorch uses (out_channels, in_channels, kernel_size).
                # Use each convolution's singleton dimension to distinguish those
                # layouts, making sanitization safe both during conversion and when
                # loading the resulting unquantized checkpoint.
                is_pointwise = "up_conv" in k or "down_conv" in k
                pytorch_pointwise = is_pointwise and v.shape[-1] == 1
                pytorch_depthwise = (
                    "depth_conv" in k and v.shape[1] == 1 and v.shape[-1] != 1
                )
                if pytorch_pointwise or pytorch_depthwise:
                    v = v.transpose(0, 2, 1)

            sanitized[k] = v
        return sanitized

    @classmethod
    def post_load_hook(cls, model: "Model", model_path: Path) -> "Model":
        import transformers
        from transformers import AutoTokenizer

        prev = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        try:
            model._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
        finally:
            transformers.logging.set_verbosity(prev)

        return model

    @staticmethod
    def _normalize_waveform(audio: Union[mx.array, np.ndarray]) -> mx.array:
        if isinstance(audio, np.ndarray):
            is_floating = np.issubdtype(audio.dtype, np.floating)
        else:
            is_floating = mx.issubdtype(audio.dtype, mx.floating)
        if not is_floating:
            raise ValueError(
                "Granite Speech expects normalized floating-point waveform samples; "
                "integer PCM arrays must be converted and scaled before generate()."
            )

        waveform = mx.array(audio, dtype=mx.float32)
        if waveform.ndim != 1:
            raise ValueError(
                "Granite Speech expects a mono 1-D waveform shaped (samples,), "
                f"got {waveform.shape}. Downmix multichannel audio before calling "
                "generate()."
            )
        return waveform

    def _extract_features(
        self, audio: Union[mx.array, np.ndarray]
    ) -> Tuple[mx.array, int]:
        from mlx_audio.dsp import hanning, mel_filters, stft

        n_fft = 512
        win_length = 400
        hop_length = 160
        n_mels = 80

        audio_1d = Model._normalize_waveform(audio)

        win = hanning(win_length, periodic=True)
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win_padded = mx.concatenate(
            [mx.zeros((pad_left,)), win, mx.zeros((pad_right,))]
        )

        spec = stft(
            audio_1d,
            n_fft=n_fft,
            hop_length=hop_length,
            window=win_padded,
            center=True,
            pad_mode="reflect",
        )

        power = mx.abs(spec) ** 2
        mel_fb = mel_filters(SAMPLE_RATE, n_fft, n_mels, mel_scale="htk")
        mel_spec = power @ mel_fb.T

        logmel = mx.log10(mx.clip(mel_spec, 1e-10, None))
        mx_val = mx.max(logmel)
        logmel = mx.maximum(logmel, mx_val - 8.0) / 4.0 + 1.0

        if logmel.shape[0] % 2 == 1:
            logmel = logmel[:-1]

        encoder_input = logmel.reshape(-1, 2 * n_mels)

        encoder_length = encoder_input.shape[0]
        nblocks = math.ceil(encoder_length / self.config.window_size)
        num_audio_tokens = nblocks * (
            self.config.window_size // self.config.downsample_rate
        )

        input_features = encoder_input[None, :, :]
        return input_features, num_audio_tokens

    def _build_prompt(
        self,
        num_audio_tokens: int,
        user_prompt: str = None,
        *,
        system_prompt: Optional[str] = None,
        prefix_text: Optional[str] = None,
    ) -> mx.array:
        if user_prompt is None:
            user_prompt = TASK_PROMPTS["asr"]

        # The plus checkpoint was trained with a separator after the audio
        # placeholder; the 4.0/4.1 checkpoints concatenate the instruction.
        audio_placeholder = "<|audio|>" * num_audio_tokens
        separator = " " if self.is_plus else ""
        content = f"{audio_placeholder}{separator}{user_prompt.lstrip()}"

        template = getattr(self._tokenizer, "chat_template", None)
        if template:
            chat = [{"role": "user", "content": content}]
            if system_prompt:
                chat.insert(0, {"role": "system", "content": system_prompt})
            # The plus template appends prefix_text right after the assistant
            # generation prompt; the 4.0/4.1 templates have no such hook, so
            # append by hand for them.
            native_prefix = bool(prefix_text) and "prefix_text" in template
            extra = {"prefix_text": prefix_text} if native_prefix else {}
            prompt_str = self._tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, **extra
            )
            if prefix_text and not native_prefix:
                prompt_str += prefix_text
        else:
            prompt_str = f"USER: {content}\nASSISTANT:{prefix_text or ''}"

        prompt_ids = self._tokenizer.encode(prompt_str)

        return mx.array(prompt_ids)

    def _build_inputs_embeds(
        self, input_ids: mx.array, audio_features: mx.array
    ) -> mx.array:
        is_audio = input_ids == self.audio_token_id
        llm_ids = mx.where(is_audio, 0, input_ids)

        inputs_embeds = self.language_model.model.embed_tokens(llm_ids[None])

        is_audio_np = np.array(is_audio)
        audio_positions = np.flatnonzero(is_audio_np)

        orig_dtype = inputs_embeds.dtype
        embeds_np = np.array(inputs_embeds.astype(mx.float32)).copy()
        audio_np = np.array(audio_features.astype(mx.float32))

        if audio_np.ndim != 3 or audio_np.shape[0] != 1:
            raise ValueError(
                "Granite projected audio features must have shape "
                f"(1, frames, hidden_size), got {audio_np.shape}."
            )

        placeholder_count = int(audio_positions.size)
        feature_count = int(audio_np.shape[1])
        if placeholder_count != feature_count:
            raise ValueError(
                "Granite audio placeholder mismatch: the rendered prompt "
                f"contains {placeholder_count} <|audio|> token(s), but the "
                f"projector produced {feature_count} audio frame(s). Do not "
                "include the reserved token in prompt, language, system_prompt, "
                "prefix_text, or hotwords. Also verify that the tokenizer, chat "
                "template, and model config come from the same checkpoint revision."
            )

        embeds_np[0, audio_positions] = audio_np[0]

        return mx.array(embeds_np).astype(orig_dtype)

    def generate(
        self,
        audio: Union[str, mx.array, np.ndarray],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        task: str = "asr",
        prompt: str = None,
        language: str = None,
        system_prompt: Optional[str] = None,
        prefix_text: Optional[str] = None,
        hotwords: Optional[Union[str, List[str]]] = None,
        word_timestamps: bool = False,
        prefill_step_size: int = 2048,
        verbose: bool = False,
        stream: bool = False,
        sample_rate: int = SAMPLE_RATE,
        **kwargs,
    ) -> Union[STTOutput, Generator[StreamingResult, None, None]]:
        from mlx_audio.stt.utils import merge_hotwords

        task = _normalize_task(task, word_timestamps=word_timestamps)
        Model.validate_generation_request(
            self,
            task=task,
            prompt=prompt,
            language=language,
            system_prompt=system_prompt,
            prefix_text=prefix_text,
            hotwords=hotwords,
        )
        prompt = _resolve_prompt(task, prompt, language)

        # Granite biases toward rare vocabulary via an inline "Keywords:" clause.
        keywords = merge_hotwords(None, hotwords)
        if keywords:
            prompt = f"{prompt} Keywords: {keywords}"

        if system_prompt is None and self.is_plus:
            system_prompt = PLUS_SYSTEM_PROMPT

        if stream:
            return self._stream_generate(
                audio,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
                task=task,
                prompt=prompt,
                system_prompt=system_prompt,
                prefix_text=prefix_text,
                prefill_step_size=prefill_step_size,
                verbose=verbose,
                sample_rate=sample_rate,
            )

        start_time = time.time()

        from mlx_audio.lm.generate import generate_step
        from mlx_audio.lm.sample_utils import make_logits_processors, make_sampler

        audio_data = self._load_audio(audio, sample_rate=sample_rate)
        input_features, num_audio_tokens = self._extract_features(audio_data)

        if verbose:
            print("Encoding audio...")
        audio_features = self.get_audio_features(input_features)
        mx.eval(audio_features)

        prompt_ids = self._build_prompt(
            num_audio_tokens,
            prompt,
            system_prompt=system_prompt,
            prefix_text=prefix_text,
        )
        inputs_embeds = self._build_inputs_embeds(prompt_ids, audio_features)
        mx.eval(inputs_embeds)

        prompt_tokens = len(prompt_ids)

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        eos_token_id = self._tokenizer.eos_token_id
        tokens = []
        hit_eos = False

        for token, logprobs in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds.squeeze(0),
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if int(token) == eos_token_id:
                hit_eos = True
                break
            tokens.append(int(token))

        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        finish_reason = "stop" if hit_eos else "length"
        if task != "asr" and not hit_eos:
            raise IncompleteTranscription(
                task=task,
                max_tokens=max_tokens,
                partial_text=text,
            )
        segments = _parse_segments(task, text, prefix_text)
        elapsed = time.time() - start_time
        gen_tokens = len(tokens)

        if verbose:
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Generation tokens: {gen_tokens}")
            print(f"Total time: {elapsed:.2f}s")
            if gen_tokens > 0:
                print(f"Generation TPS: {gen_tokens / elapsed:.1f}")

        return STTOutput(
            text=text,
            segments=segments,
            prompt_tokens=prompt_tokens,
            generation_tokens=gen_tokens,
            total_tokens=prompt_tokens + gen_tokens,
            total_time=elapsed,
            prompt_tps=prompt_tokens / elapsed if elapsed > 0 else 0,
            generation_tps=gen_tokens / elapsed if elapsed > 0 else 0,
            finish_reason=finish_reason,
            complete=hit_eos,
            raw_text=text if not hit_eos else None,
        )

    def _stream_generate(
        self,
        audio: Union[str, mx.array, np.ndarray],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: int = 100,
        task: str = "asr",
        prompt: str = None,
        system_prompt: Optional[str] = None,
        prefix_text: Optional[str] = None,
        prefill_step_size: int = 2048,
        verbose: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ) -> Generator[StreamingResult, None, None]:
        from mlx_audio.lm.generate import StreamingDetokenizer, generate_step
        from mlx_audio.lm.sample_utils import make_logits_processors, make_sampler

        audio_data = self._load_audio(audio, sample_rate=sample_rate)
        input_features, num_audio_tokens = self._extract_features(audio_data)

        audio_features = self.get_audio_features(input_features)
        mx.eval(audio_features)

        prompt_ids = self._build_prompt(
            num_audio_tokens,
            prompt,
            system_prompt=system_prompt,
            prefix_text=prefix_text,
        )
        inputs_embeds = self._build_inputs_embeds(prompt_ids, audio_features)
        mx.eval(inputs_embeds)

        prompt_token_count = len(prompt_ids)

        sampler = make_sampler(temperature, top_p=top_p, min_p=min_p, top_k=top_k)
        logits_processors = make_logits_processors(
            repetition_penalty=repetition_penalty,
            repetition_context_size=repetition_context_size,
        )

        eos_token_id = self._tokenizer.eos_token_id
        gen_tokens = 0
        hit_eos = False
        detokenizer = StreamingDetokenizer(self._tokenizer, skip_special_tokens=True)

        for token, _ in generate_step(
            prompt=prompt_ids,
            input_embeddings=inputs_embeds.squeeze(0),
            model=self,
            max_tokens=max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            prefill_step_size=prefill_step_size,
        ):
            if int(token) == eos_token_id:
                hit_eos = True
                break
            gen_tokens += 1
            detokenizer.add_token(token)
            if text := detokenizer.last_segment:
                yield StreamingResult(
                    text=text,
                    is_final=False,
                    start_time=None,
                    end_time=None,
                    prompt_tokens=prompt_token_count,
                    generation_tokens=gen_tokens,
                )

        detokenizer.finalize()
        full_text = detokenizer.text
        finish_reason = "stop" if hit_eos else "length"
        complete = hit_eos
        raw_text = full_text if not hit_eos else None
        error_type = None
        error = None
        segments = None

        if task != "asr" and not hit_eos:
            incomplete = IncompleteTranscription(
                task=task,
                max_tokens=max_tokens,
                partial_text=full_text,
            )
            error_type = type(incomplete).__name__
            error = str(incomplete)
        elif task != "asr":
            try:
                segments = _parse_segments(task, full_text, prefix_text)
            except StructuredTranscriptError as exc:
                complete = False
                raw_text = exc.raw_text
                error_type = type(exc).__name__
                error = str(exc)

        yield StreamingResult(
            text=detokenizer.last_segment,
            is_final=True,
            start_time=None,
            end_time=None,
            prompt_tokens=prompt_token_count,
            generation_tokens=gen_tokens,
            segments=segments,
            finish_reason=finish_reason,
            complete=complete,
            raw_text=raw_text,
            error_type=error_type,
            error=error,
        )

    def _load_audio(
        self,
        audio: Union[str, mx.array, np.ndarray],
        *,
        sample_rate: int = SAMPLE_RATE,
    ) -> mx.array:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        if isinstance(audio, str):
            from mlx_audio.stt.utils import load_audio

            return load_audio(audio, sr=SAMPLE_RATE)
        elif isinstance(audio, np.ndarray):
            waveform = audio
        elif isinstance(audio, mx.array):
            waveform = audio
        elif isinstance(audio, list):
            audio_item = audio[0]
            if isinstance(audio_item, str):
                from mlx_audio.stt.utils import load_audio

                return load_audio(audio_item, sr=SAMPLE_RATE)
            waveform = np.array(audio_item)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        waveform = Model._normalize_waveform(waveform)
        if sample_rate != SAMPLE_RATE:
            from mlx_audio.stt.utils import resample_audio

            waveform = resample_audio(waveform, sample_rate, SAMPLE_RATE)
        return waveform.astype(mx.float32)
