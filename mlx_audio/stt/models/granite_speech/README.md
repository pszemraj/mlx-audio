# Granite Speech

MLX implementation of IBM's Granite Speech, a speech-to-text model that combines a CTC Conformer encoder with a Granite LLM decoder via a BLIP-2 QFormer projector. Supports ASR (transcription), AST (speech translation), and — with the plus checkpoint — speaker-attributed ASR and word-level timestamps.

## Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [ibm-granite/granite-4.0-1b-speech](https://huggingface.co/ibm-granite/granite-4.0-1b-speech) | ~1B | Speech recognition and translation |
| [ibm-granite/granite-speech-4.1-2b-plus](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus) | ~2B | Rich transcription: speaker attribution, word timestamps, keyword biasing (EN, FR, DE, ES, PT; no punctuation/casing) |

**Supported Languages:** English, French, German, Spanish, Portuguese, Japanese (plus checkpoint: no Japanese)

## CLI Usage

```bash
# Basic transcription
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output

# Verbose output with timing info
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --verbose

# Streaming output
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --stream

# Translate to French using language flag
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --language fr

# Translate using full language name
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --language Portuguese

# Output formats: txt, srt, vtt, json
mlx_audio.stt.generate --model ibm-granite/granite-4.0-1b-speech --audio audio.wav --output-path output --format json
```

## Python Usage

### ASR (Transcription)

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

# Basic transcription (default prompt)
result = model.generate("audio.wav")
print(result.text)

# With custom prompt
result = model.generate("audio.wav", prompt="Translate the speech to text.")
print(result.text)
```

### AST (Speech Translation)

Use the `language` parameter to translate speech. Accepts full names or codes (`fr`, `de`, `es`, `pt`, `ja`):

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

# Translate speech to French (using language code)
result = model.generate("audio.wav", language="fr")
print(result.text)

# Translate speech to Spanish (using full name)
result = model.generate("audio.wav", language="Spanish")
print(result.text)

# Translate speech to Portuguese
result = model.generate("audio.wav", language="pt")
print(result.text)

# Or use a custom prompt directly
result = model.generate("audio.wav", prompt="Translate the speech to German.")
print(result.text)
```

> **Note:** If the model receives an unfamiliar prompt, it falls back to transcription as the default mode.

### Rich Transcription (granite-speech-4.1-2b-plus)

The plus checkpoint selects its mode through the prompt; the `task` parameter picks the right one. Audio limits from the model card: up to 9 minutes for `asr`/`saa`, up to 3.5 minutes for `timestamps`. Timestamps mode emits roughly one tag per word, so budget `max_tokens` accordingly. Plus-mode transcripts have no punctuation or casing.

The canonical `saa` and `timestamps` prompts define their output schemas and
cannot be replaced with `prompt=`. Use `hotwords=` for contextual biasing, or
use `task="asr"` when supplying a custom instruction. Rich modes fail closed if
the checkpoint returns plain or malformed text instead of the requested tags.

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-speech-4.1-2b-plus")

# Speaker-attributed ASR: [Speaker N]: tags, parsed into segments
result = model.generate("meeting.wav", task="saa")
for seg in result.segments:
    print(f"Speaker {seg['speaker_id']}: {seg['text']}")

# Word-level timestamps: [T:N] tags, parsed into one segment with word timings
result = model.generate("audio.wav", task="timestamps", max_tokens=8192)
for word in result.segments[0]["words"]:
    print(f"{word['word']}\t{word['start']:.2f}-{word['end']:.2f}s")

# Keyword biasing (names, technical terms) works with any task. Pass either
# one keyword or a list.
result = model.generate("audio.wav", hotwords="Nativ")
result = model.generate("audio.wav", hotwords=["Nativ", "QFormer"])
```

From the CLI, the plus-specific parameters go through `--gen-kwargs`:

```bash
mlx_audio.stt.generate --model ibm-granite/granite-speech-4.1-2b-plus \
  --audio meeting.wav --output-path output --format json \
  --gen-kwargs '{"task": "saa", "hotwords": ["Acme Ledger", "Q3 close"]}'
```

#### Incremental decoding with `prefix_text`

`prefix_text` seeds the assistant response with the transcript produced so far,
so the model emits only the continuation. IBM's incremental-decoding recipe
accumulates both the audio context and the transcript; in SAA mode, the prefix
also gives the model the existing speaker numbering:

```python
import mlx.core as mx

from mlx_audio.stt import load
from mlx_audio.stt.utils import load_audio

model = load("ibm-granite/granite-speech-4.1-2b-plus")
previous_transcript = ""
all_audio = None

for path in ["meeting-000.wav", "meeting-001.wav", "meeting-002.wav"]:
    new_audio = load_audio(path)  # mono, resampled to 16 kHz
    all_audio = (
        new_audio if all_audio is None else mx.concatenate([all_audio, new_audio])
    )
    result = model.generate(
        all_audio,
        task="saa",
        prefix_text=previous_transcript or None,
    )
    print(result.text)  # only the newly decoded continuation
    previous_transcript = f"{previous_transcript} {result.text}".strip()

print(previous_transcript)  # complete transcript
```

This avoids generating the old transcript again, but it is not cached or online
audio inference. Each call re-extracts features, re-encodes all accumulated
audio, and prefills a new decoder cache with `previous_transcript`, so cost grows
with the cumulative recording length. Keep the accumulated input within the
checkpoint's limits: 9 minutes for `asr`/`saa` and 3.5 minutes for `timestamps`.
The prefix also consumes decoder context, so leave headroom rather than treating
the published no-prefix audio limit as a safe incremental target. Shorter input
increments provide more frequent speaker decisions, but do not bound total cost
or context because both `all_audio` and `previous_transcript` still grow. For a
longer recording, finish one bounded window and start another; speaker identities
then need to be reconciled across windows by the caller.

### Streaming

```python
from mlx_audio.stt import load

model = load("ibm-granite/granite-4.0-1b-speech")

for result in model.generate("audio.wav", stream=True):
    print(result.text, end="", flush=True)
```

For `saa` and `timestamps`, parsed segments are attached to the final streaming
result. Token deltas are untimed; they are not word-alignment cues. Granite
encodes the complete supplied recording before yielding decoder tokens, so this
mode does not provide online audio ingestion or reusable state across calls.

### Generation Parameters

```python
result = model.generate(
    "audio.wav",
    max_tokens=4096,
    temperature=0.0,       # 0 = greedy decoding
    top_p=1.0,
    top_k=0,
    repetition_penalty=None,
    prompt="Translate the speech to text.",
    prefill_step_size=2048,
    verbose=True,          # print timing info
)
```

## Architecture

- **Encoder**: CTC Conformer (16 layers, 1024 hidden dim, Shaw's relative positional embeddings, block-wise attention with context_size=200)
- **Projector**: BLIP-2 QFormer (2 layers, windowed cross-attention with window_size=15, downsample_rate=5)
- **Decoder**: Granite LLM (40 layers, 2048 hidden dim, GQA with 16/4 heads, RoPE, SwiGLU MLP)
- Audio input: 16 kHz, 80-bin mel spectrogram with pair stacking (160-dim input)

## Audio Input

Granite Speech extracts features at **16 kHz**. File inputs are resampled from
their stored sample rate. In-memory arrays are assumed to be 16 kHz unless their
actual rate is passed to `generate`, for example
`model.generate(samples, sample_rate=48000)`. In-memory waveforms must be mono,
normalized floating-point arrays shaped `(samples,)`; downmix multichannel PCM
and scale integer PCM to floating point before calling `generate`. File inputs
are decoded and normalized automatically. Supported input types:

- File path (WAV, FLAC, MP3, etc.)
- NumPy array (raw waveform)
- MLX array (raw waveform)

## Output Format

```python
STTOutput(
    text="Full transcription text",
    segments=[],
    prompt_tokens=154,
    generation_tokens=42,
    total_tokens=196,
    total_time=0.95,
    prompt_tps=162.1,
    generation_tps=44.2,
)
```
