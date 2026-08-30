# Granite Speech Plus parity

The pull-request gate runs a deterministic, 16-layer synthetic checkpoint
through both Transformers and MLX. It compares the log-mel frontend, selected
layer 3, final and concatenated encoder outputs, Q-Former projection,
multimodal embedding merge, tied output head, and first greedy token.

```bash
pip install -e ".[all,dev,granite-parity]"
pytest -s tests/model_parity/test_granite_speech_plus.py
```

The larger checkpoint smoke is intentionally manual because it downloads and
materializes a 2B model. It uses Transformers 5.8.1 from an isolated package
target because that version has verified multi-window Q-Former batching.
Versions 5.14.1, 5.15.1, 5.16.1, and current Transformers `main` retain a
projector reshape failure when `nblocks > 1`. The smoke pins model and tokenizer
revision
`1454e6e1e33845ca9280ff65f52cf1141ba6e6e2`, loads MLX weights with
`strict=True`, compares reference/MLX first-token logits and greedy choice,
renders all three task prompts, and requires speaker and timestamp markers from
the tracked audio fixture.

```bash
reference_dir="$(mktemp -d)"
python -m pip install --no-deps --target "$reference_dir" \
  transformers==5.8.1 tokenizers==0.23.0rc0
MLX_AUDIO_RUN_GRANITE_CHECKPOINT_PARITY=1 \
  PYTHONPATH="$reference_dir" \
  pytest -s tests/model_parity/test_granite_speech_plus_checkpoint.py
```
