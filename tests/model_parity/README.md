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
materializes a 2B model. It pins model and tokenizer revision
`1454e6e1e33845ca9280ff65f52cf1141ba6e6e2`, loads MLX weights with
`strict=True`, compares reference/MLX first-token logits and greedy choice,
renders all three task prompts, and requires speaker and timestamp markers from
the tracked audio fixture.

```bash
MLX_AUDIO_RUN_GRANITE_CHECKPOINT_PARITY=1 \
  pytest -s tests/model_parity/test_granite_speech_plus_checkpoint.py
```
