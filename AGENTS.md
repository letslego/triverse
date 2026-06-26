# triverse

Lightweight multi-LLM coordinator for this repo. Trinity-inspired T/W/V protocol; Omnigent-style harness swapping.

## Commands

```bash
pip install -e ".[dev]"
pytest
triverse demo
triverse run "your query" --pool examples/pool.mock.yaml
```

## Layout

- `coordinator.py` — multi-turn Trinity loop
- `router.py` — feature-based coordination head
- `roles.py` — Thinker / Worker / Verifier prompts
- `harness/` — swappable LLM backends (mock, openai, anthropic)
- `pool.py` — agent pool from YAML
