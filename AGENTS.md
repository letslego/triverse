# triverse

Lightweight multi-LLM coordinator. Trinity-inspired T/W/V protocol; Omnigent-style harness swapping; compressionX for transcript compression.

## Commands

```bash
pip install -e ../compressionX   # sibling package in monorepo
pip install -e ".[all]"
pytest
make bench      # compare vs OpenFugu
triverse demo
triverse run "query" --no-compress   # disable compressionX
```

## Layout

- `coordinator.py` — multi-turn Trinity loop (calls compression layer)
- `compression.py` — compressionX adapter for prompts and turn outputs
- `router.py` — feature-based coordination head
- `roles.py` — Thinker / Worker / Verifier prompts
- `harness/` — swappable LLM backends
- `pool.py` — agent pool from YAML
