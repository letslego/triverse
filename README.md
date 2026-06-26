# triverse

[![Docs](https://img.shields.io/badge/docs-architecture%20guide-60a5fa?style=flat-square)](https://letslego.github.io/triverse/)
[![GitHub](https://img.shields.io/badge/github-letslego%2Ftriverse-111827?style=flat-square&logo=github)](https://github.com/letslego/triverse)

**A lightweight multi-LLM coordinator inspired by [Trinity](https://arxiv.org/abs/2512.04695) and [Omnigent](https://github.com/omnigent-ai/omnigent).**

**Architecture guide:** https://letslego.github.io/triverse/ — Trinity, triverse, Omnigent, compressionX, and how they fit together (with diagrams).

triverse orchestrates a pool of diverse LLMs through a **Thinker → Worker → Verifier** multi-turn protocol. A small coordination head selects which model plays which role each turn — without retraining the models in your pool. Harness backends are swappable (OpenAI, Anthropic, mock), so you can change providers without rewriting coordination logic.

```
┌──────────────────────────────────────────────────────────────┐
│  Your application                                            │
└────────────────────────────┬─────────────────────────────────┘
                             │ query
                             ▼
┌──────────────────────────────────────────────────────────────┐
│  Coordinator (triverse)                                      │
│                                                              │
│  ┌─────────────┐    ┌────────────────────────────────────┐   │
│  │ Router head │───▶│  Turn loop (max K turns)           │   │
│  │ agent×role  │    │    1. Select (LLM, role)           │   │
│  └─────────────┘    │    2. Inject role prompt           │   │
│                     │    3. Call harness                 │   │
│                     │    4. Append to transcript         │   │
│                     │    5. Stop on Verifier ACCEPT      │   │
│                     └────────────────────────────────────┘   │
└────────────────────────────┬─────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   ┌──────────┐       ┌──────────┐       ┌──────────┐
   │ OpenAI   │       │Anthropic │       │  mock    │
   │ harness  │       │ harness  │       │ harness  │
   └──────────┘       └──────────┘       └──────────┘
```

## Why triverse?

[Trinity](https://arxiv.org/abs/2512.04695) shows that a tiny coordinator (~20K parameters) can orchestrate diverse LLMs through three roles — **Thinker** (plan), **Worker** (execute), **Verifier** (validate) — and outperform any single model in the pool. The paper trains its head with sep-CMA-ES on hidden states from a 0.6B SLM.

triverse makes that protocol **practical today**:

| Trinity concept | triverse implementation |
|-----------------|-------------------------|
| Tri-role coordination (T/W/V) | Role-specific prompts + verifier ACCEPT/REVISE contract |
| Multi-turn transcript | `Transcript` accumulates query + condensed outputs |
| Lightweight coordination head | `CoordinationRouter` — transparent feature scorer, tunable weights |
| Diverse LLM pool | `ModelPool` with per-agent harness + strengths |
| Agent–role action space | Router scores every `(agent_id, role)` pair each turn |

[Omnigent](https://github.com/omnigent-ai/omnigent) contributes the **meta-harness** pattern: a uniform `Harness.complete()` interface lets you swap OpenAI, Anthropic, or custom backends via YAML — no coordination rewrite.

**[compressionX](../compressionX)** (sibling package) compresses growing transcripts and turn outputs — JsonCrusher for tool results, LogCompressor for build output, CXR for reversible retrieval — so multi-turn coordination stays within token budgets.

## Installation

```bash
pip install -e .

# With LLM provider SDKs + compressionX
pip install -e ../compressionX   # from monorepo sibling
pip install -e ".[all]"

# Development
pip install -e ".[dev]"
```

Requires Python 3.10+.

## Quick start

### Demo (no API keys)

```bash
triverse demo
```

### Python API

```python
from triverse import Coordinator, CoordConfig
from triverse.pool import ModelPool

pool = ModelPool.default_demo()
coord = Coordinator(pool, CoordConfig(max_turns=5, seed=42))

result = coord.run(
    "Implement binary search in Python and verify correctness."
)

print(result.answer)
print(f"Completed in {result.total_turns} turns via {result.terminated_by}")
print(f"Saved {result.tokens_saved} tokens via compressionX")
```

### With compressionX (default)

Compression is **on by default** when `compressionx` is installed. Each turn:

1. **Before harness call** — older transcript turns are compressed (`protect_recent_turns=1` keeps the latest intact)
2. **After agent response** — large turn outputs are compressed before appending

```python
from triverse import Coordinator, CoordConfig, CompressionConfig

coord = Coordinator(
    pool,
    CoordConfig(
        compression=CompressionConfig(
            target_ratio=0.25,       # aggressive for logs/tool output
            compress_prompt=True,
            compress_outputs=True,
            enable_cxr=True,         # originals in ~/.compressionx/cxr.db
        ),
    ),
)
```

Disable with `compression=None` or CLI `--no-compress`. Without compressionX installed, coordination works unchanged (graceful fallback).

```bash
python examples/compressed_coordination.py
```

### CLI

```bash
triverse run "Explain quantum tunneling and verify the explanation"
triverse run "Debug this API" --pool my_pool.yaml --max-turns 4 --no-compress
triverse init-pool agents.yaml   # generate starter config
```

## Agent pool (YAML)

```yaml
agents:
  - id: gpt
    harness: openai
    model: gpt-4o
    strengths: [reasoning, coding]

  - id: claude
    harness: anthropic
    model: claude-sonnet-4-20250514
    strengths: [reasoning, verification]

  - id: fast
    harness: openai
    model: gpt-4o-mini
    strengths: [knowledge]
```

Set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` for live harnesses. Use `harness: mock` for offline testing.

## How coordination works

Each turn:

1. **State** — query + full transcript rendered for the router
2. **Route** — scorer picks `(agent_id, role)` using phase bias, domain signals, and agent strengths
3. **Prompt** — role-specific system contract prepended to transcript
4. **Execute** — selected harness calls the model
5. **Post-process** — output condensed into transcript; verifier parses `VERDICT: ACCEPT|REVISE`
6. **Terminate** — on Verifier ACCEPT, or when `max_turns` exhausted

### Role contracts

| Role | Responsibility |
|------|----------------|
| **Thinker** | Strategy, decomposition, meta-guidance — no full execution |
| **Worker** | Concrete progress: code, math, direct answers |
| **Verifier** | Correctness check; `VERDICT: ACCEPT` stops the loop |

## Custom harnesses

```python
from triverse.harness import Harness, HarnessResponse
from triverse.pool import ModelPool
from triverse import Coordinator

class MyHarness(Harness):
    @property
    def name(self) -> str:
        return "my-backend"

    def complete(self, prompt, *, model, temperature=0.7):
        # Call your agent runtime here
        return HarnessResponse(content="...", model=model)

pool = ModelPool.default_demo()
pool.register_harness("my-backend", MyHarness())
```

## Project layout

```
triverse/
├── triverse/
│   ├── coordinator.py   # Main Trinity loop
│   ├── compression.py   # compressionX adapter
│   ├── router.py        # Coordination head
│   ├── roles.py         # T/W/V prompts
│   ├── transcript.py    # Multi-turn state
│   ├── pool.py          # Agent pool
│   └── harness/         # Swappable backends
├── examples/
├── tests/
└── pyproject.toml
```

## Development

```bash
make install
make test
make demo
```

## Relation to Trinity (arxiv:2512.04695)

triverse implements Trinity's **coordination protocol** and **decision surface** (agent × role per turn). The paper's CMA-ES-trained linear head on SLM hidden states is the production path for learned routing; triverse's `CoordinationRouter` provides an interpretable baseline with the same action space, ready for weight export/import when you train a head offline.

## License

MIT
