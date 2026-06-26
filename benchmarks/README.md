# Benchmarks

Compare triverse against [OpenFugu](https://github.com/trotsky1997/OpenFugu).

## Prerequisites

```bash
git clone --depth 1 https://github.com/trotsky1997/OpenFugu.git /tmp/OpenFugu
pip install cma numpy
pip install -e ../compressionX
pip install -e .

# Optional: full OpenFugu Qwen3 loop
pip install torch transformers huggingface_hub

# 37-case fixture (always works):
curl -fsSL -o /tmp/OpenFugu/artifacts/qwen_router_prompt_eval_cases.json \
  https://raw.githubusercontent.com/nshkrdotcom/trinity_coordinator/main/examples/fixtures/qwen_router_prompt_eval_cases.json

# model_iter_60.npy — OpenFugu's fetch script may 404; the HF dataset now ships
# safetensors (nshkrdotcom/trinity-coordinator-adapted-qwen3-0.6b). If you have
# the legacy .npy from Sakana/OpenFugu sources, set:
export FUGU_VECTOR=/path/to/model_iter_60.npy
export FUGU_MODEL=/path/to/Qwen3-0.6B   # huggingface-cli download Qwen/Qwen3-0.6B
export FUGU_FIXTURE=/tmp/OpenFugu/artifacts/qwen_router_prompt_eval_cases.json
```

## Run

```bash
python benchmarks/compare_openfugu.py
```

Results written to `benchmarks/last_run.json` (gitignored).

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENFUGU_ROOT` | `/tmp/OpenFugu` | Path to OpenFugu clone |
| `BENCH_N_TASKS` | `5000` | Mock routing eval task count |
| `FUGU_MODEL` | — | Qwen3-0.6B directory for full loop |
