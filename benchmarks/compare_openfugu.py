#!/usr/bin/env python3
"""Head-to-head benchmark: triverse vs OpenFugu (mock routing world).

Runs OpenFugu's MockWorld orchestration eval and an equivalent triverse
heuristic-router eval on the same task distribution, plus unit-test suites.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPENFUGU = Path(os.environ.get("OPENFUGU_ROOT", "/tmp/OpenFugu"))

# OpenFugu mock world (same geometry as train/train_trinity.py)
sys.path.insert(0, str(OPENFUGU / "train"))
from train_trinity import HEAD_ROWS, N_DOMAINS, N_WORKERS, MockWorld, route  # noqa: E402

from triverse.pool import ModelPool
from triverse.router import CoordinationRouter
from triverse.types import AgentSpec, Role

# Domain-indexed queries tuned for triverse keyword router
DOMAIN_QUERIES = [
    "Calculate the integral of this equation and solve the algebra problem.",
    "Implement a Python function for this algorithm and debug the API code.",
    "What is the definition and history — who discovered this fact?",
    "Explain why this comparison fails and analyze the reasoning step by step.",
]

DOMAIN_AGENTS = [
    AgentSpec(id="math", harness="mock", model="m", strengths=["math"]),
    AgentSpec(id="code", harness="mock", model="c", strengths=["coding"]),
    AgentSpec(id="facts", harness="mock", model="f", strengths=["knowledge"]),
    AgentSpec(id="reason", harness="mock", model="r", strengths=["reasoning"]),
]

AGENT_TO_WORKER = {"math": 0, "code": 1, "facts": 2, "reason": 3}


def eval_openfugu_coordinator(world: MockWorld, head_vec: np.ndarray, n_tasks: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, feat = world.sample_task(rng)
        worker, _role = route(head_vec, feat)
        total += world.solve(domain, worker, rng)
    return total / n_tasks


def eval_triverse_router(world: MockWorld, n_tasks: int, seed: int) -> tuple[float, float]:
    """Route via triverse heuristic on domain query text; score worker choice only."""
    router = CoordinationRouter(DOMAIN_AGENTS, temperature=0.01, seed=seed)
    rng = np.random.default_rng(seed)
    total = 0.0
    correct_specialist = 0
    for _ in range(n_tasks):
        domain, _feat = world.sample_task(rng)
        query = DOMAIN_QUERIES[domain]
        state = query
        agent_id, role, _ = router.route(state, turn_index=1, max_turns=5)
        worker = AGENT_TO_WORKER.get(agent_id, 0)
        total += world.solve(domain, worker, rng)
        if worker == domain:
            correct_specialist += 1
    return total / n_tasks, correct_specialist / n_tasks


def eval_best_single(world: MockWorld, n_tasks: int, seed: int) -> tuple[float, int]:
    rng = np.random.default_rng(seed)
    scores = []
    for w in range(N_WORKERS):
        r = np.random.default_rng(seed)
        total = sum(world.solve(int(rng.integers(N_DOMAINS)), w, r) for _ in range(n_tasks))
        scores.append(total / n_tasks)
    best_w = int(np.argmax(scores))
    return max(scores), best_w


def eval_oracle(world: MockWorld, n_tasks: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_tasks):
        domain, _ = world.sample_task(rng)
        total += world.solve(domain, domain, rng)
    return total / n_tasks


def run_subprocess(cmd: list[str], cwd: Path) -> dict:
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "seconds": round(time.perf_counter() - start, 2),
        "stdout": proc.stdout[-4000:] if len(proc.stdout) > 4000 else proc.stdout,
        "stderr": proc.stderr[-2000:] if len(proc.stderr) > 2000 else proc.stderr,
    }


def main() -> int:
    n_tasks = int(os.environ.get("BENCH_N_TASKS", "5000"))
    seed = 7
    world_seed = 42
    world = MockWorld(seed=world_seed)

    print("=" * 70)
    print("ROUTING BENCHMARK (OpenFugu MockWorld — same task distribution)")
    print("=" * 70)
    print(f"Tasks: {n_tasks}  |  world_seed: {world_seed}  |  eval_seed: {seed}")
    print(f"Chance: {world.chance:.3f}  |  Oracle ceiling: {world.optimal:.3f}\n")

    # Train or load OpenFugu coordinator
    coord_path = OPENFUGU / "eval" / "trinity_mock.npy"
    if not coord_path.exists():
        print("[openfugu] Training sep-CMA-ES coordinator (60 iters)...")
        run_subprocess(
            [sys.executable, str(OPENFUGU / "train" / "train_trinity.py"), "--out", str(coord_path)],
            OPENFUGU / "train",
        )
    head = np.load(coord_path)
    openfugu_score = eval_openfugu_coordinator(world, head, n_tasks, seed)
    triverse_score, triverse_specialist_rate = eval_triverse_router(world, n_tasks, seed)
    best_single, best_worker = eval_best_single(world, n_tasks, seed)
    oracle = eval_oracle(world, n_tasks, seed)

    print(f"  best single worker (w{best_worker}) : {best_single:.3f}")
    print(f"  oracle (specialist routing)        : {oracle:.3f}")
    print(f"  OpenFugu (CMA-ES trained head)     : {openfugu_score:.3f}")
    print(f"  triverse (heuristic router)        : {triverse_score:.3f}")
    print(f"  triverse specialist pick rate      : {triverse_specialist_rate:.1%}")

    openfugu_lift = (openfugu_score - best_single) / best_single * 100
    triverse_lift = (triverse_score - best_single) / best_single * 100
    print(f"\n  OpenFugu lift vs best single: {openfugu_lift:+.1f}%")
    print(f"  triverse lift vs best single: {triverse_lift:+.1f}%")

    winner = "OpenFugu" if openfugu_score > triverse_score + 0.005 else (
        "triverse" if triverse_score > openfugu_score + 0.005 else "tie"
    )
    print(f"\n  Routing winner (mock reward): {winner}")

    # Per-domain routing table for triverse
    print("\n  triverse per-domain agent picks (turn 1, worker role bias):")
    router = CoordinationRouter(DOMAIN_AGENTS, temperature=0.01, seed=0)
    for d, q in enumerate(DOMAIN_QUERIES):
        agent_id, role, _ = router.route(q, turn_index=0, max_turns=5)
        w = AGENT_TO_WORKER.get(agent_id, -1)
        mark = "OK" if w == d else "miss"
        print(f"    domain {d} -> {agent_id} ({role.value})  [{mark}]")

    print("\n" + "=" * 70)
    print("TEST SUITES")
    print("=" * 70)

    results = {}

    # triverse pytest
    results["triverse_pytest"] = run_subprocess(
        [sys.executable, "-m", "pytest", "-v", "--tb=short"],
        ROOT,
    )
    print(f"\n[triverse pytest] exit={results['triverse_pytest']['exit_code']} "
          f"({results['triverse_pytest']['seconds']}s)")

    # OpenFugu train self-check
    results["openfugu_train"] = run_subprocess(
        [sys.executable, str(OPENFUGU / "train" / "train_trinity.py"), "--iters", "30", "--out", "/tmp/triverse_bench_train.npy"],
        OPENFUGU / "train",
    )
    print(f"[openfugu train_trinity] exit={results['openfugu_train']['exit_code']} "
          f"({results['openfugu_train']['seconds']}s)")

    # OpenFugu eval orchestration
    results["openfugu_eval"] = run_subprocess(
        [sys.executable, str(OPENFUGU / "eval" / "eval_orchestration.py"),
         "--coordinator", str(coord_path), "--n-tasks", str(n_tasks)],
        OPENFUGU / "eval",
    )
    print(f"[openfugu eval_orchestration] exit={results['openfugu_eval']['exit_code']} "
          f"({results['openfugu_eval']['seconds']}s)")
    if results["openfugu_eval"]["stdout"]:
        for line in results["openfugu_eval"]["stdout"].splitlines()[-8:]:
            print(f"  {line}")

    print("\n" + "=" * 70)
    print("COORDINATION LOOP (mock workers, same query)")
    print("=" * 70)

    from triverse import Coordinator, CoordConfig
    from triverse.compression import ContextCompressor

    query = "Implement binary search in Python and verify correctness."
    t0 = time.perf_counter()
    triverse_result = Coordinator(ModelPool.default_demo(), CoordConfig(max_turns=5, seed=42)).run(query)
    triverse_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  triverse: {triverse_result.total_turns} turns, "
          f"terminated_by={triverse_result.terminated_by}, {triverse_ms:.0f}ms")
    print(f"    compression: {triverse_result.compressions_applied} applied, "
          f"{triverse_result.tokens_saved} tokens saved")

    # OpenFugu 37-case routing self-test (needs Qwen3 + model_iter_60.npy)
    vector = Path(os.environ.get("FUGU_VECTOR", str(OPENFUGU / "artifacts" / "model_iter_60.npy")))
    fixture = Path(os.environ.get("FUGU_FIXTURE", str(OPENFUGU / "artifacts" / "qwen_router_prompt_eval_cases.json")))
    model_dir = os.environ.get("FUGU_MODEL", "")

    if fixture.exists() and vector.exists() and model_dir and Path(model_dir).exists():
        results["openfugu_selftest"] = run_subprocess(
            [sys.executable, str(OPENFUGU / "openfugu" / "mini.py"),
             "--self-test", "--model", model_dir, "--vector", str(vector), "--fixture", str(fixture)],
            OPENFUGU,
        )
        print(f"\n[openfugu mini --self-test] exit={results['openfugu_selftest']['exit_code']} "
              f"({results['openfugu_selftest']['seconds']}s)")
        for line in results["openfugu_selftest"]["stdout"].splitlines()[-6:]:
            print(f"  {line}")
    else:
        missing = []
        if not fixture.exists():
            missing.append("fixture (curl from trinity_coordinator)")
        if not vector.exists():
            missing.append("model_iter_60.npy (HF dataset now ships safetensors; see benchmarks/README.md)")
        if not model_dir or not Path(model_dir).exists():
            missing.append("FUGU_MODEL=Qwen3-0.6B path")
        print(f"\n[openfugu mini --self-test] skipped — missing: {', '.join(missing)}")
        results["openfugu_selftest"] = {"exit_code": None, "skipped": True, "missing": missing}
    openfugu_loop = {"available": False}
    if fixture.exists() and vector.exists() and model_dir and Path(model_dir).exists():
        try:
            import torch  # noqa: F401
            sys.path.insert(0, str(OPENFUGU))
            from openfugu.mini import Coordinator as FuguCoord, FuguRouter, MockWorker

            t0 = time.perf_counter()
            fugu_router = FuguRouter(model_dir, str(vector), seed=42)
            fugu_res = FuguCoord(fugu_router, MockWorker(), sample=True).run(query)
            openfugu_loop = {
                "available": True,
                "turns": len(fugu_res.turns),
                "terminated_by": fugu_res.terminated_by,
                "ms": round((time.perf_counter() - t0) * 1000, 0),
            }
            print(f"  OpenFugu: {openfugu_loop['turns']} turns, "
                  f"terminated_by={openfugu_loop['terminated_by']}, {openfugu_loop['ms']:.0f}ms")
        except Exception as exc:
            print(f"  OpenFugu full loop: skipped ({exc})")
    else:
        print("  OpenFugu full loop: skipped (set FUGU_MODEL + FUGU_VECTOR; see benchmarks/README.md)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    summary = {
        "routing": {
            "openfugu_reward": round(openfugu_score, 4),
            "triverse_reward": round(triverse_score, 4),
            "best_single": round(best_single, 4),
            "oracle": round(oracle, 4),
            "winner": winner,
        },
        "tests": {
            "triverse_pytest_pass": results["triverse_pytest"]["exit_code"] == 0,
            "openfugu_train_pass": results["openfugu_train"]["exit_code"] == 0,
            "openfugu_eval_pass": results["openfugu_eval"]["exit_code"] == 0,
            "openfugu_selftest": results.get("openfugu_selftest"),
        },
        "coordination_loop": {
            "triverse": {
                "turns": triverse_result.total_turns,
                "terminated_by": triverse_result.terminated_by,
            },
            "openfugu": openfugu_loop,
        },
        "compressionx_available": ContextCompressor.try_create(CoordConfig().compression) is not None,
    }

    print(json.dumps(summary, indent=2))

    out = ROOT / "benchmarks" / "last_run.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out}")

    # Overall: routing winner is OpenFugu on mock world; triverse wins on compression + pytest portability
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
