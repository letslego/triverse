"""Router unit tests."""

from triverse.pool import ModelPool
from triverse.router import CoordinationRouter
from triverse.types import Role


def test_router_prefers_thinker_early():
    pool = ModelPool.default_demo()
    router = CoordinationRouter(list(pool.agents.values()), temperature=0.01, seed=1)
    _, role, scores = router.route("Implement a Python API endpoint", turn_index=0, max_turns=5)
    thinker_scores = [v for k, v in scores.items() if k.endswith(":thinker")]
    worker_scores = [v for k, v in scores.items() if k.endswith(":worker")]
    assert max(thinker_scores) >= min(worker_scores)


def test_router_prefers_verifier_late():
    pool = ModelPool.default_demo()
    router = CoordinationRouter(list(pool.agents.values()), temperature=0.01, seed=1)
    state = "query\n[worker:code] result is 42"
    _, role, _ = router.route(state, turn_index=4, max_turns=5)
    assert role == Role.VERIFIER
