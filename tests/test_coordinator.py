"""Coordinator integration tests."""

from triverse import Coordinator, CoordConfig
from triverse.pool import ModelPool
from triverse.types import Role, Verdict


def test_demo_coordination_terminates_on_accept():
    pool = ModelPool.default_demo()
    coord = Coordinator(pool, CoordConfig(max_turns=5, seed=42))
    result = coord.run("What is 6 times 7? Calculate and verify.")

    assert result.total_turns >= 1
    assert result.terminated_by == "verifier_accept"
    assert "ACCEPT" in result.answer.upper() or "42" in result.answer


def test_respects_max_turns():
    pool = ModelPool.default_demo()
    coord = Coordinator(pool, CoordConfig(max_turns=1, seed=0))
    result = coord.run("Short query")

    assert result.total_turns == 1
    assert result.terminated_by == "max_turns"


def test_turns_have_roles():
    pool = ModelPool.default_demo()
    coord = Coordinator(pool, CoordConfig(max_turns=3, seed=7))
    result = coord.run("Implement binary search and verify correctness.")

    roles = {t.role for t in result.turns}
    assert Role.THINKER in roles or Role.WORKER in roles


def test_verifier_parsing():
    from triverse.roles import parse_verifier_response

    verdict, _ = parse_verifier_response("VERDICT: ACCEPT\n\nLooks good.")
    assert verdict == Verdict.ACCEPT

    verdict, _ = parse_verifier_response("VERDICT: REVISE\n\nNeeds work.")
    assert verdict == Verdict.REVISE
