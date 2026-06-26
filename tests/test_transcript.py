"""Transcript tests."""

from triverse.transcript import Transcript
from triverse.types import Role, TurnRecord, Verdict


def test_transcript_rendering():
    t = Transcript("What is 2+2?")
    t.append(
        TurnRecord(
            turn=1,
            agent_id="a",
            role=Role.WORKER,
            raw_message="4",
            processed_output="4",
        )
    )
    text = t.render_for_agent()
    assert "What is 2+2?" in text
    assert "WORKER" in text
    assert "4" in text


def test_accept_detection():
    t = Transcript("q")
    t.append(
        TurnRecord(
            turn=1,
            agent_id="v",
            role=Role.VERIFIER,
            raw_message="ok",
            processed_output="VERDICT: ACCEPT",
            verdict=Verdict.ACCEPT,
        )
    )
    assert t.has_accepted_solution()
