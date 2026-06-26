"""Multi-turn conversation state for Trinity-style coordination."""

from __future__ import annotations

from triverse.types import TurnRecord, Verdict


class Transcript:
    """Accumulates query + processed outputs across coordination turns."""

    def __init__(self, query: str) -> None:
        self.query = query
        self.turns: list[TurnRecord] = []

    def append(self, record: TurnRecord) -> None:
        self.turns.append(record)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def last_output(self) -> str | None:
        if not self.turns:
            return None
        return self.turns[-1].processed_output

    def last_verdict(self) -> Verdict | None:
        if not self.turns:
            return None
        return self.turns[-1].verdict

    def has_accepted_solution(self) -> bool:
        for record in reversed(self.turns):
            if record.role.value == "verifier" and record.verdict == Verdict.ACCEPT:
                return True
        return False

    def render_for_agent(self) -> str:
        """Full transcript passed to each selected LLM."""
        parts = [f"## Original query\n{self.query}"]
        for record in self.turns:
            parts.append(
                f"\n## Turn {record.turn} — {record.role.value.upper()} "
                f"({record.agent_id})\n{record.processed_output}"
            )
        return "\n".join(parts)

    def render_for_router(self) -> str:
        """Compact state summary for routing decisions."""
        lines = [self.query]
        for record in self.turns:
            preview = record.processed_output[:200].replace("\n", " ")
            lines.append(f"[{record.role.value}:{record.agent_id}] {preview}")
        return "\n".join(lines)
