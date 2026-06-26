"""Trinity-style multi-turn LLM coordinator."""

from __future__ import annotations

from triverse.pool import ModelPool
from triverse.roles import build_role_prompt, post_process
from triverse.router import CoordinationRouter
from triverse.transcript import Transcript
from triverse.types import CoordConfig, CoordinationResult, Role, TurnRecord, Verdict


class Coordinator:
    """Orchestrate a pool of LLMs through Thinker → Worker → Verifier turns.

    Inspired by Trinity (arxiv:2512.04695): a lightweight coordinator delegates
    complex reasoning to diverse LLMs while enforcing a tri-role protocol.
    Harness swapping follows Omnigent's meta-harness pattern.
    """

    def __init__(self, pool: ModelPool, config: CoordConfig | None = None) -> None:
        self.pool = pool
        self.config = config or CoordConfig()
        self.router = CoordinationRouter(
            list(pool.agents.values()),
            temperature=self.config.router_temperature,
            seed=self.config.seed,
        )

    def run(self, query: str) -> CoordinationResult:
        transcript = Transcript(query)
        terminated_by = "max_turns"

        for turn in range(1, self.config.max_turns + 1):
            state = transcript.render_for_router()
            agent_id, role, scores = self.router.route(
                state, turn - 1, self.config.max_turns
            )

            agent = self.pool.agents[agent_id]
            harness = self.pool.get_harness(agent_id)
            prompt = build_role_prompt(role, transcript.render_for_agent())

            response = harness.complete(
                prompt,
                model=agent.model,
                temperature=self.config.temperature,
            )

            processed, verdict = post_process(role, response.content)
            record = TurnRecord(
                turn=turn,
                agent_id=agent_id,
                role=role,
                raw_message=response.content,
                processed_output=processed,
                verdict=verdict,
                router_scores={k: round(v, 3) for k, v in scores.items()},
            )
            transcript.append(record)

            if role == Role.VERIFIER and verdict == Verdict.ACCEPT:
                terminated_by = "verifier_accept"
                break

        answer = self._extract_answer(transcript)
        return CoordinationResult(
            query=query,
            answer=answer,
            turns=transcript.turns,
            terminated_by=terminated_by,
            total_turns=len(transcript.turns),
        )

    def _extract_answer(self, transcript: Transcript) -> str:
        # Prefer accepted verifier output
        for record in reversed(transcript.turns):
            if record.role == Role.VERIFIER and record.verdict == Verdict.ACCEPT:
                return record.processed_output
        # Fall back to last worker output
        for record in reversed(transcript.turns):
            if record.role == Role.WORKER:
                return record.processed_output
        return transcript.last_output() or ""
