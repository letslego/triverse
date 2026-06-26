"""Trinity-style multi-turn LLM coordinator."""

from __future__ import annotations

from triverse.compression import CompressionStats, ContextCompressor
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

    When compressionX is installed, transcripts and turn outputs are compressed
    before each harness call — keeping token costs bounded across long runs.
    """

    def __init__(self, pool: ModelPool, config: CoordConfig | None = None) -> None:
        self.pool = pool
        self.config = config or CoordConfig()
        self.router = CoordinationRouter(
            list(pool.agents.values()),
            temperature=self.config.router_temperature,
            seed=self.config.seed,
        )
        self._compressor = ContextCompressor.try_create(self.config.compression)

    def run(self, query: str) -> CoordinationResult:
        transcript = Transcript(query)
        terminated_by = "max_turns"
        stats = CompressionStats()

        for turn in range(1, self.config.max_turns + 1):
            state = transcript.render_for_router()
            agent_id, role, scores = self.router.route(
                state, turn - 1, self.config.max_turns
            )

            agent = self.pool.agents[agent_id]
            harness = self.pool.get_harness(agent_id)
            transcript_text = self._render_transcript(transcript, model=agent.model, stats=stats)
            prompt = build_role_prompt(role, transcript_text)

            response = harness.complete(
                prompt,
                model=agent.model,
                temperature=self.config.temperature,
            )

            processed, verdict = post_process(role, response.content)
            if self._compressor:
                processed = self._compressor.compress_turn_output(
                    processed,
                    query=query,
                    model=agent.model,
                    stats=stats,
                )

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
            tokens_saved=stats.tokens_saved,
            compressions_applied=stats.compressions_applied,
            compression_strategies=stats.strategies,
        )

    def _render_transcript(
        self,
        transcript: Transcript,
        *,
        model: str,
        stats: CompressionStats,
    ) -> str:
        if self._compressor:
            return self._compressor.render_transcript(
                transcript.query,
                transcript.turns,
                model=model,
                stats=stats,
            )
        return transcript.render_for_agent()

    def _extract_answer(self, transcript: Transcript) -> str:
        for record in reversed(transcript.turns):
            if record.role == Role.VERIFIER and record.verdict == Verdict.ACCEPT:
                return record.processed_output
        for record in reversed(transcript.turns):
            if record.role == Role.WORKER:
                return record.processed_output
        return transcript.last_output() or ""
