"""Lightweight coordination head — practical substitute for Trinity's CMA-ES-trained head.

Trinity uses hidden states from a 0.6B SLM + a 10K-parameter linear head optimized
with sep-CMA-ES. triverse exposes the same *decision surface* (agent × role) with a
transparent feature-based scorer that can be tuned from YAML weights or replaced
with a learned head later.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field

from triverse.types import AgentSpec, Role

# Task-domain keyword signals (mirrors Trinity's contextual routing without ML infra)
DOMAIN_SIGNALS: dict[str, list[str]] = {
    "math": ["calculate", "equation", "integral", "derivative", "proof", "theorem", "algebra"],
    "code": ["function", "implement", "debug", "python", "rust", "code", "algorithm", "api"],
    "reasoning": ["explain", "why", "compare", "analyze", "evaluate", "reason"],
    "knowledge": ["what is", "define", "history", "who", "when", "fact"],
}

STRENGTH_TO_DOMAIN: dict[str, str] = {
    "coding": "code",
    "math": "math",
    "reasoning": "reasoning",
    "verification": "reasoning",
    "knowledge": "knowledge",
}


@dataclass
class RouterWeights:
    """Tunable coordination head parameters (< 20 scalar weights, Trinity-scale)."""

    role_phase_bias: dict[str, float] = field(
        default_factory=lambda: {
            "thinker_early": 2.5,
            "worker_mid": 2.0,
            "verifier_late": 2.5,
        }
    )
    domain_agent_boost: float = 1.5
    repeat_agent_penalty: float = 0.8
    force_verify_after_worker: float = 1.2


class CoordinationRouter:
    """Select (agent_id, role) given transcript state."""

    def __init__(
        self,
        agents: list[AgentSpec],
        weights: RouterWeights | None = None,
        *,
        temperature: float = 0.3,
        seed: int | None = None,
    ) -> None:
        self.agents = agents
        self.weights = weights or RouterWeights()
        self.temperature = max(temperature, 1e-6)
        self._rng = random.Random(seed)

    def route(self, state_text: str, turn_index: int, max_turns: int) -> tuple[str, Role, dict[str, float]]:
        scores: dict[str, float] = {}
        domains = self._detect_domains(state_text)

        for agent in self.agents:
            for role in Role:
                key = f"{agent.id}:{role.value}"
                scores[key] = self._score(agent, role, domains, turn_index, max_turns, state_text)

        chosen_key = self._sample(scores)
        agent_id, role_str = chosen_key.rsplit(":", 1)
        return agent_id, Role(role_str), scores

    def _detect_domains(self, text: str) -> set[str]:
        lower = text.lower()
        found: set[str] = set()
        for domain, keywords in DOMAIN_SIGNALS.items():
            if any(kw in lower for kw in keywords):
                found.add(domain)
        return found or {"reasoning"}

    def _score(
        self,
        agent: AgentSpec,
        role: Role,
        domains: set[str],
        turn_index: int,
        max_turns: int,
        state_text: str,
    ) -> float:
        score = 0.0
        phase = turn_index / max(max_turns - 1, 1)

        # Phase-appropriate role bias (Trinity's adaptive T→W→V flow)
        if role == Role.THINKER and phase < 0.35:
            score += self.weights.role_phase_bias["thinker_early"]
        elif role == Role.WORKER and 0.2 <= phase <= 0.75:
            score += self.weights.role_phase_bias["worker_mid"]
        elif role == Role.VERIFIER and phase > 0.5:
            score += self.weights.role_phase_bias["verifier_late"]

        # Agent-domain affinity
        for strength in agent.strengths:
            domain = STRENGTH_TO_DOMAIN.get(strength, strength)
            if domain in domains:
                score += self.weights.domain_agent_boost

        # Verifier specialization
        if role == Role.VERIFIER and "verification" in agent.strengths:
            score += 0.5

        # Coding tasks → code-strength agents as workers
        if role == Role.WORKER and "code" in domains and "coding" in agent.strengths:
            score += 1.0

        # Penalize repeating the same agent (encourage diversity like Trinity's pool routing)
        repeat = len(re.findall(rf"\[{role.value}:{re.escape(agent.id)}\]", state_text))
        if repeat > 0:
            score -= self.weights.repeat_agent_penalty * repeat

        # Nudge verifier on final turns if worker output exists
        if role == Role.VERIFIER and "worker:" in state_text.lower():
            score += self.weights.force_verify_after_worker

        # Small noise for exploration
        score += self._rng.gauss(0, 0.05)
        return score

    def _sample(self, scores: dict[str, float]) -> str:
        if not scores:
            raise ValueError("No routing candidates")
        if self.temperature < 1e-3:
            return max(scores, key=scores.get)  # type: ignore[arg-type]

        keys = list(scores.keys())
        logits = [scores[k] / self.temperature for k in keys]
        max_logit = max(logits)
        exp = [math.exp(v - max_logit) for v in logits]
        total = sum(exp)
        r = self._rng.random() * total
        cumulative = 0.0
        for key, weight in zip(keys, exp):
            cumulative += weight
            if r <= cumulative:
                return key
        return keys[-1]
