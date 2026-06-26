"""LLM pool management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from triverse.harness import Harness, create_harness
from triverse.types import AgentSpec


class ModelPool:
    """Pool of agents, each bound to a swappable harness."""

    def __init__(self, agents: list[AgentSpec], harnesses: dict[str, Harness] | None = None) -> None:
        self.agents = {a.id: a for a in agents}
        self._harnesses = harnesses or {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelPool:
        data = yaml.safe_load(Path(path).read_text())
        agents = [AgentSpec(**spec) for spec in data.get("agents", [])]
        pool = cls(agents)
        for agent in agents:
            if agent.harness not in pool._harnesses:
                pool._harnesses[agent.harness] = create_harness(agent.harness, **agent.config)
        return pool

    @classmethod
    def default_demo(cls) -> ModelPool:
        return cls(
            [
                AgentSpec(id="fast", harness="mock", model="mock-fast", strengths=["reasoning"]),
                AgentSpec(id="code", harness="mock", model="mock-code", strengths=["coding"]),
                AgentSpec(id="verify", harness="mock", model="mock-verify", strengths=["verification"]),
            ],
            harnesses={"mock": create_harness("mock")},
        )

    def agent_ids(self) -> list[str]:
        return list(self.agents.keys())

    def get_harness(self, agent_id: str) -> Harness:
        agent = self.agents[agent_id]
        if agent.harness not in self._harnesses:
            self._harnesses[agent.harness] = create_harness(agent.harness, **agent.config)
        return self._harnesses[agent.harness]

    def register_harness(self, name: str, harness: Harness) -> None:
        self._harnesses[name] = harness

    def to_dict(self) -> dict[str, Any]:
        return {"agents": [a.model_dump() for a in self.agents.values()]}
