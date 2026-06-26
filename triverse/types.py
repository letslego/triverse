"""Core types for triverse coordination."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    """Trinity tri-role coordination contracts."""

    THINKER = "thinker"
    WORKER = "worker"
    VERIFIER = "verifier"


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"


class AgentSpec(BaseModel):
    """One LLM in the coordination pool."""

    id: str
    harness: str = "mock"
    model: str = "mock-model"
    strengths: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)


class CompressionConfig(BaseModel):
    """compressionX settings for triverse coordination."""

    enabled: bool = True
    model: str = "gpt-4o"
    target_ratio: float = Field(default=0.35, ge=0.05, le=1.0)
    enable_cxr: bool = True
    compress_prompt: bool = True
    compress_outputs: bool = True
    protect_recent_turns: int = 1
    min_tokens: int = 120


class CoordConfig(BaseModel):
    """Coordinator runtime configuration."""

    max_turns: int = 5
    temperature: float = 0.7
    router_temperature: float = 0.3
    seed: int | None = None
    verbose: bool = False
    compression: CompressionConfig | None = Field(default_factory=CompressionConfig)


class TurnRecord(BaseModel):
    """One coordination turn in the transcript."""

    turn: int
    agent_id: str
    role: Role
    raw_message: str
    processed_output: str
    verdict: Verdict | None = None
    router_scores: dict[str, float] = Field(default_factory=dict)


class CoordinationResult(BaseModel):
    """Final output of a coordination run."""

    query: str
    answer: str
    turns: list[TurnRecord]
    terminated_by: str
    total_turns: int
    tokens_saved: int = 0
    compressions_applied: int = 0
    compression_strategies: list[str] = Field(default_factory=list)
