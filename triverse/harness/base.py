"""Omnigent-inspired harness abstraction — swap LLM backends without rewriting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class HarnessResponse:
    content: str
    model: str
    usage: dict[str, int] | None = None


class Harness(ABC):
    """Uniform interface for any LLM backend."""

    @abstractmethod
    def complete(self, prompt: str, *, model: str, temperature: float = 0.7) -> HarnessResponse:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class MockHarness(Harness):
    """Deterministic harness for tests and demos."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or {}
        self._call_count = 0

    @property
    def name(self) -> str:
        return "mock"

    def complete(self, prompt: str, *, model: str, temperature: float = 0.7) -> HarnessResponse:
        self._call_count += 1
        key = self._detect_key(prompt)
        content = self._responses.get(key, self._default_response(prompt, key))
        return HarnessResponse(content=content, model=model)

    def _detect_key(self, prompt: str) -> str:
        upper = prompt.upper()
        if "VERIFIER" in upper:
            return "verifier"
        if "WORKER" in upper:
            return "worker"
        if "THINKER" in upper:
            return "thinker"
        return "default"

    def _default_response(self, prompt: str, key: str) -> str:
        if key == "thinker":
            return (
                "Strategy:\n"
                "1. Parse the problem constraints\n"
                "2. Apply the core formula step by step\n"
                "3. Hand off calculation to worker"
            )
        if key == "worker":
            return "Calculation: Applying the formula yields 42 as the result."
        if key == "verifier":
            if "42" in prompt:
                return "VERDICT: ACCEPT\n\nThe solution 42 is correct and complete."
            return "VERDICT: REVISE\n\nMissing final numerical answer."
        return "Acknowledged."


class OpenAIHarness(Harness):
    """OpenAI-compatible API harness."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install triverse with openai extra: pip install triverse[openai]") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai"

    def complete(self, prompt: str, *, model: str, temperature: float = 0.7) -> HarnessResponse:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        choice = response.choices[0]
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }
        return HarnessResponse(content=choice.message.content or "", model=model, usage=usage)


class AnthropicHarness(Harness):
    """Anthropic Messages API harness."""

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Install triverse with anthropic extra: pip install triverse[anthropic]"
            ) from exc
        self._client = Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        return "anthropic"

    def complete(self, prompt: str, *, model: str, temperature: float = 0.7) -> HarnessResponse:
        response = self._client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
            }
        return HarnessResponse(content=text, model=model, usage=usage)


def create_harness(kind: str, **kwargs: Any) -> Harness:
    """Factory for harness backends."""
    registry: dict[str, type[Harness]] = {
        "mock": MockHarness,
        "openai": OpenAIHarness,
        "anthropic": AnthropicHarness,
    }
    cls = registry.get(kind)
    if cls is None:
        raise ValueError(f"Unknown harness '{kind}'. Available: {list(registry)}")
    return cls(**kwargs)
