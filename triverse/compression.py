"""compressionX integration — shrink transcripts and turn outputs before LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from triverse.types import CompressionConfig, TurnRecord


@dataclass
class CompressionStats:
    """Accumulated compression metrics for a coordination run."""

    tokens_saved: int = 0
    compressions_applied: int = 0
    strategies: list[str] = field(default_factory=list)

    def record(self, original: int, compressed: int, strategy: str) -> None:
        if compressed < original:
            self.tokens_saved += original - compressed
            self.compressions_applied += 1
            if strategy not in self.strategies:
                self.strategies.append(strategy)


@dataclass
class TextCompressResult:
    content: str
    original_tokens: int
    compressed_tokens: int
    strategy: str

    @property
    def was_compressed(self) -> bool:
        return self.compressed_tokens < self.original_tokens


class ContextCompressor:
    """Adapter over compressionX ContentRouter for triverse coordination."""

    def __init__(self, config: CompressionConfig) -> None:
        self.config = config
        from compressionx.transforms.content_router import ContentRouter
        from compressionx.tokenizer import count_tokens

        self._router = ContentRouter()
        self._count_tokens = count_tokens

    @classmethod
    def try_create(cls, config: CompressionConfig | None) -> ContextCompressor | None:
        if config is None or not config.enabled:
            return None
        try:
            import compressionx  # noqa: F401
        except ImportError:
            return None
        return cls(config)

    def compress_text(
        self,
        content: str,
        *,
        context: str = "",
        model: str | None = None,
        stats: CompressionStats | None = None,
    ) -> TextCompressResult:
        model = model or self.config.model
        original = self._count_tokens(content, model)
        if original < self.config.min_tokens:
            return TextCompressResult(content, original, original, "passthrough")

        result = self._router.compress(
            content,
            model=model,
            target_ratio=self.config.target_ratio,
            context=context,
            enable_cxr=self.config.enable_cxr,
        )
        compressed = result.compressed_tokens or self._count_tokens(result.content, model)
        if stats and result.was_compressed:
            stats.record(original, compressed, result.strategy)

        return TextCompressResult(
            content=result.content,
            original_tokens=original,
            compressed_tokens=compressed,
            strategy=result.strategy,
        )

    def compress_turn_output(
        self,
        text: str,
        *,
        query: str,
        model: str,
        stats: CompressionStats | None = None,
    ) -> str:
        if not self.config.compress_outputs:
            return text
        return self.compress_text(text, context=query, model=model, stats=stats).content

    def render_transcript(
        self,
        query: str,
        turns: list[TurnRecord],
        *,
        model: str,
        stats: CompressionStats | None = None,
    ) -> str:
        """Build agent-facing transcript, compressing older turns."""
        parts = [f"## Original query\n{query}"]
        protect_from = max(0, len(turns) - self.config.protect_recent_turns)

        for i, record in enumerate(turns):
            output = record.processed_output
            if self.config.compress_prompt and i < protect_from:
                output = self.compress_text(
                    output, context=query, model=model, stats=stats
                ).content
            parts.append(
                f"\n## Turn {record.turn} — {record.role.value.upper()} "
                f"({record.agent_id})\n{output}"
            )
        return "\n".join(parts)
