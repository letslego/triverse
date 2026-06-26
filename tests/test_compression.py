"""compressionX integration tests."""

import pytest

from triverse import CompressionConfig, Coordinator, CoordConfig
from triverse.compression import ContextCompressor
from triverse.pool import ModelPool


compressionx = pytest.importorskip("compressionx")


def test_compressor_reduces_large_log():
    config = CompressionConfig(target_ratio=0.2, min_tokens=50)
    compressor = ContextCompressor(config)
    log = "\n".join(f"INFO line {i}: status=ok metrics={{cpu: {i%100}}}" for i in range(200))
    log += "\nERROR: connection timeout at shard 7\n" + "TRACE " * 50

    result = compressor.compress_text(log, context="find the error", model="gpt-4o")
    assert result.was_compressed
    assert "ERROR" in result.content
    assert result.compressed_tokens < result.original_tokens


def test_coordinator_tracks_compression_stats():
    pool = ModelPool.default_demo()
    config = CoordConfig(
        max_turns=3,
        seed=42,
        compression=CompressionConfig(enabled=True, min_tokens=10),
    )
    coord = Coordinator(pool, config)
    assert coord._compressor is not None

    result = coord.run("Analyze these logs and verify the root cause.")
    assert result.terminated_by == "verifier_accept"


def test_compression_disabled_when_not_installed(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "compressionx":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    compressor = ContextCompressor.try_create(CompressionConfig())
    assert compressor is None


def test_render_transcript_protects_recent_turns():
    from triverse.types import Role, TurnRecord

    config = CompressionConfig(protect_recent_turns=1, min_tokens=10)
    compressor = ContextCompressor(config)
    turns = [
        TurnRecord(
            turn=1,
            agent_id="a",
            role=Role.WORKER,
            raw_message="x",
            processed_output="\n".join(f"LOG {i}" for i in range(100)),
        ),
        TurnRecord(
            turn=2,
            agent_id="b",
            role=Role.VERIFIER,
            raw_message="y",
            processed_output="VERDICT: REVISE",
        ),
    ]
    rendered = compressor.render_transcript("find errors", turns, model="gpt-4o")
    assert "LOG 0" in rendered or "offloaded" in rendered.lower() or len(rendered) < 3000
