#!/usr/bin/env python3
"""triverse + compressionX: coordinate with compressed transcripts."""

from triverse import CompressionConfig, Coordinator, CoordConfig
from triverse.pool import ModelPool


def main() -> None:
    pool = ModelPool.default_demo()
    config = CoordConfig(
        max_turns=4,
        seed=42,
        compression=CompressionConfig(
            enabled=True,
            target_ratio=0.25,
            compress_prompt=True,
            compress_outputs=True,
        ),
    )
    coord = Coordinator(pool, config)

    # Simulate a query that would accumulate bulky tool-like output across turns
    query = (
        "Review the build logs below, find the fatal error, fix it, and verify.\n\n"
        + "\n".join(f"[build] step {i}: compiling module_{i % 20}.rs ok" for i in range(150))
        + "\n[build] FATAL: linker error undefined symbol `init_cache`"
        + "\n" + "\n".join(f"[build] note: candidate {i}" for i in range(80))
    )

    result = coord.run(query)

    print(f"Answer ({result.terminated_by}):")
    print(result.answer[:400])
    print()
    print(f"Turns: {result.total_turns}")
    if result.compressions_applied:
        print(
            f"compressionX saved {result.tokens_saved} tokens "
            f"via {', '.join(result.compression_strategies)}"
        )
    else:
        print("compressionX: no compression applied (install compressionx for savings)")


if __name__ == "__main__":
    main()
