#!/usr/bin/env python3
"""Basic triverse coordination example."""

from triverse import Coordinator, CoordConfig
from triverse.pool import ModelPool


def main() -> None:
    pool = ModelPool.default_demo()
    coord = Coordinator(pool, CoordConfig(max_turns=4, seed=42))

    query = (
        "A machine depreciates from $10,000 at 20% per year for 3 years. "
        "Calculate the final value and verify the result."
    )
    result = coord.run(query)

    print(f"Query: {result.query}\n")
    for turn in result.turns:
        print(f"Turn {turn.turn} [{turn.role.value}] via {turn.agent_id}:")
        print(turn.processed_output[:200])
        print()

    print(f"Answer ({result.terminated_by}):")
    print(result.answer)


if __name__ == "__main__":
    main()
