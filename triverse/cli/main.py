"""triverse CLI."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triverse.coordinator import Coordinator
from triverse.pool import ModelPool
from triverse.types import CoordConfig

console = Console()


@click.group()
def cli() -> None:
    """triverse — lightweight multi-LLM coordinator."""


@cli.command()
def demo() -> None:
    """Run a demo coordination with mock harnesses."""
    pool = ModelPool.default_demo()
    coord = Coordinator(pool, CoordConfig(verbose=True, max_turns=3))
    result = coord.run("Calculate the answer to life: apply the formula and verify.")

    console.print(Panel(result.answer, title="Final answer", border_style="green"))
    _print_turns(result.turns)
    console.print(f"\nTerminated by: [bold]{result.terminated_by}[/] ({result.total_turns} turns)")


@cli.command()
@click.argument("query")
@click.option("--pool", "pool_path", type=click.Path(exists=True), help="YAML agent pool config")
@click.option("--max-turns", default=5, show_default=True)
@click.option("--json-out", is_flag=True, help="Emit JSON result")
def run(query: str, pool_path: str | None, max_turns: int, json_out: bool) -> None:
    """Coordinate a query across the agent pool."""
    pool = ModelPool.from_yaml(pool_path) if pool_path else ModelPool.default_demo()
    coord = Coordinator(pool, CoordConfig(max_turns=max_turns))
    result = coord.run(query)

    if json_out:
        click.echo(result.model_dump_json(indent=2))
        return

    console.print(Panel(result.answer, title="Answer"))
    _print_turns(result.turns)


@cli.command()
@click.argument("output", type=click.Path())
def init_pool(output: str) -> None:
    """Write a starter agent pool YAML."""
    template = """# triverse agent pool — swap harnesses without rewriting coordination logic
agents:
  - id: gpt
    harness: openai
    model: gpt-4o
    strengths: [reasoning, coding]
    config: {}

  - id: claude
    harness: anthropic
    model: claude-sonnet-4-20250514
    strengths: [reasoning, verification]
    config: {}

  - id: fast
    harness: openai
    model: gpt-4o-mini
    strengths: [knowledge]
    config: {}
"""
    Path(output).write_text(template)
    console.print(f"Wrote pool config to [bold]{output}[/]")


def _print_turns(turns: list) -> None:
    table = Table(title="Coordination turns")
    table.add_column("Turn", style="cyan")
    table.add_column("Agent")
    table.add_column("Role")
    table.add_column("Output preview")

    for record in turns:
        preview = record.processed_output[:80].replace("\n", " ")
        table.add_row(str(record.turn), record.agent_id, record.role.value, preview + "…")

    console.print(table)
