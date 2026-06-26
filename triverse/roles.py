"""Thinker, Worker, and Verifier role contracts from Trinity."""

from __future__ import annotations

import re

from triverse.types import Role, Verdict

ROLE_SYSTEM_PROMPTS: dict[Role, str] = {
    Role.THINKER: (
        "You are the THINKER in a multi-agent coordination system. "
        "Analyze the query and transcript. Produce high-level strategy: "
        "decompose the problem, identify subgoals, note risks, and suggest "
        "what the next agent should focus on. Do not execute the full solution — "
        "provide meta-level guidance only. Be concise."
    ),
    Role.WORKER: (
        "You are the WORKER in a multi-agent coordination system. "
        "Make concrete progress toward solving the query using the transcript. "
        "Produce actionable content: derivations, code, calculations, or direct answers. "
        "Build on prior thinker plans and worker outputs."
    ),
    Role.VERIFIER: (
        "You are the VERIFIER in a multi-agent coordination system. "
        "Evaluate whether the accumulated solution is correct, complete, and "
        "responsive to the original query. "
        "Start your response with exactly one line: VERDICT: ACCEPT or VERDICT: REVISE. "
        "Then provide a brief diagnosis. If ACCEPT, include the final answer clearly."
    ),
}


def build_role_prompt(role: Role, transcript_text: str) -> str:
    system = ROLE_SYSTEM_PROMPTS[role]
    return f"{system}\n\n---\n\n{transcript_text}"


_VERDICT_RE = re.compile(r"VERDICT:\s*(ACCEPT|REVISE)", re.IGNORECASE)


def parse_verifier_response(text: str) -> tuple[Verdict, str]:
    match = _VERDICT_RE.search(text)
    verdict = Verdict.ACCEPT if match and match.group(1).upper() == "ACCEPT" else Verdict.REVISE
    return verdict, text.strip()


def post_process(role: Role, raw: str) -> tuple[str, Verdict | None]:
    """Condense agent output into transcript entry."""
    text = raw.strip()
    if role == Role.VERIFIER:
        verdict, _ = parse_verifier_response(text)
        return text, verdict
    if role == Role.THINKER:
        # Keep strategy bullets, drop excessive prose
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) > 12:
            text = "\n".join(lines[:12] + ["... (strategy truncated)"])
    return text, None
