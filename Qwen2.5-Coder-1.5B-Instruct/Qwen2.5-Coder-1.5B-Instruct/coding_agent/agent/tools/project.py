"""Project-intelligence tools: summary, symbols, dependencies, docs RAG, memory."""
from __future__ import annotations

from ..project.deps import check_dependency as _check_dep
from .base import ToolContext


def project_summary(ctx: ToolContext) -> str:
    return ctx.index.structure()


def find_symbol(ctx: ToolContext, name: str) -> str:
    rows = ctx.index.find_symbol(name)
    return "\n".join(f"{n} ({k}) {p}:{l}" for n, k, p, l in rows) or f"no symbol matching '{name}'"


def check_dependency(ctx: ToolContext, name: str) -> str:
    return _check_dep(ctx.ws.root, name)


def search_docs(ctx: ToolContext, query: str) -> str:
    hits = ctx.docs.search(query)
    if not hits:
        return "no documentation matches. (Docs are only what the user ingested with `docs-add`; do not invent APIs.)"
    return "\n\n".join(f"[{s}]\n{t}" for s, t in hits)


def remember(ctx: ToolContext, key: str, value: str) -> str:
    ctx.memory.remember(key, value)
    return f"OK: remembered '{key}'"


def recall(ctx: ToolContext, query: str = "") -> str:
    rows = ctx.memory.recall(query)
    return "\n".join(f"- {k}: {v}" for k, v in rows) or "no project notes"
