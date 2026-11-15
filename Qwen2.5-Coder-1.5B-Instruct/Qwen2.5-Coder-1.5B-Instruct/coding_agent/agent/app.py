"""Wiring: builds all components for a workspace."""
from __future__ import annotations

from pathlib import Path

from .backends import create_backend
from .backends.base import Backend
from .config import Config
from .core.loop import Agent
from .logging_setup import setup_logging
from .memory.store import MemoryStore
from .project.index import ProjectIndex
from .rag.docs import DocsIndex
from .safety.approvals import Approver
from .safety.workspace import Workspace
from .tools import ToolContext, build_registry


def build_agent(cfg: Config, approver: Approver, backend: Backend | None = None, on_event=None, verbose=False) -> Agent:
    ws = Workspace(cfg.workspace, cfg.model_dir or None)
    setup_logging(ws.data_dir, cfg.log_level, verbose)
    index = ProjectIndex(ws, cfg)
    memory = MemoryStore(ws.data_dir)
    docs = DocsIndex(ws.data_dir)
    for d in cfg.docs_dirs:
        p = ws.root / d
        if p.is_dir():
            docs.ingest(p)
    registry = build_registry(ToolContext(cfg, ws, approver, index, memory, docs))
    return Agent(cfg, backend or create_backend(cfg), registry, index, memory, docs, on_event)
