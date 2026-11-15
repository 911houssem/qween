"""Agent loop: retrieve context -> generate -> parse -> execute tools -> feed back -> repeat."""
from __future__ import annotations

import json
import logging
from typing import Callable

from ..backends.base import Backend
from ..config import Config
from ..memory.store import MemoryStore
from ..project.index import ProjectIndex
from ..rag.docs import DocsIndex
from ..tools.base import Registry
from .parser import parse
from .prompt import system_prompt

log = logging.getLogger("agent.loop")
MUTATING = {"write_file", "edit_file", "delete_file"}


class Agent:
    def __init__(self, cfg: Config, backend: Backend, registry: Registry, index: ProjectIndex,
                 memory: MemoryStore, docs: DocsIndex, on_event: Callable[[str], None] | None = None):
        self.cfg, self.backend, self.reg = cfg, backend, registry
        self.index, self.memory, self.docs = index, memory, docs
        self.emit = on_event or (lambda s: None)
        self.system = system_prompt(registry.schemas())
        self.last_changes = None

    # -- context retrieval --------------------------------------------------
    def build_context(self, task: str) -> str:
        budget = self.cfg.context_budget_chars
        parts = ["Project overview:\n" + self.index.structure()]
        if self.last_changes:
            parts.append("Files changed outside the agent since last run: " + self.last_changes.summary())
        notes = self.memory.recall(limit=8)
        if notes:
            parts.append("Project notes:\n" + "\n".join(f"- {k}: {v}" for k, v in notes))
        chunks = self.index.search_chunks(task, k=4)
        if chunks:
            parts.append("Possibly relevant code:\n" + "\n\n".join(f"{p}:{s}\n{t[:700]}" for p, s, t in chunks))
        docs = self.docs.search(task, k=2)
        if docs:
            parts.append("Relevant documentation:\n" + "\n\n".join(f"[{s}] {t[:600]}" for s, t in docs))
        ctx = "\n\n".join(parts)
        return ctx if len(ctx) <= budget else ctx[:budget] + "\n...[context truncated]"

    def _fit(self, messages: list[dict]) -> list[dict]:
        """Shrink old tool outputs until the prompt fits the context window."""
        limit = self.cfg.n_ctx - self.cfg.max_new_tokens - 64
        for i, m in enumerate(messages):
            if self.backend.count_tokens(messages) <= limit:
                break
            if m["role"] == "user" and m["content"].startswith("<tool_response>") and len(m["content"]) > 400:
                messages[i] = {"role": "user", "content": m["content"][:300] + "\n...[older output trimmed]...</tool_response>"}
        return messages

    # -- main entry ------------------------------------------------------
    def ask(self, task: str) -> str:
        self.last_changes = self.index.refresh()
        messages = [{"role": "system", "content": self.system}]
        messages += self.memory.recent_turns()
        messages.append({"role": "user", "content": f"<project_context>\n{self.build_context(task)}\n</project_context>\n\nTask: {task}"})
        self.memory.add_message("user", task)
        seen: set[str] = set()
        malformed = 0
        for step in range(1, self.cfg.max_steps + 1):
            out = self.backend.generate(self._fit(messages), self.cfg.max_new_tokens)
            log.info("step %d raw output: %s", step, out[:500])
            p = parse(out)
            if not p.calls and not p.errors:
                self.memory.add_message("assistant", p.text or out)
                return p.text or out.strip()
            messages.append({"role": "assistant", "content": out.strip()})
            if p.errors and not p.calls:
                malformed += 1
                if malformed > 2:
                    return "Stopped: the model repeatedly produced malformed tool calls. " + p.errors[0]
                messages.append({"role": "user", "content": "<tool_response>\nERROR: " + p.errors[0] +
                                 '\nUse exactly: <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>\n</tool_response>'})
                continue
            results = []
            for name, args in p.calls[: self.cfg.max_calls_per_step]:
                key = name + json.dumps(args, sort_keys=True)
                self.emit(f"-> {name} {json.dumps(args, ensure_ascii=False)[:120]}")
                if key in seen:
                    res = "ERROR: identical call already made. Use a different approach or give your final answer."
                else:
                    seen.add(key)
                    res = self.reg.call(name, args)
                    if name in MUTATING and res.startswith("OK"):
                        seen.clear()   # state changed: re-running tests/reads is legitimate again
                log.info("tool %s -> %s", name, res[:300])
                self.emit("   " + res.splitlines()[0][:140] if res else "   (no output)")
                results.append(f"[{name}]\n{res}")
            messages.append({"role": "user", "content": "<tool_response>\n" + "\n\n".join(results) + "\n</tool_response>"})
        final = f"Stopped after {self.cfg.max_steps} steps without a final answer. See .agent/agent.log for details."
        self.memory.add_message("assistant", final)
        return final
