"""Approval policy. Edits and commands need consent; destructive actions ALWAYS ask."""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger("agent.approvals")


class Approver:
    """prompt: callable(question)->bool. If None, anything needing consent is denied."""

    def __init__(self, prompt: Callable[[str], bool] | None = None,
                 auto_edits: bool = False, auto_commands: bool = False):
        self.prompt = prompt
        self.auto_edits = auto_edits
        self.auto_commands = auto_commands

    def confirm(self, level: str, summary: str, detail: str = "") -> bool:
        """level: edit | command | risky_command | destructive"""
        if level == "edit" and self.auto_edits:
            ok = True
        elif level == "command" and self.auto_commands:
            ok = True
        elif self.prompt is None:
            ok = False
        else:
            ok = bool(self.prompt(f"[{level.upper()}] {summary}\n{detail}\nAllow? [y/N] "))
        log.info("approval level=%s allowed=%s summary=%s", level, ok, summary)
        return ok
