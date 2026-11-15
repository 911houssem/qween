"""Scripted backend for tests/offline demos. It does NOT simulate model quality."""
from __future__ import annotations

from .base import Backend


class MockBackend(Backend):
    name = "mock"

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def generate(self, messages, max_new_tokens):
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            return "Done."
        return self.script.pop(0)
