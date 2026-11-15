from __future__ import annotations

from abc import ABC, abstractmethod


class Backend(ABC):
    name = "base"

    @abstractmethod
    def generate(self, messages: list[dict], max_new_tokens: int) -> str:
        """messages: ChatML-style [{'role','content'}]. Returns the assistant text only."""

    def count_tokens(self, messages: list[dict]) -> int:
        # Conservative approximation (code tokenizes at roughly 3-4 chars/token).
        return int(sum(len(m["content"]) for m in messages) / 3.2) + 8 * len(messages)
