"""Optional lighter backend: llama.cpp on a GGUF *converted from* the original weights."""
from __future__ import annotations

import os

from ..config import Config
from .base import Backend


class GGUFBackend(Backend):
    name = "gguf"

    def __init__(self, cfg: Config):
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("backend 'gguf' needs: pip install llama-cpp-python") from e
        if not os.path.isfile(cfg.gguf_path):
            raise FileNotFoundError(f"gguf_path not found: {cfg.gguf_path!r}")
        self.llm = Llama(model_path=cfg.gguf_path, n_ctx=cfg.n_ctx,
                         n_threads=cfg.n_threads or os.cpu_count() or 4, verbose=False)
        self.sampling = cfg.sampling()

    def generate(self, messages, max_new_tokens):
        s = self.sampling
        r = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_new_tokens, temperature=s["temperature"],
            top_p=s["top_p"], top_k=s["top_k"], repeat_penalty=s["repetition_penalty"],
            stop=["<|im_end|>"])
        return r["choices"][0]["message"]["content"]
