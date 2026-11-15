"""Workspace jail: every path the agent touches is resolved and must stay inside the root."""
from __future__ import annotations

from pathlib import Path

IGNORE_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "env",
               ".idea", ".vscode", "dist", "build", ".mypy_cache", ".pytest_cache", ".agent",
               "bin", "obj", ".next", "target"}
MODEL_FILES = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "LICENSE", "README.md", "gitattributes", ".gitattributes"}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".bin", ".pt", ".pth", ".onnx", ".ckpt", ".h5"}


class SecurityError(Exception):
    """Raised when an operation would leave the workspace or touch protected files."""


class Workspace:
    def __init__(self, root: str | Path, model_dir: str | Path | None = None):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"Workspace does not exist: {self.root}")
        self.data_dir = self.root / ".agent"
        self.model_dir = Path(model_dir).resolve() if model_dir else None

    def resolve(self, path: str | Path) -> Path:
        p = Path(str(path).strip().strip('"'))
        full = (p if p.is_absolute() else self.root / p).resolve()  # resolves symlinks and ..
        try:
            full.relative_to(self.root)
        except ValueError:
            raise SecurityError(f"Path escapes workspace: {path}") from None
        return full

    def rel(self, full: Path) -> str:
        return full.relative_to(self.root).as_posix()

    def is_protected(self, full: Path) -> bool:
        if full.suffix.lower() in WEIGHT_SUFFIXES:
            return True
        if self.model_dir and full.parent == self.model_dir and full.name in MODEL_FILES:
            return True
        return False

    def check_writable(self, path: str | Path) -> Path:
        full = self.resolve(path)
        if self.is_protected(full):
            raise SecurityError(f"Protected model file, refusing to modify: {self.rel(full)}")
        try:
            full.relative_to(self.data_dir)
            raise SecurityError("The .agent/ directory is managed by the agent and is read-only to tools")
        except ValueError:
            pass
        if full.is_dir():
            raise IsADirectoryError(self.rel(full))
        return full

    def is_ignored(self, full: Path) -> bool:
        try:
            parts = full.relative_to(self.root).parts
        except ValueError:
            return True
        return any(p in IGNORE_DIRS for p in parts[:-1]) or (full.is_dir() and full.name in IGNORE_DIRS)
