"""Configuration: dataclass defaults, optional JSON file, CLI overrides."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class Config:
    workspace: str = "."
    model_dir: str = ""            # directory holding the ORIGINAL model files (read-only use)
    backend: str = "hf"            # hf | gguf | mock
    gguf_path: str = ""            # only for backend=gguf (converted from the original weights)
    dtype: str = "bfloat16"        # bf16 ~3.1 GB RAM; float32 would need ~6.2 GB -> too much for 8 GB
    n_threads: int = 0             # 0 = all cores
    n_ctx: int = 8192              # KV cache at 8k tokens ~ 235 MB (see README); model supports 32768
    max_new_tokens: int = 768
    # Sampling. None -> take the value from the model's own generation_config.json.
    temperature: float | None = 0.2   # agent-tuned (more deterministic tool calls); NOT benchmarked
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    context_budget_chars: int = 12000
    max_steps: int = 8
    max_calls_per_step: int = 2
    max_file_bytes: int = 200_000
    max_index_files: int = 5000
    max_tool_output_chars: int = 6000
    command_timeout: int = 90
    auto_approve_edits: bool = False
    auto_approve_commands: bool = False
    docs_dirs: list = field(default_factory=lambda: ["docs"])
    log_level: str = "INFO"

    @classmethod
    def load(cls, path: str | None = None, **overrides) -> "Config":
        data: dict = {}
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def sampling(self) -> dict:
        """Merge model generation_config.json (original defaults) with overrides."""
        s = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.1}
        gc = Path(self.model_dir) / "generation_config.json" if self.model_dir else None
        if gc and gc.is_file():
            try:
                g = json.loads(gc.read_text(encoding="utf-8"))
                for k in s:
                    if k in g:
                        s[k] = g[k]
            except (OSError, ValueError):
                pass
        for k in s:
            v = getattr(self, k)
            if v is not None:
                s[k] = v
        return s
