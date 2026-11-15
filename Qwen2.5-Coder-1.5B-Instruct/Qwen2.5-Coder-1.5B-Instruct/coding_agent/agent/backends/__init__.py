from ..config import Config
from .base import Backend


def create_backend(cfg: Config) -> Backend:
    if cfg.backend == "hf":
        from .hf import HFBackend
        return HFBackend(cfg)
    if cfg.backend == "gguf":
        from .gguf import GGUFBackend
        return GGUFBackend(cfg)
    if cfg.backend == "mock":
        from .mock import MockBackend
        return MockBackend([])
    raise ValueError(f"Unknown backend: {cfg.backend}")
