import logging
import logging.handlers
from pathlib import Path


def setup_logging(data_dir: Path, level: str = "INFO", verbose: bool = False) -> logging.Logger:
    data_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("agent")
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)
    fh = logging.handlers.RotatingFileHandler(
        data_dir / "agent.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO if verbose else logging.WARNING)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    log.addHandler(sh)
    log.propagate = False
    return log
