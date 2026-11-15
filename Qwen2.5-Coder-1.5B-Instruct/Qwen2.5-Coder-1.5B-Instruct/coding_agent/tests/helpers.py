import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.config import Config                     # noqa: E402
from agent.memory.store import MemoryStore          # noqa: E402
from agent.project.index import ProjectIndex        # noqa: E402
from agent.rag.docs import DocsIndex                # noqa: E402
from agent.safety.approvals import Approver         # noqa: E402
from agent.safety.workspace import Workspace        # noqa: E402
from agent.tools import ToolContext, build_registry # noqa: E402

MODEL_DIR = ROOT.parent


def make_env(files: dict[str, str] | None = None, approve=True, **cfg_kw):
    td = tempfile.TemporaryDirectory()
    root = Path(td.name) / "ws"
    root.mkdir()
    for rel, content in (files or {}).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    cfg = Config(workspace=str(root), model_dir=str(MODEL_DIR), backend="mock", **cfg_kw)
    ws = Workspace(root, MODEL_DIR)
    index = ProjectIndex(ws, cfg)
    index.refresh()
    ctx = ToolContext(cfg, ws, Approver(lambda q: approve), index, MemoryStore(ws.data_dir), DocsIndex(ws.data_dir))
    return td, root, ctx, build_registry(ctx)
