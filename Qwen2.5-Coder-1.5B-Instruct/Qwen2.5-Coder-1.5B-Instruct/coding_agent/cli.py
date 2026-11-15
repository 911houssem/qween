"""Command line interface:  python cli.py --workspace <dir> <command>"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from agent.app import build_agent                      # noqa: E402
from agent.config import Config                        # noqa: E402
from agent.rag.docs import DocsIndex                   # noqa: E402
from agent.safety.approvals import Approver            # noqa: E402
from agent.safety.workspace import Workspace           # noqa: E402
from agent.verify_model import verify                  # noqa: E402


def ask_user(q: str) -> bool:
    try:
        return input(q).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="coding-agent")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--model-dir", default=str(HERE.parent), help="folder with the original Qwen files")
    ap.add_argument("--config")
    ap.add_argument("--backend", choices=["hf", "gguf", "mock"])
    ap.add_argument("--gguf-path")
    ap.add_argument("--yes-edits", action="store_true", help="auto-approve file edits (still validated + backed up)")
    ap.add_argument("--yes-commands", action="store_true", help="auto-approve ALLOWLISTED commands only")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat")
    a = sub.add_parser("ask"); a.add_argument("task")
    sub.add_parser("index")
    sub.add_parser("status")
    d = sub.add_parser("docs-add"); d.add_argument("path")
    v = sub.add_parser("verify-model"); v.add_argument("--no-tokenizer", action="store_true")
    sub.add_parser("tools")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config, workspace=args.workspace, model_dir=args.model_dir, backend=args.backend,
                      gguf_path=args.gguf_path, auto_approve_edits=args.yes_edits or None,
                      auto_approve_commands=args.yes_commands or None)

    if args.cmd == "verify-model":
        ok, rep = verify(cfg.model_dir, str(HERE / "model_manifest.json"), not args.no_tokenizer)
        print("\n".join(rep))
        print("\nRESULT:", "COMPATIBLE" if ok else "PROBLEMS FOUND")
        return 0 if ok else 1

    if args.cmd == "docs-add":
        ws = Workspace(cfg.workspace)
        n = DocsIndex(ws.data_dir).ingest(Path(args.path).resolve())
        print(f"ingested {n} chunks")
        return 0

    approver = Approver(ask_user, cfg.auto_approve_edits, cfg.auto_approve_commands)
    # index/status/tools do not need the model: use a stub backend to avoid loading 3 GB
    light = args.cmd in ("index", "status", "tools")
    if light:
        from agent.backends.mock import MockBackend
        agent = build_agent(cfg, approver, backend=MockBackend([]), verbose=args.verbose)
    else:
        try:
            agent = build_agent(cfg, approver, on_event=print, verbose=args.verbose)
        except (RuntimeError, FileNotFoundError, ValueError) as e:
            print(f"Cannot start the model backend ({cfg.backend}):\n{e}", file=sys.stderr)
            return 2

    if args.cmd == "index":
        cs = agent.index.refresh()
        print(agent.index.structure()); print("changes:", cs.summary())
    elif args.cmd == "status":
        cs = agent.index.refresh()
        print(agent.index.structure()); print("changes since last scan:", cs.summary())
        notes = agent.memory.recall()
        print("notes:", "; ".join(f"{k}={v}" for k, v in notes) or "none")
    elif args.cmd == "tools":
        for t in agent.reg.tools.values():
            print(f"{t.name:18} {t.description}")
    elif args.cmd == "ask":
        print(agent.ask(args.task))
    elif args.cmd == "chat":
        print("Coding agent ready. /exit to quit, /changes to rescan.")
        while True:
            try:
                task = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task in ("/exit", "/quit"):
                break
            if task == "/changes":
                print(agent.index.refresh().summary()); continue
            if task:
                print(agent.ask(task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
