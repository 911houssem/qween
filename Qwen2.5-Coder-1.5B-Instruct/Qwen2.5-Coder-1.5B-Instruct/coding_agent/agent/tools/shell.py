"""Command execution. shell=False, allowlist, workspace-confined args, timeout.

Allowlisting a command does not make project code safe: interpreters, test runners and
package scripts are always treated as risky and require an interactive approval.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from ..safety.workspace import SecurityError
from .base import ToolContext

ALLOWED = {
    "python": None, "python3": None, "py": None, "pytest": None, "node": None, "php": None,
    "tsc": None, "javac": None, "java": None, "g++": None, "gcc": None, "clang++": None,
    "git": {"status", "diff", "log", "show", "branch", "ls-files"},
    "pip": {"list", "show", "freeze", "check"},
    "npm": {"test", "run", "ls", "list"},
    "npx": {"tsc", "jest", "eslint", "vitest"},
    "dotnet": {"build", "test", "run"},
}
# These commands execute project-controlled code or scripts. They must never be
# auto-approved by --yes-commands, even when their executable is allowlisted.
CODE_EXECUTORS = {
    "python", "python3", "py", "pytest", "node", "php", "java",
}
SCRIPT_SUBCOMMANDS = {
    "npm": {"test", "run"},
    "npx": {"jest", "eslint", "vitest"},
    "dotnet": {"test", "run"},
}
DENIED = {"rm", "rmdir", "del", "erase", "rd", "format", "shutdown", "restart", "mkfs", "dd", "diskpart",
          "reg", "takeown", "icacls", "chmod", "chown", "sudo", "runas", "taskkill", "kill", "cmd",
          "powershell", "pwsh", "bash", "sh", "wsl", "curl", "wget", "ssh", "scp", "net", "sc"}
META = re.compile(r"[|&;<>\x60$]|\\$\\(")


def _looks_like_path(a: str) -> bool:
    return os.path.isabs(a) or bool(re.match(r"^[A-Za-z]:", a)) or ".." in re.split(r"[\\/]", a)


def parse_command(ctx: ToolContext, command: str) -> tuple[list[str], str]:
    """Return argv and risk level. 'risky_command' always needs interactive approval."""
    if META.search(command):
        raise SecurityError("shell operators (| & ; < > \\x60 $) are not allowed; run one plain command")
    try:
        argv = shlex.split(command, posix=(os.name != "nt"))
    except ValueError as e:
        raise SecurityError(f"cannot parse command: {e}") from None
    if not argv:
        raise SecurityError("empty command")
    exe = Path(argv[0]).name.lower().removesuffix(".exe").removesuffix(".cmd")
    if exe in DENIED:
        raise SecurityError(f"command '{exe}' is blocked for safety")
    for a in argv[1:]:
        if not a.startswith("-") and _looks_like_path(a):
            ctx.ws.resolve(a)
        elif a.startswith("--") and "=" in a and _looks_like_path(a.split("=", 1)[1]):
            ctx.ws.resolve(a.split("=", 1)[1])

    risk = "risky_command"
    if exe in ALLOWED:
        sub = ALLOWED[exe]
        first = next((a for a in argv[1:] if not a.startswith("-")), None)
        if sub is None or first in sub:
            risk = "command"

    # Never let the broad --yes-commands switch silently execute arbitrary
    # project code via interpreters, test runners, or package lifecycle scripts.
    if exe in CODE_EXECUTORS or first in SCRIPT_SUBCOMMANDS.get(exe, set()):
        risk = "risky_command"
    return argv, risk


def _execute(ctx: ToolContext, argv: list[str], risk: str) -> str:
    exe = Path(argv[0]).name.lower().removesuffix(".exe")
    if exe in ("python", "python3", "py"):
        argv = [sys.executable] + argv[1:]
    else:
        resolved = shutil.which(argv[0])
        if not resolved:
            return f"ERROR: '{argv[0]}' not found on PATH"
        argv = [resolved] + argv[1:]
    shown = " ".join(shlex.quote(a) for a in argv)
    warn = ("Project-controlled code will execute with your operating-system permissions. "
            "This is NOT a sandbox; inspect the command and project first.") if risk == "risky_command" else ""
    if not ctx.approver.confirm(risk, f"run: {shown}", warn):
        return "DENIED by user: command was not run."
    env = {k: v for k, v in os.environ.items() if not re.search(r"(KEY|TOKEN|SECRET|PASSWORD)", k, re.I)}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        p = subprocess.run(argv, cwd=ctx.ws.root, capture_output=True, text=True, errors="replace",
                           timeout=ctx.cfg.command_timeout, env=env, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return f"ERROR: timed out after {ctx.cfg.command_timeout}s (process killed)"
    out = (p.stdout or "") + (("\\n[stderr]\\n" + p.stderr) if p.stderr.strip() else "")
    if len(out) > 5000:
        out = out[:1500] + "\\n...[output trimmed]...\\n" + out[-3000:]
    return f"exit code: {p.returncode}\\n{out.strip()}"


def run_command(ctx: ToolContext, command: str) -> str:
    argv, risk = parse_command(ctx, command)
    return _execute(ctx, argv, risk)


RUNNERS = {".py": "python", ".js": "node", ".mjs": "node", ".php": "php"}


def run_file(ctx: ToolContext, path: str, args: str = "") -> str:
    full = ctx.ws.resolve(path)
    if not full.is_file():
        return f"ERROR: no such file: {path}"
    runner = RUNNERS.get(full.suffix.lower())
    if not runner:
        return f"ERROR: don't know how to run '{full.suffix}' files directly; use run_command (e.g. javac/g++/dotnet)."
    return run_command(ctx, f"{runner} {ctx.ws.rel(full)} {args}".strip())


def run_tests(ctx: ToolContext, command: str = "") -> str:
    if command:
        return run_command(ctx, command)
    root = ctx.ws.root
    if (root / "package.json").is_file() and '"test"' in (root / "package.json").read_text(encoding="utf-8", errors="ignore"):
        return run_command(ctx, "npm test")
    if any(root.glob("*.csproj")) or any(root.glob("*.sln")):
        return run_command(ctx, "dotnet test")
    has_py = any(root.rglob("test_*.py")) or any(root.rglob("*_test.py")) or (root / "tests").is_dir()
    if has_py:
        try:
            import pytest  # noqa: F401
            return run_command(ctx, "python -m pytest -x -q")
        except ImportError:
            start = "-s tests " if (root / "tests").is_dir() else ""
            return run_command(ctx, f"python -m unittest discover {start}-v")
    return "ERROR: could not detect a test runner. Pass `command`, e.g. 'python -m unittest' or 'npm test'."
