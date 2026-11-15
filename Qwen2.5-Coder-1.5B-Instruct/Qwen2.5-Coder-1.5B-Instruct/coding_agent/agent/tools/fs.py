"""File tools: list, read, write, edit (search/replace), delete (to trash), search."""
from __future__ import annotations

import difflib
import os
import re
import shutil
import time
from pathlib import Path

from ..project.deps import unverified_imports
from ..safety.workspace import IGNORE_DIRS
from ..validation.validators import validate
from .base import ToolContext


def _read_text(ctx: ToolContext, full: Path) -> str:
    if not full.is_file():
        raise FileNotFoundError(f"not a file: {ctx.ws.rel(full)}")
    if full.stat().st_size > ctx.cfg.max_file_bytes:
        raise ValueError(f"file too large (> {ctx.cfg.max_file_bytes} bytes); use start_line/end_line on a smaller file")
    data = full.read_bytes()
    if b"\x00" in data[:4096]:
        raise ValueError("binary file")
    return data.decode("utf-8", errors="replace")


def list_dir(ctx: ToolContext, path: str = ".", depth: int = 1) -> str:
    base = ctx.ws.resolve(path)
    if not base.is_dir():
        return f"ERROR: not a directory: {path}"
    depth = max(1, min(depth, 4))
    out, count = [], 0
    for dp, dns, fns in os.walk(base):
        rel_depth = len(Path(dp).relative_to(base).parts)
        dns[:] = sorted(d for d in dns if d not in IGNORE_DIRS)
        if rel_depth >= depth:
            dns[:] = []
        for d in dns:
            out.append(f"{ctx.ws.rel(Path(dp) / d)}/")
        for f in sorted(fns):
            out.append(ctx.ws.rel(Path(dp) / f))
        count = len(out)
        if count >= 200:
            out.append("... (truncated at 200 entries)")
            break
    return "\n".join(out) or "(empty)"


def read_file(ctx: ToolContext, path: str, start_line: int = 1, end_line: int = 0) -> str:
    full = ctx.ws.resolve(path)
    lines = _read_text(ctx, full).splitlines()
    start = max(1, start_line)
    end = end_line if end_line > 0 else min(len(lines), start + 399)
    end = min(end, len(lines), start + 399)
    body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
    tail = f"\n[showing {start}-{end} of {len(lines)} lines]" if (start > 1 or end < len(lines)) else ""
    return body + tail if body else "(empty file)"


def _backup(ctx: ToolContext, full: Path):
    if full.exists():
        d = ctx.ws.data_dir / "backups"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full, d / f"{time.strftime('%Y%m%d-%H%M%S')}_{full.name}")


def _atomic_write(full: Path, text: str):
    full.parent.mkdir(parents=True, exist_ok=True)
    tmp = full.with_name(full.name + ".agent.tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, full)


def _diff(rel: str, old: str, new: str, limit=60) -> str:
    d = list(difflib.unified_diff(old.splitlines(), new.splitlines(), f"a/{rel}", f"b/{rel}", lineterm="", n=2))
    return "\n".join(d[:limit]) + (f"\n... ({len(d) - limit} more diff lines)" if len(d) > limit else "")


def _commit(ctx: ToolContext, full: Path, old: str | None, new: str, verb: str) -> str:
    rel = ctx.ws.rel(full)
    v_new = validate(rel, new)
    if v_new.failed:
        v_old = validate(rel, old) if old is not None else None
        if v_old is None or not v_old.failed:
            return f"REJECTED (not written): the resulting {rel} fails validation -> {v_new.message}\nFix the code and try again."
    diff = _diff(rel, old or "", new)
    if not ctx.approver.confirm("edit", f"{verb} {rel}", diff):
        return f"DENIED by user: {verb} {rel} was not applied."
    _backup(ctx, full)
    _atomic_write(full, new)
    ctx.index.update_file(full)
    msg = [f"OK: {verb} {rel}", f"validation: {v_new}"]
    bad = unverified_imports(ctx.ws.root, rel, new)
    if bad:
        msg.append("WARNING: imports not declared in any manifest and not installed: " + ", ".join(bad)
                   + ". Verify with check_dependency before relying on them.")
    return "\n".join(msg)


def write_file(ctx: ToolContext, path: str, content: str) -> str:
    full = ctx.ws.check_writable(path)
    old = _read_text(ctx, full) if full.exists() else None
    return _commit(ctx, full, old, content, "overwrote" if old is not None else "created")


def edit_file(ctx: ToolContext, path: str, search: str, replace: str) -> str:
    full = ctx.ws.check_writable(path)
    old = _read_text(ctx, full)
    if not search:
        return "ERROR: 'search' must be a non-empty exact snippet from the file"
    n = old.count(search)
    if n == 0:
        first = search.strip().splitlines()[0] if search.strip() else search
        close = difflib.get_close_matches(first.strip(), [l.strip() for l in old.splitlines()], n=2, cutoff=0.6)
        hint = f" Closest lines: {close}" if close else ""
        return f"ERROR: 'search' text not found in {path} (must match exactly, including whitespace).{hint} Use read_file first."
    if n > 1:
        return f"ERROR: 'search' matches {n} places in {path}; include more surrounding lines to make it unique."
    return _commit(ctx, full, old, old.replace(search, replace, 1), "edited")


def delete_file(ctx: ToolContext, path: str) -> str:
    full = ctx.ws.check_writable(path)
    if not full.exists():
        return f"ERROR: no such file: {path}"
    rel = ctx.ws.rel(full)
    if not ctx.approver.confirm("destructive", f"DELETE {rel}", "(file is moved to .agent/trash, not erased)"):
        return f"DENIED by user: {rel} was not deleted."
    trash = ctx.ws.data_dir / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    shutil.move(str(full), trash / f"{time.strftime('%Y%m%d-%H%M%S')}_{full.name}")
    ctx.index.update_file(full)
    return f"OK: moved {rel} to .agent/trash/"


def search_code(ctx: ToolContext, pattern: str, glob: str = "*", max_results: int = 40) -> str:
    try:
        rx = re.compile(pattern, re.I)
    except re.error:
        rx = re.compile(re.escape(pattern), re.I)
    out = []
    for full in ctx.index.iter_files():
        if not Path(full.name).match(glob):
            continue
        try:
            data = full.read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:4096]:
            continue
        for i, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if rx.search(line):
                out.append(f"{ctx.ws.rel(full)}:{i}: {line.strip()[:160]}")
                if len(out) >= max_results:
                    return "\n".join(out) + "\n... (max_results reached)"
    return "\n".join(out) or "no matches"
