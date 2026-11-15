"""Syntax validation per language. Uses only what is installed; reports 'skipped' honestly."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Validation:
    status: str   # ok | error | skipped
    message: str = ""

    @property
    def failed(self) -> bool:
        return self.status == "error"

    def __str__(self):
        return f"{self.status}" + (f": {self.message}" if self.message else "")


def _run(cmd: list[str], timeout=30) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stderr or p.stdout).strip()[:1500]
    except subprocess.TimeoutExpired:
        return 1, "validator timed out"


def _tool(cmd: list[str], suffix: str, content: str, name: str) -> Validation:
    exe = shutil.which(cmd[0])
    if not exe:
        return Validation("skipped", f"{cmd[0]} not installed")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / (name + suffix)
        f.write_text(content, encoding="utf-8")
        full = [exe] + [a.replace("{f}", str(f)).replace("{d}", td) for a in cmd[1:]]
        rc, out = _run(full)
    return Validation("ok") if rc == 0 else Validation("error", out)


class _Tags(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
            "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack, self.err = [], None

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if tag in self.stack:
            while self.stack and self.stack[-1] != tag:   # allow implied closes (li, p...)
                self.stack.pop()
            self.stack.pop()
        else:
            self.err = self.err or f"unexpected closing </{tag}> at line {self.getpos()[0]}"


def _html(content: str) -> Validation:
    p = _Tags()
    p.feed(content)
    return Validation("error", p.err) if p.err else Validation("ok", "tag structure only")


def _css(content: str) -> Validation:
    s = re.sub(r"/\*.*?\*/", "", content, flags=re.S)
    s = re.sub(r"(\"[^\"]*\"|'[^']*')", "", s)
    depth = 0
    for i, ch in enumerate(s):
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return Validation("error", "unmatched '}'")
    return Validation("ok", "brace balance only") if depth == 0 else Validation("error", "unclosed '{'")


def validate(path: str, content: str) -> Validation:
    ext = Path(path).suffix.lower()
    name = Path(path).stem or "x"
    try:
        if ext in (".py", ".pyw"):
            ast.parse(content, filename=path)
            return Validation("ok")
        if ext == ".json":
            json.loads(content)
            return Validation("ok")
        if ext in (".js", ".mjs", ".cjs"):
            return _tool(["node", "--check", "{f}"], ext, content, name)
        if ext == ".php":
            return _tool(["php", "-l", "{f}"], ext, content, name)
        if ext in (".cpp", ".cc", ".cxx", ".hpp", ".h"):
            exe = "g++" if shutil.which("g++") else "clang++"
            return _tool([exe, "-fsyntax-only", "-x", "c++", "{f}"], ext, content, name)
        if ext == ".c":
            exe = "gcc" if shutil.which("gcc") else "clang"
            return _tool([exe, "-fsyntax-only", "{f}"], ext, content, name)
        if ext == ".java":
            m = re.search(r"public\s+(?:final\s+|abstract\s+)?(?:class|interface|enum|record)\s+(\w+)", content)
            return _tool(["javac", "-d", "{d}", "{f}"], ".java", content, m.group(1) if m else name)
        if ext == ".ts":
            return _tool(["tsc", "--noEmit", "--skipLibCheck", "{f}"], ext, content, name)
        if ext in (".html", ".htm"):
            return _html(content)
        if ext == ".css":
            return _css(content)
        if ext == ".cs":
            return Validation("skipped", "C# needs a project; run `dotnet build` via run_command")
        if ext == ".sql":
            return Validation("skipped", "SQL dialect unknown; not validated")
    except SyntaxError as e:
        return Validation("error", f"{e.msg} (line {e.lineno}, col {e.offset})")
    except ValueError as e:
        return Validation("error", str(e))
    return Validation("skipped", f"no validator for '{ext or 'no extension'}'")
