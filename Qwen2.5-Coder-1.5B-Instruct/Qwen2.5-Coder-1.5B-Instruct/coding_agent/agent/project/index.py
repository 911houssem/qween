"""Project index: file inventory, symbols, FTS5 chunks, change detection. SQLite, no services."""
from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..safety.workspace import IGNORE_DIRS, Workspace
from .deps import parse_manifests

log = logging.getLogger("agent.index")

LANGS = {".py": "Python", ".pyw": "Python", ".js": "JavaScript", ".mjs": "JavaScript",
         ".cjs": "JavaScript", ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
         ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".hpp": "C++", ".h": "C/C++", ".c": "C",
         ".java": "Java", ".php": "PHP", ".sql": "SQL", ".html": "HTML", ".htm": "HTML",
         ".css": "CSS", ".cs": "C#", ".json": "JSON", ".md": "Markdown", ".txt": "Text",
         ".toml": "TOML", ".yml": "YAML", ".yaml": "YAML", ".xml": "XML", ".csproj": "C# project"}
CHUNK_LINES = 40

_GENERIC_SYMS = [
    (re.compile(r"^\s*(?:(?:export|default|abstract|public|private|protected|internal|static|final|sealed|partial)\s+)*(class|interface|enum|struct|namespace|trait|record)\s+(\w+)"), None),
    (re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s*\*?\s*(\w+)\s*\("), "function"),
    (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"), "function"),
    (re.compile(r"^\s*(?:public|private|protected|static|internal|virtual|override|final|async|\s)+[\w<>\[\],.:?\s*&]+?\s+(\w+)\s*\([^;{)]*\)?\s*(?:const|throws [\w., ]+)?\s*\{?\s*$"), "method"),
    (re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(TABLE|VIEW|FUNCTION|PROCEDURE|INDEX)\s+(?:IF\s+NOT\s+EXISTS\s+)?([\w.\"`\[\]]+)", re.I), None),
]
_KEYWORDS = {"if", "for", "while", "switch", "catch", "return", "else", "do", "new", "using", "foreach"}


@dataclass
class ChangeSet:
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)

    def __bool__(self):
        return bool(self.added or self.modified or self.deleted)

    def summary(self, limit=15) -> str:
        parts = []
        for label, items in (("added", self.added), ("modified", self.modified), ("deleted", self.deleted)):
            if items:
                more = f" (+{len(items) - limit} more)" if len(items) > limit else ""
                parts.append(f"{label}: " + ", ".join(items[:limit]) + more)
        return "; ".join(parts) or "no changes"


def _is_text(data: bytes) -> bool:
    return b"\x00" not in data[:4096]


def extract_symbols(rel: str, text: str) -> list[tuple[str, str, int]]:
    ext = Path(rel).suffix.lower()
    out: list[tuple[str, str, int]] = []
    if ext in (".py", ".pyw"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return out

        def walk(node, prefix=""):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(ch, ast.ClassDef) else ("method" if prefix else "function")
                    out.append((prefix + ch.name, kind, ch.lineno))
                    if isinstance(ch, ast.ClassDef):
                        walk(ch, prefix + ch.name + ".")
        walk(tree)
        return out
    if ext not in (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx", ".java", ".cs", ".cpp", ".cc", ".cxx",
                   ".hpp", ".h", ".c", ".php", ".sql"):
        return out
    for i, line in enumerate(text.splitlines(), 1):
        if len(line) > 300:
            continue
        for rx, kind in _GENERIC_SYMS:
            m = rx.match(line)
            if not m:
                continue
            if kind is None:
                out.append((m.group(2), m.group(1).lower(), i))
            else:
                name = m.group(1)
                if name not in _KEYWORDS:
                    out.append((name, kind, i))
            break
    return out


def fts_query(text: str) -> str:
    toks = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)][:12]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))


class ProjectIndex:
    def __init__(self, ws: Workspace, cfg: Config):
        self.ws, self.cfg = ws, cfg
        ws.data_dir.mkdir(exist_ok=True)
        self.db = sqlite3.connect(ws.data_dir / "index.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, size INT, mtime_ns INT, sha1 TEXT, lang TEXT);
            CREATE TABLE IF NOT EXISTS symbols(name TEXT, kind TEXT, path TEXT, line INT);
            CREATE INDEX IF NOT EXISTS idx_sym ON symbols(name);
        """)
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
                            "path UNINDEXED, start UNINDEXED, text, tokenize=\"unicode61 tokenchars '_'\")")
            self.fts = True
        except sqlite3.OperationalError:
            self.db.execute("CREATE TABLE IF NOT EXISTS chunks(path, start, text)")
            self.fts = False
        self.db.commit()

    # -- scanning ---------------------------------------------------------
    def iter_files(self):
        self._scan_truncated = False
        n = 0
        for dp, dns, fns in os.walk(self.ws.root):
            dns[:] = sorted(d for d in dns if d not in IGNORE_DIRS)
            for fn in sorted(fns):
                full = Path(dp) / fn
                if full.is_symlink():
                    continue
                try:
                    if full.stat().st_size > self.cfg.max_file_bytes:
                        continue
                except OSError:
                    continue
                n += 1
                if n > self.cfg.max_index_files:
                    self._scan_truncated = True
                    log.warning("index file cap reached (%d); deletion reconciliation skipped for this partial scan",
                                self.cfg.max_index_files)
                    return
                yield full

    def _index_file(self, full: Path, st=None) -> bool:
        rel = self.ws.rel(full)
        try:
            data = full.read_bytes()
        except OSError:
            return False
        st = st or full.stat()
        if not _is_text(data):
            return False
        text = data.decode("utf-8", errors="replace")
        self.db.execute("DELETE FROM chunks WHERE path=?", (rel,))
        self.db.execute("DELETE FROM symbols WHERE path=?", (rel,))
        lines = text.splitlines()
        for i in range(0, len(lines), CHUNK_LINES):
            self.db.execute("INSERT INTO chunks(path,start,text) VALUES(?,?,?)",
                            (rel, i + 1, "\n".join(lines[i:i + CHUNK_LINES])))
        for name, kind, line in extract_symbols(rel, text):
            self.db.execute("INSERT INTO symbols VALUES(?,?,?,?)", (name, kind, rel, line))
        self.db.execute("INSERT OR REPLACE INTO files VALUES(?,?,?,?,?)",
                        (rel, st.st_size, st.st_mtime_ns, hashlib.sha1(data).hexdigest(),
                         LANGS.get(full.suffix.lower(), "Other")))
        return True

    def refresh(self) -> ChangeSet:
        """Re-scan; return files added/modified/deleted since the previous refresh."""
        known = {r[0]: (r[1], r[2], r[3]) for r in self.db.execute("SELECT path,size,mtime_ns,sha1 FROM files")}
        first_run = not known
        seen, cs = set(), ChangeSet()
        for full in self.iter_files():
            rel = self.ws.rel(full)
            seen.add(rel)
            st = full.stat()
            old = known.get(rel)
            if old and old[0] == st.st_size and old[1] == st.st_mtime_ns:
                continue
            if old:
                if hashlib.sha1(full.read_bytes()).hexdigest() == old[2]:
                    self.db.execute("UPDATE files SET mtime_ns=? WHERE path=?", (st.st_mtime_ns, rel))
                    continue
                if self._index_file(full, st):
                    cs.modified.append(rel)
            elif self._index_file(full, st):
                cs.added.append(rel)
        if not self._scan_truncated:
            for rel in set(known) - seen:
                for t in ("files", "chunks", "symbols"):
                    self.db.execute(f"DELETE FROM {t} WHERE path=?", (rel,))
                cs.deleted.append(rel)
        self.db.commit()
        if first_run:
            cs = ChangeSet()   # nothing is "changed" on the very first scan
        log.info("index refresh: %s", cs.summary())
        return cs

    def update_file(self, full: Path):
        """Call after the agent's own writes so they are not reported as external changes."""
        if full.exists():
            self._index_file(full)
        else:
            rel = self.ws.rel(full)
            for t in ("files", "chunks", "symbols"):
                self.db.execute(f"DELETE FROM {t} WHERE path=?", (rel,))
        self.db.commit()

    # -- queries ----------------------------------------------------------
    def search_chunks(self, query: str, k=5) -> list[tuple[str, int, str]]:
        q = fts_query(query)
        if not q:
            return []
        if self.fts:
            try:
                rows = self.db.execute(
                    "SELECT path,start,text FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
                    (q, k)).fetchall()
                return [(p, int(s), t) for p, s, t in rows]
            except sqlite3.OperationalError:
                return []
        terms = re.findall(r"\w{3,}", query.lower())[:8]
        rows = self.db.execute("SELECT path,start,text FROM chunks").fetchall()
        scored = sorted(((sum(t in x.lower() for t in terms), p, int(s), x) for p, s, x in rows), reverse=True)
        return [(p, s, x) for sc, p, s, x in scored[:k] if sc]

    def find_symbol(self, name: str, limit=15):
        return self.db.execute(
            "SELECT name,kind,path,line FROM symbols WHERE name=? OR name LIKE ? ORDER BY name LIMIT ?",
            (name, f"%{name}%", limit)).fetchall()

    def languages(self) -> dict[str, int]:
        return dict(self.db.execute(
            "SELECT lang, COUNT(*) FROM files WHERE lang NOT IN ('JSON','Markdown','Text','TOML','YAML','XML','Other') "
            "GROUP BY lang ORDER BY 2 DESC").fetchall())

    def structure(self, max_entries=40) -> str:
        paths = [r[0] for r in self.db.execute("SELECT path FROM files ORDER BY path")]
        top: dict[str, int] = {}
        for p in paths:
            head = p.split("/")[0] if "/" in p else "."
            top[head] = top.get(head, 0) + 1
        lines = [f"{len(paths)} indexed files"]
        langs = self.languages()
        if langs:
            lines.append("languages: " + ", ".join(f"{k} ({v})" for k, v in langs.items()))
        entries = [p for p in paths if Path(p).name in
                   {"main.py", "app.py", "__main__.py", "manage.py", "index.js", "server.js", "app.js",
                    "index.ts", "main.ts", "Program.cs", "Main.java", "main.cpp", "index.php", "index.html"}]
        if entries:
            lines.append("entry points: " + ", ".join(entries[:6]))
        lines.append("top level: " + ", ".join(f"{k}{'/' if k != '.' else ' (root files)'} [{v}]"
                                               for k, v in sorted(top.items())[:max_entries]))
        deps = parse_manifests(self.ws.root)
        for eco, d in deps.items():
            names = list(d)[:12]
            lines.append(f"{eco} deps ({len(d)}): " + ", ".join(names) + (" ..." if len(d) > 12 else ""))
        return "\n".join(lines)
