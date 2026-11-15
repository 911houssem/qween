"""RAG over technical documentation: BM25 (SQLite FTS5). No embedding model, no network."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path

from ..project.index import fts_query

DOC_EXT = {".md", ".txt", ".rst", ".html", ".htm"}


class _Strip(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, a):
        self.skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self.skip -= tag in ("script", "style") and self.skip > 0

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.parts.append(d.strip())


def _chunks(text: str, limit=900) -> list[str]:
    out, cur, heading = [], "", ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            heading = para.splitlines()[0].lstrip("# ").strip()
        if len(cur) + len(para) > limit and cur:
            out.append(cur)
            cur = f"[{heading}] " if heading else ""
        cur += ("\n" if cur else "") + para
        while len(cur) > limit * 2:       # very long paragraph
            out.append(cur[:limit * 2])
            cur = cur[limit * 2:]
    if cur:
        out.append(cur)
    return out


class DocsIndex:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(data_dir / "docs.db")
        self.db.execute("CREATE TABLE IF NOT EXISTS sources(path TEXT PRIMARY KEY, sha TEXT)")
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5("
                            "source UNINDEXED, text, tokenize=\"unicode61 tokenchars '_'\")")
            self.fts = True
        except sqlite3.OperationalError:
            self.db.execute("CREATE TABLE IF NOT EXISTS docs(source, text)")
            self.fts = False

    def ingest(self, path: Path) -> int:
        """Ingest a file or directory. Returns number of chunks added (unchanged files skipped)."""
        files = [path] if path.is_file() else [p for p in sorted(path.rglob("*")) if p.is_file()]
        added = 0
        for f in files:
            if f.suffix.lower() not in DOC_EXT or f.stat().st_size > 2_000_000:
                continue
            raw = f.read_text(encoding="utf-8", errors="replace")
            sha = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            row = self.db.execute("SELECT sha FROM sources WHERE path=?", (str(f),)).fetchone()
            if row and row[0] == sha:
                continue
            if f.suffix.lower() in (".html", ".htm"):
                p = _Strip()
                p.feed(raw)
                raw = "\n\n".join(p.parts)
            self.db.execute("DELETE FROM docs WHERE source=?", (str(f),))
            for c in _chunks(raw):
                self.db.execute("INSERT INTO docs(source,text) VALUES(?,?)", (f.name, c))
                added += 1
            self.db.execute("INSERT OR REPLACE INTO sources VALUES(?,?)", (str(f), sha))
        self.db.commit()
        return added

    def search(self, query: str, k=4) -> list[tuple[str, str]]:
        q = fts_query(query)
        if not q:
            return []
        if self.fts:
            try:
                return self.db.execute("SELECT source,text FROM docs WHERE docs MATCH ? "
                                       "ORDER BY bm25(docs) LIMIT ?", (q, k)).fetchall()
            except sqlite3.OperationalError:
                return []
        terms = re.findall(r"\w{3,}", query.lower())[:8]
        rows = self.db.execute("SELECT source,text FROM docs").fetchall()
        scored = sorted(((sum(t in x.lower() for t in terms), s, x) for s, x in rows), reverse=True)
        return [(s, x) for sc, s, x in scored[:k] if sc]
