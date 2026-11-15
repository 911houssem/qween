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
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, d):
        if not self.skip and d.strip():
            self.parts.append(d.strip())


def _chunks(text: str, limit=900) -> list[str]:
    out, cur, heading = [], "", ""
    for para in re.split(r"\\n\\s*\\n", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            heading = para.splitlines()[0].lstrip("# ").strip()
        if len(cur) + len(para) > limit and cur:
            out.append(cur)
            cur = f"[{heading}] " if heading else ""
        cur += ("\\n" if cur else "") + para
        while len(cur) > limit * 2:
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
                            "source UNINDEXED, text, tokenize=\"unicode61 tokenchars '_'\" )")
            self.fts = True
        except sqlite3.OperationalError:
            self.db.execute("CREATE TABLE IF NOT EXISTS docs(source, text)")
            self.fts = False
        self.db.commit()

    def _remove_source_chunks(self, path: str, legacy_name: str | None = None):
        self.db.execute("DELETE FROM docs WHERE source=?", (path,))
        if legacy_name:
            # Older versions stored only f.name, so remove those stale chunks on re-ingest.
            self.db.execute("DELETE FROM docs WHERE source=?", (legacy_name,))

    def ingest(self, path: Path) -> int:
        """Ingest a file or directory; update changed chunks and prune removed files in directories."""
        path = path.expanduser().resolve()
        is_dir = path.is_dir()
        if not is_dir and not path.is_file():
            raise FileNotFoundError(path)
        files = [path] if path.is_file() else [p for p in sorted(path.rglob("*")) if p.is_file()]
        added = 0
        seen: set[str] = set()
        for f in files:
            if f.suffix.lower() not in DOC_EXT:
                continue
            try:
                if f.stat().st_size > 2_000_000:
                    continue
                raw = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            source = str(f.resolve())
            seen.add(source)
            sha = hashlib.sha1(raw.encode("utf-8")).hexdigest()
            row = self.db.execute("SELECT sha FROM sources WHERE path=?", (source,)).fetchone()
            if row and row[0] == sha:
                continue
            self._remove_source_chunks(source, f.name)
            if f.suffix.lower() in (".html", ".htm"):
                parser = _Strip()
                parser.feed(raw)
                raw = "\\n\\n".join(parser.parts)
            for chunk in _chunks(raw):
                self.db.execute("INSERT INTO docs(source,text) VALUES(?,?)", (source, chunk))
                added += 1
            self.db.execute("INSERT OR REPLACE INTO sources VALUES(?,?)", (source, sha))

        if is_dir:
            # Remove stale chunks when a document has been deleted from an ingested tree.
            rows = self.db.execute("SELECT path FROM sources").fetchall()
            for (stored,) in rows:
                try:
                    Path(stored).relative_to(path)
                except ValueError:
                    continue
                if stored not in seen:
                    self._remove_source_chunks(stored)
                    self.db.execute("DELETE FROM sources WHERE path=?", (stored,))
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
        terms = re.findall(r"\\w{3,}", query.lower())[:8]
        rows = self.db.execute("SELECT source,text FROM docs").fetchall()
        scored = sorted(((sum(t in x.lower() for t in terms), s, x) for s, x in rows), reverse=True)
        return [(s, x) for sc, s, x in scored[:k] if sc]
