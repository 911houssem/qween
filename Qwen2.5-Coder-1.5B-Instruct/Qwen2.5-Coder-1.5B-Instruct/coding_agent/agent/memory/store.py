"""Persistent per-project memory + conversation history in a local SQLite file."""
from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path


class MemoryStore:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(data_dir / "memory.db")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS notes(key TEXT PRIMARY KEY, value TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT,
                                                role TEXT, content TEXT, ts REAL);
        """)
        self.session = uuid.uuid4().hex[:8]

    def remember(self, key: str, value: str):
        self.db.execute("INSERT OR REPLACE INTO notes VALUES(?,?,?)", (key.strip(), value.strip(), time.time()))
        self.db.commit()

    def recall(self, query: str = "", limit=20) -> list[tuple[str, str]]:
        if query:
            like = f"%{query}%"
            rows = self.db.execute("SELECT key,value FROM notes WHERE key LIKE ? OR value LIKE ? "
                                   "ORDER BY ts DESC LIMIT ?", (like, like, limit)).fetchall()
        else:
            rows = self.db.execute("SELECT key,value FROM notes ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return rows

    def forget(self, key: str) -> bool:
        cur = self.db.execute("DELETE FROM notes WHERE key=?", (key,))
        self.db.commit()
        return cur.rowcount > 0

    def add_message(self, role: str, content: str):
        self.db.execute("INSERT INTO messages(session,role,content,ts) VALUES(?,?,?,?)",
                        (self.session, role, content, time.time()))
        self.db.commit()

    def recent_turns(self, max_chars=2500, max_msgs=8) -> list[dict]:
        """Last user/assistant messages across sessions (final answers only), newest budgeted."""
        rows = self.db.execute("SELECT role,content FROM messages ORDER BY id DESC LIMIT ?",
                               (max_msgs,)).fetchall()
        out, used = [], 0
        for role, content in rows:
            content = content[:800]
            used += len(content)
            if used > max_chars:
                break
            out.append({"role": role, "content": content})
        out.reverse()
        while out and out[0]["role"] != "user":
            out.pop(0)
        return out
