import json
import time
import unittest
from helpers import make_env
from agent.project.deps import parse_manifests, unverified_imports, check_dependency


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.td, self.root, self.ctx, self.reg = make_env({
            "src/main.py": "import requests\nclass Svc:\n    def run(self):\n        return 1\n\ndef helper():\n    pass\n",
            "web/app.js": "const x = require('express');\nfunction start() {}\nconst go = async (a) => a;\nclass Server {}\n",
            "Api.java": "public class Api {\n    public static void main(String[] args) {\n    }\n}\n",
            "db/schema.sql": "CREATE TABLE users (id INT);\n",
            "requirements.txt": "requests>=2.0\nflask==3.0  # web\n",
            "package.json": json.dumps({"dependencies": {"express": "^4"}, "devDependencies": {"jest": "1"}}),
            "node_modules/junk/x.js": "ignored\n"})
        self.index = self.ctx.index

    def tearDown(self):
        self.td.cleanup()

    def test_structure_languages_deps(self):
        s = self.index.structure()
        self.assertIn("Python", s); self.assertIn("JavaScript", s); self.assertIn("Java", s); self.assertIn("SQL", s)
        self.assertIn("python deps (2)", s); self.assertIn("npm deps (2)", s)
        self.assertNotIn("node_modules", s)

    def test_symbols_multi_language(self):
        names = {r[0] for r in self.index.db.execute("SELECT name FROM symbols")}
        for n in ("Svc", "Svc.run", "helper", "start", "go", "Server", "Api", "main", "users"):
            self.assertIn(n, names)

    def test_change_detection(self):
        self.assertFalse(self.index.refresh())
        time.sleep(0.01)
        (self.root / "src/main.py").write_text("def helper():\n    return 2\n")
        (self.root / "new.py").write_text("x=1\n")
        (self.root / "Api.java").unlink()
        cs = self.index.refresh()
        self.assertEqual((cs.modified, cs.added, cs.deleted), (["src/main.py"], ["new.py"], ["Api.java"]))
        self.assertFalse(self.index.refresh())

    def test_agent_own_edit_not_reported_as_external(self):
        self.reg.call("write_file", {"path": "mine.py", "content": "y = 2\n"})
        self.assertFalse(self.index.refresh())

    def test_chunk_search(self):
        hits = self.index.search_chunks("where is the helper function")
        self.assertTrue(any(p == "src/main.py" for p, _, _ in hits))

    def test_dependency_checks(self):
        self.assertIn("declared in python", check_dependency(self.root, "requests"))
        self.assertIn("NOT declared", check_dependency(self.root, "fantasylib"))
        self.assertIn("declared in npm", check_dependency(self.root, "express"))
        self.assertEqual(unverified_imports(self.root, "a.py", "import requests\nimport fantasylib\nfrom os import path\n"), ["fantasylib"])
        self.assertEqual(unverified_imports(self.root, "a.js", "const a=require('express'); import z from 'zzz'; import './local'"), ["zzz"])


class MemoryDocsTests(unittest.TestCase):
    def setUp(self):
        self.td, self.root, self.ctx, self.reg = make_env()

    def tearDown(self):
        self.td.cleanup()

    def test_memory_persists_across_instances(self):
        self.reg.call("remember", {"key": "style", "value": "use black, 4 spaces"})
        from agent.memory.store import MemoryStore
        again = MemoryStore(self.ctx.ws.data_dir)
        self.assertEqual(again.recall("black"), [("style", "use black, 4 spaces")])
        again.add_message("user", "q1"); again.add_message("assistant", "a1")
        self.assertEqual([m["role"] for m in again.recent_turns()], ["user", "assistant"])

    def test_docs_rag(self):
        d = self.root / "docs"; d.mkdir()
        (d / "api.md").write_text("# Widget API\n\nCall widget_create(name) to make a widget.\n\n# Other\n\nUnrelated text about databases.\n")
        self.assertGreater(self.ctx.docs.ingest(d), 0)
        self.assertEqual(self.ctx.docs.ingest(d), 0)           # unchanged -> skipped
        self.assertIn("widget_create", self.reg.call("search_docs", {"query": "how to create a widget"}))
        self.assertIn("no documentation matches", self.reg.call("search_docs", {"query": "zebra quantum"}))


if __name__ == "__main__":
    unittest.main()
