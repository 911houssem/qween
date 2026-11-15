import unittest
from helpers import make_env


class FileToolTests(unittest.TestCase):
    def setUp(self):
        self.td, self.root, self.ctx, self.reg = make_env({
            "app.py": "def add(a, b):\n    return a - b\n",
            "dup.py": "x = 1\nx = 1\n",
            "web/index.html": "<html><body><p>hi</p></body></html>\n"})

    def tearDown(self):
        self.td.cleanup()

    def test_read_and_list(self):
        self.assertIn("return a - b", self.reg.call("read_file", {"path": "app.py"}))
        self.assertIn("app.py", self.reg.call("list_dir", {}))
        self.assertIn("web/index.html", self.reg.call("list_dir", {"depth": 2}))

    def test_edit_ok_and_backup(self):
        r = self.reg.call("edit_file", {"path": "app.py", "search": "a - b", "replace": "a + b"})
        self.assertIn("OK: edited", r)
        self.assertIn("validation: ok", r)
        self.assertEqual((self.root / "app.py").read_text(), "def add(a, b):\n    return a + b\n")
        self.assertTrue(list((self.root / ".agent" / "backups").glob("*_app.py")))

    def test_edit_not_found_and_ambiguous(self):
        self.assertIn("not found", self.reg.call("edit_file", {"path": "app.py", "search": "zzz", "replace": "y"}))
        self.assertIn("matches 2 places", self.reg.call("edit_file", {"path": "dup.py", "search": "x = 1", "replace": "x = 2"}))

    def test_syntax_error_is_rejected_and_file_unchanged(self):
        before = (self.root / "app.py").read_text()
        r = self.reg.call("edit_file", {"path": "app.py", "search": "return a - b", "replace": "return a +"})
        self.assertIn("REJECTED", r)
        self.assertEqual((self.root / "app.py").read_text(), before)

    def test_write_new_file_and_unverified_import_warning(self):
        r = self.reg.call("write_file", {"path": "pkg/new.py", "content": "import totally_made_up_pkg\nimport os\n"})
        self.assertIn("OK: created", r)
        self.assertIn("totally_made_up_pkg", r)
        self.assertNotIn("os,", r)

    def test_invalid_json_and_html_rejected(self):
        self.assertIn("REJECTED", self.reg.call("write_file", {"path": "c.json", "content": "{bad"}))
        self.assertIn("REJECTED", self.reg.call("write_file", {"path": "x.html", "content": "<p>a</div>"}))

    def test_delete_goes_to_trash_and_needs_approval(self):
        self.ctx.approver.prompt = lambda q: False
        self.assertIn("DENIED", self.reg.call("delete_file", {"path": "dup.py"}))
        self.assertTrue((self.root / "dup.py").exists())
        self.ctx.approver.prompt = lambda q: True
        self.assertIn("OK", self.reg.call("delete_file", {"path": "dup.py"}))
        self.assertFalse((self.root / "dup.py").exists())
        self.assertTrue(list((self.root / ".agent" / "trash").glob("*_dup.py")))

    def test_arg_validation(self):
        self.assertIn("missing required", self.reg.call("read_file", {}))
        self.assertIn("unknown argument", self.reg.call("read_file", {"path": "app.py", "bogus": 1}))
        self.assertIn("unknown tool", self.reg.call("hack", {}))
        self.assertIn("1| def add", self.reg.call("read_file", {"path": "app.py", "start_line": "1"}))  # "1" -> 1 coerced
        self.assertIn("must be integer", self.reg.call("read_file", {"path": "app.py", "start_line": "abc"}))

    def test_search_and_symbols(self):
        self.assertIn("app.py:2", self.reg.call("search_code", {"pattern": r"return\s+a"}))
        self.assertIn("add (function) app.py:1", self.reg.call("find_symbol", {"name": "add"}))

    def test_pre_broken_file_can_still_be_fixed(self):
        (self.root / "broken.py").write_text("def f(:\n  pass\n")
        r = self.reg.call("write_file", {"path": "broken.py", "content": "def f():\n    pass\n"})
        self.assertIn("OK", r)


class ShellTests(unittest.TestCase):
    def setUp(self):
        self.td, self.root, self.ctx, self.reg = make_env({"hello.py": "print('hi')\n", "slow.py": "import time; time.sleep(5)\n"})

    def tearDown(self):
        self.td.cleanup()

    def test_run_file_and_command(self):
        self.assertIn("hi", self.reg.call("run_file", {"path": "hello.py"}))
        self.assertIn("exit code: 0", self.reg.call("run_command", {"command": "python hello.py"}))

    def test_blocked_commands(self):
        for cmd in ("rm -rf .", "del /s *.*", "powershell -enc AAAA", "curl http://x", "python hello.py && echo hi",
                    "python hello.py > out.txt", "type C:\\Windows\\win.ini", "python ../x.py"):
            r = self.reg.call("run_command", {"command": cmd})
            self.assertIn("ERROR", r, cmd)

    def test_non_allowlisted_asks_and_can_be_denied(self):
        seen = []
        self.ctx.approver.prompt = lambda q: seen.append(q) or False
        r = self.reg.call("run_command", {"command": "python -c print(1)"})
        self.assertIn("DENIED", r)
        self.assertIn("RISKY_COMMAND", seen[0])

    def test_timeout(self):
        self.ctx.cfg.command_timeout = 1
        self.assertIn("timed out", self.reg.call("run_file", {"path": "slow.py"}))

    def test_run_tests_autodetect(self):
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_a.py").write_text(
            "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(1, 1)\n")
        r = self.reg.call("run_tests", {})
        self.assertIn("exit code: 0", r)


if __name__ == "__main__":
    unittest.main()
