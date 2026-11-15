import os
import sys
import unittest
from helpers import make_env, MODEL_DIR
from agent.safety.workspace import SecurityError, Workspace
from agent.tools.shell import parse_command


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.td, self.root, self.ctx, self.reg = make_env({"a.py": "x = 1\n"})

    def tearDown(self):
        self.td.cleanup()

    def test_traversal_blocked(self):
        for bad in ("../secret.txt", "..\\..\\x", "sub/../../x", "/etc/passwd"):
            self.assertIn("ERROR", self.reg.call("read_file", {"path": bad}), bad)

    def test_symlink_escape_blocked(self):
        outside = self.root.parent / "outside.txt"
        outside.write_text("secret")
        try:
            os.symlink(outside, self.root / "link.txt")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        self.assertIn("ERROR", self.reg.call("read_file", {"path": "link.txt"}))
        self.assertEqual(outside.read_text(), "secret")

    def test_agent_dir_and_model_files_protected(self):
        ws = Workspace(self.root, MODEL_DIR)
        with self.assertRaises(SecurityError):
            ws.check_writable(".agent/index.db")
        mw = Workspace(MODEL_DIR, MODEL_DIR)
        for name in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
            with self.assertRaises(SecurityError):
                mw.check_writable(name)
        with self.assertRaises(SecurityError):
            mw.check_writable("model.safetensors")

    def test_denied_edit_leaves_file_untouched(self):
        self.ctx.approver.prompt = lambda q: False
        r = self.reg.call("edit_file", {"path": "a.py", "search": "x = 1", "replace": "x = 2"})
        self.assertIn("DENIED", r)
        self.assertEqual((self.root / "a.py").read_text(), "x = 1\\n")

    def test_python_code_is_never_auto_approved_as_safe_command(self):
        self.ctx.approver.auto_commands = True
        argv, risk = parse_command(self.ctx, f'{sys.executable} -c "print(1)"')
        self.assertEqual(Path(argv[0]).name.lower().removesuffix(".exe"), "python")
        self.assertEqual(risk, "risky_command")

    def test_test_and_package_scripts_require_interactive_approval(self):
        self.ctx.approver.auto_commands = True
        for command in ("pytest -q", "npm test", "npm run build", "dotnet test", "node app.js"):
            with self.subTest(command=command):
                _, risk = parse_command(self.ctx, command)
                self.assertEqual(risk, "risky_command")


if __name__ == "__main__":
    unittest.main()
