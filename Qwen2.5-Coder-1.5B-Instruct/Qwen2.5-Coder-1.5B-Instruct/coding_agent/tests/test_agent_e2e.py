"""End-to-end loop tests with a scripted backend. They test the AGENT machinery, not model quality."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT, MODEL_DIR
from agent.app import build_agent
from agent.backends.mock import MockBackend
from agent.config import Config
from agent.safety.approvals import Approver


def call(name, **args):
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


PROJECT = {
    "calc.py": "def add(a, b):\n    return a - b\n",
    "tests/test_calc.py": "import unittest\nfrom calc import add\n\nclass T(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n",
    "requirements.txt": "",
}


class E2E(unittest.TestCase):
    def make(self, script, approve=True, files=PROJECT):
        if hasattr(self, 'td'):
            self.td.cleanup()
        self.td = tempfile.TemporaryDirectory()
        self.root = Path(self.td.name)
        for rel, c in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(c)
        cfg = Config(workspace=str(self.root), model_dir=str(MODEL_DIR), backend="mock", max_steps=8)
        self.backend = MockBackend(script)
        self.events = []
        self.agent = build_agent(cfg, Approver(lambda q: approve), backend=self.backend, on_event=self.events.append)
        return self.agent

    def tearDown(self):
        if hasattr(self, "td"):
            self.td.cleanup()

    def test_context_fit_compresses_large_tool_output_and_preserves_task(self):
        a = self.make(["ok"])
        a.cfg.n_ctx = 8192
        a.cfg.max_new_tokens = 768
        a.backend.count_tokens = lambda msgs: sum(len(m["content"]) for m in msgs)
        messages = [
            {"role": "system", "content": a.system},
            {"role": "user", "content": "<tool_response>\n" + ("A" * 20000) + "\n</tool_response>"},
            {"role": "assistant", "content": "tool call"},
            {"role": "user", "content": "Current task: fix the parser and keep this instruction."},
        ]
        original_messages = [dict(message) for message in messages]
        fitted = a._fit(messages)
        self.assertLessEqual(a.backend.count_tokens(fitted), a.cfg.n_ctx - a.cfg.max_new_tokens - 64)
        self.assertIn("Current task: fix the parser", fitted[-1]["content"])
        self.assertEqual(messages, original_messages)  # input history is not mutated

    def test_context_fit_rejects_impossible_system_prompt(self):
        a = self.make(["ok"])
        a.cfg.n_ctx = 1024
        a.cfg.max_new_tokens = 128
        a.backend.count_tokens = lambda msgs: sum(len(m["content"]) for m in msgs)
        with self.assertRaisesRegex(ValueError, "System prompt alone"):
            a._fit([{"role": "system", "content": "S" * 1000},
                    {"role": "user", "content": "task"}])

    def test_fix_bug_and_verify_with_tests(self):
        a = self.make([call("run_tests"), call("read_file", path="calc.py"),
                       call("edit_file", path="calc.py", search="a - b", replace="a + b"),
                       call("run_tests"), "Fixed add(): it subtracted instead of adding. Tests pass."])
        answer = a.ask("add() returns wrong results, fix it")
        self.assertIn("Fixed", answer)
        self.assertEqual((self.root / "calc.py").read_text(), "def add(a, b):\n    return a + b\n")
        msgs = self.backend.calls[-1]
        tool_outputs = [m["content"] for m in msgs if m["content"].startswith("<tool_response>")]
        self.assertIn("exit code: 1", tool_outputs[0])      # tests failed before the fix
        self.assertIn("exit code: 0", tool_outputs[-1])     # and pass after
        self.assertEqual(len(self.backend.calls), 5)

    def test_context_is_injected_and_system_prompt_has_tools(self):
        a = self.make(["ok"])
        a.ask("explain the add function")
        first = self.backend.calls[0]
        self.assertEqual(first[0]["role"], "system")
        self.assertIn("<tools>", first[0]["content"])
        self.assertIn("edit_file", first[0]["content"])
        self.assertIn("calc.py", first[-1]["content"])          # retrieved chunk / structure
        self.assertIn("Python", first[-1]["content"])

    def test_hallucinated_import_warning_reaches_model(self):
        a = self.make([call("write_file", path="m.py", content="import fantasylib\nprint(1)\n"), "done"])
        a.ask("write m.py")
        fed_back = self.backend.calls[1][-1]["content"]
        self.assertIn("fantasylib", fed_back)
        self.assertIn("WARNING", fed_back)

    def test_broken_code_rejected_and_model_sees_why(self):
        a = self.make([call("write_file", path="bad.py", content="def f(:\n"), "gave up"])
        a.ask("write bad.py")
        self.assertFalse((self.root / "bad.py").exists())
        self.assertIn("REJECTED", self.backend.calls[1][-1]["content"])

    def test_malformed_call_recovers_then_aborts_when_persistent(self):
        a = self.make(["<tool_call>not json</tool_call>", call("list_dir"), "fine"])
        self.assertEqual(a.ask("list"), "fine")
        b = self.make(["<tool_call>x</tool_call>"] * 4)
        self.assertIn("malformed", b.ask("list"))

    def test_duplicate_call_is_flagged(self):
        a = self.make([call("list_dir"), call("list_dir"), "done"])
        a.ask("list")
        self.assertIn("identical call already made", self.backend.calls[2][-1]["content"])

    def test_step_limit_prevents_infinite_loop(self):
        a = self.make([call("read_file", path="calc.py", start_line=i) for i in range(1, 30)])
        self.assertIn("Stopped after 8 steps", a.ask("loop forever"))

    def test_user_denial_is_respected_end_to_end(self):
        a = self.make([call("edit_file", path="calc.py", search="a - b", replace="a + b"), "I could not edit."], approve=False)
        a.ask("fix")
        self.assertEqual((self.root / "calc.py").read_text(), PROJECT["calc.py"])
        self.assertIn("DENIED", self.backend.calls[1][-1]["content"])

    def test_external_changes_detected_between_runs_and_history_kept(self):
        a = self.make(["first answer", "second answer"])
        a.ask("hello")
        (self.root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        a.ask("what changed?")
        second = self.backend.calls[1]
        self.assertIn("modified: calc.py", second[-1]["content"])
        self.assertTrue(any(m["role"] == "assistant" and "first answer" in m["content"] for m in second))

    def test_model_files_cannot_be_modified_via_agent(self):
        files = {n: (MODEL_DIR / n).read_text(encoding="utf-8") for n in ("config.json", "merges.txt")}
        td = tempfile.TemporaryDirectory()
        cfg = Config(workspace=str(MODEL_DIR), model_dir=str(MODEL_DIR), backend="mock")
        b = MockBackend([call("write_file", path="config.json", content="{}"),
                         call("delete_file", path="merges.txt"), "done"])
        # use a copy of the model dir inside tmp so the original is never at risk even if the guard failed
        import shutil
        copy = Path(td.name) / "m"
        shutil.copytree(MODEL_DIR, copy, ignore=shutil.ignore_patterns("coding_agent"))
        cfg = Config(workspace=str(copy), model_dir=str(copy), backend="mock")
        agent = build_agent(cfg, Approver(lambda q: True), backend=b)
        agent.ask("overwrite config")
        for n, content in files.items():
            self.assertEqual((copy / n).read_text(encoding="utf-8"), content)
        self.assertIn("Protected model file", b.calls[1][-1]["content"])
        td.cleanup()


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "cli.py"), *args], capture_output=True, text=True, timeout=60)

    def test_cli_commands(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "a.py").write_text("def f():\n    pass\n")
            r = self.run_cli("--workspace", td, "status")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("Python", r.stdout)
            r = self.run_cli("--workspace", td, "tools")
            self.assertIn("edit_file", r.stdout)
        r = self.run_cli("verify-model", "--no-tokenizer")
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("COMPATIBLE", r.stdout)

    def test_hf_backend_gives_clear_error_without_deps(self):
        with tempfile.TemporaryDirectory() as td:
            r = self.run_cli("--workspace", td, "ask", "hi")
            self.assertNotEqual(r.returncode, 0)
            self.assertEqual(r.returncode, 2)
            self.assertIn("Model weights not found", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
