import unittest
from helpers import ROOT  # noqa: F401
from agent.core.parser import parse


class ParserTests(unittest.TestCase):
    def test_plain_answer(self):
        p = parse("All done.")
        self.assertEqual((p.text, p.calls, p.errors), ("All done.", [], []))

    def test_single_call(self):
        p = parse('I will look.\n<tool_call>\n{"name": "list_dir", "arguments": {"path": "."}}\n</tool_call>')
        self.assertEqual(p.calls, [("list_dir", {"path": "."})])
        self.assertEqual(p.text, "I will look.")

    def test_two_calls(self):
        p = parse('<tool_call>{"name":"a","arguments":{}}</tool_call><tool_call>{"name":"b","arguments":{"x":1}}</tool_call>')
        self.assertEqual([c[0] for c in p.calls], ["a", "b"])

    def test_raw_newlines_in_string_tolerated(self):
        p = parse('<tool_call>{"name":"write_file","arguments":{"path":"a.py","content":"def f():\n    return 1\n"}}</tool_call>')
        self.assertEqual(p.calls[0][1]["content"], "def f():\n    return 1\n")

    def test_missing_closing_tag_and_fences(self):
        p = parse('<tool_call>\n```json\n{"name":"recall","arguments":{}}\n```')
        self.assertEqual(p.calls, [("recall", {})])

    def test_arguments_as_json_string(self):
        p = parse('<tool_call>{"name":"read_file","arguments":"{\\"path\\": \\"a.py\\"}"}</tool_call>')
        self.assertEqual(p.calls, [("read_file", {"path": "a.py"})])

    def test_garbage_reports_error(self):
        p = parse("<tool_call>this is not json</tool_call>")
        self.assertEqual(p.calls, [])
        self.assertEqual(len(p.errors), 1)


if __name__ == "__main__":
    unittest.main()
