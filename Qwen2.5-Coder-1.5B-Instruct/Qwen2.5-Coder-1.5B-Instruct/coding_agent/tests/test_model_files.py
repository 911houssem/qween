"""Original-model integrity + tokenizer/config consistency (stdlib only)."""
import hashlib
import json
import unittest
from helpers import ROOT, MODEL_DIR
from agent.verify_model import verify


class ModelFileTests(unittest.TestCase):
    def test_original_files_byte_identical(self):
        want = json.loads((ROOT / "model_manifest.json").read_text())["files"]
        for name, h in want.items():
            self.assertEqual(hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest(), h, name)

    def test_verify_model_passes(self):
        ok, rep = verify(str(MODEL_DIR), str(ROOT / "model_manifest.json"), load_tokenizer=False)
        self.assertTrue(ok, "\n".join(rep))

    def test_chat_template_has_tool_format_we_mimic(self):
        t = json.loads((MODEL_DIR / "tokenizer_config.json").read_text())["chat_template"]
        for needle in ("<tool_call>", "<tool_response>", "<|im_start|>", "You may call one or more functions"):
            self.assertIn(needle, t)

    def test_context_memory_math(self):
        c = json.loads((MODEL_DIR / "config.json").read_text())
        head_dim = c["hidden_size"] // c["num_attention_heads"]
        per_tok = 2 * c["num_hidden_layers"] * c["num_key_value_heads"] * head_dim * 2   # K+V, bf16
        self.assertEqual(per_tok, 28672)
        self.assertLess(per_tok * 8192 / 2**20, 250)


if __name__ == "__main__":
    unittest.main()


class ChatTemplateRenderTest(unittest.TestCase):
    def test_agent_messages_render_with_original_template(self):
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
        except ImportError:
            self.skipTest("jinja2 not installed")
        from agent.core.prompt import system_prompt
        tpl = json.loads((MODEL_DIR / "tokenizer_config.json").read_text())["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        msgs = [{"role": "system", "content": system_prompt([])},
                {"role": "user", "content": "Task: fix"},
                {"role": "assistant", "content": '<tool_call>\n{"name": "list_dir", "arguments": {}}\n</tool_call>'},
                {"role": "user", "content": "<tool_response>\n[list_dir]\na.py\n</tool_response>"}]
        out = env.from_string(tpl).render(messages=msgs, add_generation_prompt=True, tools=None)
        self.assertEqual(out.count("<|im_start|>"), 5)
        self.assertTrue(out.endswith("<|im_start|>assistant\n"))
