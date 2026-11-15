"""Compatibility check against the ORIGINAL model files. Read-only; never writes to model_dir."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REQUIRED = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json", "merges.txt"]
WEIGHTS = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


def verify(model_dir: str, manifest: str | None = None, load_tokenizer: bool = True) -> tuple[bool, list[str]]:
    d, ok, rep = Path(model_dir), True, []

    def check(cond: bool, msg: str, fatal=True):
        nonlocal ok
        rep.append(("PASS  " if cond else ("FAIL  " if fatal else "WARN  ")) + msg)
        if not cond and fatal:
            ok = False

    for f in REQUIRED:
        check((d / f).is_file(), f"present: {f}")
    if not ok:
        return ok, rep
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    check(cfg.get("model_type") == "qwen2" and cfg.get("architectures") == ["Qwen2ForCausalLM"], "architecture is Qwen2ForCausalLM")
    check(cfg.get("num_hidden_layers") == 28 and cfg.get("hidden_size") == 1536, "28 layers, hidden size 1536 (1.5B)")
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    added = {a["content"]: a["id"] for a in tj["added_tokens"]}
    check(max(vocab.values()) + 1 == 151643 and added.get("<|im_end|>") == 151645, "tokenizer ids match (151643 base, <|im_end|>=151645)")
    check(all(k in added for k in ("<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|file_sep|>")), "FIM/repo tokens present")
    check(cfg["vocab_size"] >= max(added.values()) + 1, "embedding vocab_size covers all token ids")
    present = [w for w in WEIGHTS if (d / w).exists()] or [p.name for p in d.glob("model-*.safetensors")]
    check(bool(present), f"weights present: {present or 'NONE (needed for real inference)'}", fatal=False)
    if manifest and Path(manifest).is_file():
        want = json.loads(Path(manifest).read_text(encoding="utf-8"))["files"]
        for n, h in want.items():
            if (d / n).is_file():
                check(hashlib.sha256((d / n).read_bytes()).hexdigest() == h, f"unmodified (SHA-256): {n}")
            else:
                check(False, f"missing from model dir: {n}", fatal=False)
    if load_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(d))
            text = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
            check(text.endswith("<|im_start|>assistant\n"), "transformers tokenizer + chat template render correctly")
            check(tok.convert_tokens_to_ids("<|im_end|>") == 151645, "transformers resolves <|im_end|> to 151645")
        except ImportError:
            rep.append("SKIP  transformers not installed: tokenizer load test skipped")
        except Exception as e:  # noqa: BLE001
            check(False, f"tokenizer load failed: {e}")
    return ok, rep
