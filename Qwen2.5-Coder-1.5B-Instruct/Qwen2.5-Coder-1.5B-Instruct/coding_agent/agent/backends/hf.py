"""Original inference path: Hugging Face transformers (Qwen2ForCausalLM), weights untouched."""
from __future__ import annotations

import logging
import os

from ..config import Config
from .base import Backend

log = logging.getLogger("agent.backend.hf")


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg: Config):
        d = os.path.abspath(cfg.model_dir or "")
        if not any(os.path.exists(os.path.join(d, w)) for w in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")):
            raise FileNotFoundError(
                f"Model weights not found in {d}. Download them into that folder (they are not part of this package):\n"
                "  huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct model.safetensors --local-dir <model_dir>")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("backend 'hf' needs: pip install torch transformers (>=4.37)") from e
        if not cfg.model_dir or not os.path.isdir(cfg.model_dir):
            raise FileNotFoundError(f"model_dir not found: {cfg.model_dir!r}")
        self.torch = torch
        self.cfg = cfg
        torch.set_num_threads(cfg.n_threads or os.cpu_count() or 4)
        self.tok = AutoTokenizer.from_pretrained(cfg.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_dir, torch_dtype=getattr(torch, cfg.dtype), low_cpu_mem_usage=True)
        self.model.eval()
        self.sampling = cfg.sampling()
        log.info("HF model loaded dtype=%s sampling=%s", cfg.dtype, self.sampling)

    def count_tokens(self, messages):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.tok(text).input_ids)

    def generate(self, messages, max_new_tokens):
        text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt")
        s = self.sampling
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=s["temperature"] > 0,
                temperature=max(s["temperature"], 1e-5), top_p=s["top_p"], top_k=s["top_k"],
                repetition_penalty=s["repetition_penalty"],
                eos_token_id=[151645, 151643], pad_token_id=151643)
        new_ids = out[0][inputs.input_ids.shape[1]:]
        return self.tok.decode(new_ids, skip_special_tokens=True)
