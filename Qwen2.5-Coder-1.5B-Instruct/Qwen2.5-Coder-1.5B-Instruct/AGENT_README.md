# Qwen2.5-Coder-1.5B Coding Agent

A local, modular coding agent built **around** the unmodified `Qwen2.5-Coder-1.5B-Instruct` model.
Nothing in the model is retrained, replaced or edited. The agent lives entirely in `coding_agent/`.

> **Status:** everything except real-model inference was tested (see *Testing results*). The package you
> received did **not** contain the weight file, so generation quality and speed on your PC are **unmeasured**.

## What it does
* Agent loop: retrieve context -> model proposes a tool call -> tool runs (validated, approved) -> result goes back -> repeat -> final answer.
* Tools: `list_dir, read_file, write_file, edit_file, delete_file, search_code, find_symbol, project_summary,
  check_dependency, search_docs, run_command, run_file, run_tests, remember, recall`.
* Project intelligence: SQLite index (files, symbols, FTS5/BM25 chunks), manifest/dependency parsing,
  external-change detection, persistent notes + conversation history per project, docs RAG (BM25, no embeddings).
* Languages: validators/symbol extraction for Python, JavaScript, TypeScript, C++, Java, PHP, SQL, HTML, CSS, C#
  (see *Known limitations* for which checks need external compilers).

## Safety model
| Rule | Enforcement |
|---|---|
| Stay in the workspace | every path resolved (symlinks and `..` included) and must be inside the root |
| Model files protected | `config.json`, tokenizer files, `*.safetensors/.gguf/.bin/...` refuse writes; `.agent/` is tool-read-only |
| Edits need consent | unified diff shown; `--yes-edits` auto-approves, still validated and backed up to `.agent/backups` |
| Destructive actions always ask | `delete_file` asks every time and moves to `.agent/trash` (never erases) |
| Commands | `shell=False`, no pipes/redirects, allowlist (python, pytest, node, git read-only, ...), denylist (rm, del, curl, powershell, ...), path args confined, 90 s timeout, secrets stripped from env |
| Bad generated code | syntax-validated **before** writing; a change that breaks a valid file is rejected |
| Hallucinated packages | imports not in a manifest / not installed produce a warning; `check_dependency` tool |

## Install (Windows, Python 3.10+)
```bat
cd Qwen2.5-Coder-1.5B-Instruct\coding_agent
python -m venv .venv && .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-hf.txt
huggingface-cli download Qwen/Qwen2.5-Coder-1.5B-Instruct model.safetensors --local-dir ..
python cli.py verify-model          :: must print COMPATIBLE and the weights line must PASS
```
The agent core needs no third-party packages; only the inference backend does.

## Run
```bat
python cli.py --workspace C:\work\myproject chat
python cli.py --workspace C:\work\myproject ask "add input validation to parse_args"
python cli.py --workspace C:\work\myproject status        :: structure + changes (no model loaded)
python cli.py --workspace C:\work\myproject docs-add C:\docs\framework   :: ingest docs for RAG
run_agent.bat C:\work\myproject chat
```
Flags: `--yes-edits`, `--yes-commands` (allowlisted commands only), `--config config.json`, `-v`.
Logs: `<workspace>\.agent\agent.log`. Per-project data: `<workspace>\.agent\`.

## Optional lighter backend (GGUF) - **not tested here**
`pip install llama-cpp-python`, convert the original weights with llama.cpp's `convert_hf_to_gguf.py`
and `llama-quantize` (e.g. Q4_K_M), then `--backend gguf --gguf-path model.gguf`. The original weights stay untouched.

## Memory budget (arithmetic from `config.json`, not a measurement)
bf16 weights: 1.54 B x 2 bytes ~= 3.1 GB. KV cache: 2 x 28 layers x 2 KV heads x 128 head-dim x 2 bytes = 28,672 B/token
-> ~235 MB at the default 8,192-token window. `float32` would need ~6.2 GB and is not recommended for 8 GB RAM.

## Files
**Original files modified: none.** (SHA-256 manifest in `coding_agent/model_manifest.json`; `verify-model` re-checks it.)
`README.md` (the model card) is untouched; this file is a new addition.

**New files:** `AGENT_README.md` and everything under `coding_agent/` (`cli.py`, `agent/**`, `tests/**`,
`config.example.json`, `requirements*.txt`, `run_agent.bat`, `model_manifest.json`).

## Known limitations
* **No real-model run was performed.** Tool-call reliability of a 1.5B model is unknown; the loop mitigates (strict arg
  validation, malformed-call retries, duplicate-call detection, step limit, pre-write validation) but cannot guarantee success.
  Expect weaker multi-step planning than larger models; keep tasks small.
* CPU generation will be slow on an i5 (no figure given because none was measured).
* **Tested on Linux only.** Windows-specific code paths (`.cmd` resolution for npm, backslash/drive-letter paths) are written
  defensively but unverified on Windows.
* Code you *run* via `run_file`/`run_tests` is **not sandboxed**: after approval it runs with your user permissions. Review before approving.
* Validators for JS/PHP/C++/Java/TS call `node`/`php`/`g++`/`javac`/`tsc` only if installed; otherwise the result is "skipped".
  C# and SQL are not syntax-checked (use `dotnet build` through `run_command`). HTML/CSS checks are structural only.
* Import/dependency checks are heuristic (e.g. pip name vs import name mismatches like `cv2`) and may warn needlessly.
* Symbol extraction for non-Python languages is regex-based and approximate.
* RAG covers only documentation you ingest (BM25 keyword search, no semantic matching).
* The agent sees the last few conversation turns, not full history; context is trimmed to the 8k window.
