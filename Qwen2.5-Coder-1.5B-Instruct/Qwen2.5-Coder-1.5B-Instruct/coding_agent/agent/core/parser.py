"""Parse Qwen-style <tool_call>{json}</tool_call> blocks, tolerating small-model sloppiness."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


@dataclass
class Parsed:
    text: str = ""
    calls: list[tuple[str, dict]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _loads(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw, strict=False)      # strict=False tolerates raw newlines in strings
    except ValueError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1], strict=False)
        raise


def parse(output: str) -> Parsed:
    p = Parsed()
    p.text = _BLOCK.sub("", output).strip()
    for m in _BLOCK.finditer(output):
        raw = m.group(1)
        if not raw:
            continue
        try:
            obj = _loads(raw)
            name = obj.get("name")
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args, strict=False)
            if not isinstance(name, str) or not isinstance(args, dict):
                raise ValueError("need {'name': str, 'arguments': object}")
            p.calls.append((name, args))
        except (ValueError, AttributeError) as e:
            p.errors.append(f"invalid tool_call ({e}): {raw[:120]!r}")
    return p
