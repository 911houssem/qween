"""Tool registry with JSON-schema argument validation (small models send sloppy args)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Config
from ..safety.approvals import Approver
from ..safety.workspace import SecurityError, Workspace

log = logging.getLogger("agent.tools")
_TYPES = {"string": str, "integer": int, "boolean": bool}


@dataclass
class ToolContext:
    cfg: Config
    ws: Workspace
    approver: Approver
    index: Any
    memory: Any
    docs: Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict           # {"arg": {"type":..., "description":..., "required": bool, "default": ...}}
    fn: Callable[..., str]

    def schema(self) -> dict:
        props = {k: {"type": v["type"], "description": v.get("description", "")} for k, v in self.parameters.items()}
        req = [k for k, v in self.parameters.items() if v.get("required")]
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                  "parameters": {"type": "object", "properties": props, "required": req}}}


class Registry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"duplicate tool {tool.name}")
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools)}"
        if not isinstance(args, dict):
            return "ERROR: arguments must be a JSON object"
        clean: dict[str, Any] = {}
        for k, spec in tool.parameters.items():
            if k in args:
                v, want = args[k], _TYPES[spec["type"]]
                if want is int and isinstance(v, str) and v.strip().lstrip("-").isdigit():
                    v = int(v)
                if want is bool and isinstance(v, str):
                    v = v.strip().lower() in ("true", "1", "yes")
                if not isinstance(v, want) or (want is int and isinstance(v, bool)):
                    return f"ERROR: argument '{k}' must be {spec['type']}"
                clean[k] = v
            elif spec.get("required"):
                return f"ERROR: missing required argument '{k}' for {name}"
            elif "default" in spec:
                clean[k] = spec["default"]
        extra = set(args) - set(tool.parameters)
        if extra:
            return f"ERROR: unknown argument(s) {sorted(extra)} for {name}. Expected: {list(tool.parameters)}"
        try:
            out = tool.fn(self.ctx, **clean)
        except SecurityError as e:
            log.warning("tool %s blocked: %s", name, e)
            return f"ERROR: blocked by safety policy: {e}"
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, PermissionError, ValueError) as e:
            log.warning("tool %s: %s: %s", name, type(e).__name__, e)
            return f"ERROR: {e}"
        except Exception as e:  # tools must never crash the loop
            log.exception("tool %s failed", name)
            return f"ERROR: {type(e).__name__}: {e}"
        limit = self.ctx.cfg.max_tool_output_chars
        if len(out) > limit:
            out = out[:limit // 2] + f"\n... [truncated {len(out) - limit} chars] ...\n" + out[-limit // 2:]
        return out
