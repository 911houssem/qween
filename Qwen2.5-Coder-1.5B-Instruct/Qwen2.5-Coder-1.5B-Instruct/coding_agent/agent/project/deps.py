"""Dependency awareness: manifests + import scanning to catch hallucinated packages."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover
    tomllib = None

NODE_BUILTINS = {"fs", "path", "os", "http", "https", "url", "util", "events", "stream", "crypto",
                 "child_process", "assert", "buffer", "zlib", "readline", "net", "dns", "tls",
                 "querystring", "timers", "worker_threads", "perf_hooks", "process", "module", "vm"}


def _norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n.strip().lower())


def parse_manifests(root: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}

    def add(eco, name, spec=""):
        out.setdefault(eco, {})[name] = spec

    for f in root.glob("requirements*.txt"):
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.split("#")[0].strip()
            m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(\[.*?\])?\s*(.*)$", line)
            if m and not line.startswith("-"):
                add("python", m.group(1), m.group(3))
    py = root / "pyproject.toml"
    if py.is_file() and tomllib:
        try:
            d = tomllib.loads(py.read_text(encoding="utf-8"))
            for dep in d.get("project", {}).get("dependencies", []):
                m = re.match(r"^([A-Za-z0-9_.\-]+)(.*)$", dep.strip())
                if m:
                    add("python", m.group(1), m.group(2))
        except ValueError:
            pass
    pj = root / "package.json"
    if pj.is_file():
        try:
            d = json.loads(pj.read_text(encoding="utf-8"))
            for sect in ("dependencies", "devDependencies"):
                for n, v in d.get(sect, {}).items():
                    add("npm", n, v)
        except ValueError:
            pass
    cj = root / "composer.json"
    if cj.is_file():
        try:
            d = json.loads(cj.read_text(encoding="utf-8"))
            for sect in ("require", "require-dev"):
                for n, v in d.get(sect, {}).items():
                    add("composer", n, v)
        except ValueError:
            pass
    pom = root / "pom.xml"
    if pom.is_file():
        for g, a in re.findall(r"<groupId>([^<]+)</groupId>\s*<artifactId>([^<]+)</artifactId>",
                               pom.read_text(encoding="utf-8", errors="ignore")):
            add("maven", f"{g}:{a}")
    for f in root.glob("*.csproj"):
        for n, v in re.findall(r'PackageReference\s+Include="([^"]+)"(?:\s+Version="([^"]+)")?',
                               f.read_text(encoding="utf-8", errors="ignore")):
            add("nuget", n, v)
    cm = root / "CMakeLists.txt"
    if cm.is_file():
        for n in re.findall(r"find_package\(\s*(\w+)", cm.read_text(encoding="utf-8", errors="ignore")):
            add("cmake", n)
    return out


def python_imports(source: str) -> set[str]:
    mods = set()
    for m in re.finditer(r"^\s*(?:from\s+([A-Za-z_][\w]*)[\w.]*\s+import|import[ \t]+([A-Za-z_][\w.,\t ]*))",
                         source, re.M):
        if m.group(1):
            mods.add(m.group(1))
        else:
            for part in m.group(2).split(","):
                part = part.strip().split(" as ")[0].strip().split(".")[0]
                if part:
                    mods.add(part)
    return mods


def js_imports(source: str) -> set[str]:
    mods = set()
    for m in re.finditer(r"""(?:require\(\s*|from\s+|import\s+)['"]([^'"]+)['"]""", source):
        spec = m.group(1)
        if spec.startswith((".", "/")) or spec.startswith("node:"):
            continue
        parts = spec.split("/")
        mods.add("/".join(parts[:2]) if spec.startswith("@") else parts[0])
    return mods


def unverified_imports(root: Path, path: str, source: str) -> list[str]:
    """Imports that are neither stdlib/builtin, local, declared in a manifest, nor installed."""
    ext = Path(path).suffix.lower()
    manifests = parse_manifests(root)
    bad: list[str] = []
    if ext in (".py", ".pyw"):
        declared = {_norm(n) for n in manifests.get("python", {})}
        for mod in sorted(python_imports(source)):
            if mod in sys.stdlib_module_names or mod == "__future__":
                continue
            if (root / mod).is_dir() or (root / f"{mod}.py").is_file() or (root / "src" / mod).is_dir():
                continue
            if _norm(mod) in declared or _norm(mod.replace("_", "-")) in declared:
                continue
            try:
                if importlib.util.find_spec(mod):
                    continue
            except (ImportError, ValueError):
                pass
            bad.append(mod)
    elif ext in (".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx"):
        declared = set(manifests.get("npm", {}))
        for mod in sorted(js_imports(source)):
            if mod in NODE_BUILTINS or mod in declared or (root / "node_modules" / mod).is_dir():
                continue
            bad.append(mod)
    return bad


def check_dependency(root: Path, name: str) -> str:
    manifests = parse_manifests(root)
    hits = []
    for eco, deps in manifests.items():
        for n, spec in deps.items():
            if _norm(n) == _norm(name) or n.lower() == name.lower():
                hits.append(f"declared in {eco} manifest: {n} {spec}".strip())
    installed = []
    try:
        if importlib.util.find_spec(name.replace("-", "_")):
            installed.append("importable in the Python environment running the agent")
    except (ImportError, ValueError):
        pass
    if (root / "node_modules" / name).is_dir():
        installed.append("present in node_modules")
    if name in sys.stdlib_module_names:
        installed.append("Python standard library")
    if name in NODE_BUILTINS:
        installed.append("Node.js builtin")
    if hits or installed:
        return "; ".join(hits + installed)
    return (f"'{name}' is NOT declared in any manifest and not found installed. "
            "Do not assume it exists; ask the user or add it to the manifest first.")
