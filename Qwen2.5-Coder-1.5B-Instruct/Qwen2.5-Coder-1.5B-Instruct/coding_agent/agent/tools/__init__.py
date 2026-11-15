from . import fs, project, shell
from .base import Registry, Tool, ToolContext

S, I, B = "string", "integer", "boolean"


def build_registry(ctx: ToolContext) -> Registry:
    r = Registry(ctx)
    T = lambda *a: r.register(Tool(*a))  # noqa: E731
    T("list_dir", "List files/folders in a workspace directory.",
      {"path": {"type": S, "default": "."}, "depth": {"type": I, "default": 1}}, fs.list_dir)
    T("read_file", "Read a text file with line numbers. Always read before editing.",
      {"path": {"type": S, "required": True}, "start_line": {"type": I, "default": 1},
       "end_line": {"type": I, "default": 0}}, fs.read_file)
    T("write_file", "Create a new file or fully overwrite one. Prefer edit_file for existing files.",
      {"path": {"type": S, "required": True}, "content": {"type": S, "required": True}}, fs.write_file)
    T("edit_file", "Replace ONE exact snippet in a file. 'search' must match exactly once.",
      {"path": {"type": S, "required": True}, "search": {"type": S, "required": True},
       "replace": {"type": S, "required": True}}, fs.edit_file)
    T("delete_file", "Delete a file (moved to trash; requires user approval).",
      {"path": {"type": S, "required": True}}, fs.delete_file)
    T("search_code", "Regex search across project files.",
      {"pattern": {"type": S, "required": True}, "glob": {"type": S, "default": "*"}}, fs.search_code)
    T("find_symbol", "Find where a function/class/table is defined.",
      {"name": {"type": S, "required": True}}, project.find_symbol)
    T("project_summary", "Project structure, languages, entry points and declared dependencies.", {},
      project.project_summary)
    T("check_dependency", "Check whether a package is declared/installed BEFORE using it.",
      {"name": {"type": S, "required": True}}, project.check_dependency)
    T("search_docs", "Search the ingested technical documentation.",
      {"query": {"type": S, "required": True}}, project.search_docs)
    T("run_command", "Run ONE command in the workspace (no pipes/redirects). Needs user approval.",
      {"command": {"type": S, "required": True}}, shell.run_command)
    T("run_file", "Run a .py/.js/.php file.",
      {"path": {"type": S, "required": True}, "args": {"type": S, "default": ""}}, shell.run_file)
    T("run_tests", "Run the project's tests (auto-detects pytest/unittest/npm/dotnet).",
      {"command": {"type": S, "default": ""}}, shell.run_tests)
    T("remember", "Save a durable project note (conventions, decisions).",
      {"key": {"type": S, "required": True}, "value": {"type": S, "required": True}}, project.remember)
    T("recall", "Read saved project notes.", {"query": {"type": S, "default": ""}}, project.recall)
    return r
