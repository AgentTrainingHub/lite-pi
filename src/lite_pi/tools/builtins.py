"""Definitions of built-in tools; registry owns policy and hook execution."""

from collections.abc import Callable
from pathlib import Path
from threading import Event

from lite_pi.resources import read_skill
from lite_pi.tools.contracts import JsonObject, ToolDefinition, obj
from lite_pi.tools.edit import edit_many, write_file
from lite_pi.tools.read import read_text
from lite_pi.tools.shell import run_shell
from lite_pi.tools.workspace import Workspace

STRING = {"type": "string"}
PATH = {"type": "string", "minLength": 1}
PAIR = obj({"old": {"type": "string", "minLength": 1}, "new": STRING}, ["old", "new"])


def builtin_tools(
    workspace: Workspace,
    *,
    max_bytes: int,
    shell_timeout: float,
    cancel_bash_after: float | None,
    evidence: Path,
    cancel_event: Event | None,
    scope: Callable[[], tuple[str, ...] | None],
    catalog: dict[str, JsonObject],
) -> list[ToolDefinition]:
    definitions = [
        ToolDefinition(
            "read",
            "Read UTF-8 project text. Continue using returned offset and column.",
            obj(
                {
                    "path": PATH,
                    "offset": {"type": "integer", "minimum": 1},
                    "column": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1},
                },
                ["path"],
            ),
            lambda args: read_text(workspace, **args, max_bytes=max_bytes),
            "safe",
            read_only=True,
        ),
        ToolDefinition(
            "edit",
            "Unique exact replacements in a UTF-8 file. Preserve exact line endings.",
            obj(
                {
                    "path": PATH,
                    "old": PAIR["properties"]["old"],
                    "new": STRING,
                    "edits": {"type": "array", "minItems": 1, "items": PAIR},
                },
                ["path"],
                oneOf=[
                    {"required": ["old", "new"], "not": {"required": ["edits"]}},
                    {
                        "required": ["edits"],
                        "not": {"anyOf": [{"required": ["old"]}, {"required": ["new"]}]},
                    },
                ],
            ),
            lambda args: edit_many(
                workspace,
                args["path"],
                args["edits"] if "edits" in args else [{"old": args["old"], "new": args["new"]}],
            ),
        ),
        ToolDefinition(
            "write",
            "Create a UTF-8 project file; replacing one requires overwrite=true.",
            obj({"path": PATH, "content": STRING, "overwrite": {"type": "boolean"}}, ["path", "content"]),
            lambda args: write_file(workspace, **args),
        ),
        ToolDefinition(
            "bash",
            "Run a command in an isolated project, with no network or host credentials.",
            obj({"command": {"type": "string", "minLength": 1}}, ["command"]),
            lambda args: run_shell(
                workspace.root,
                **args,
                timeout=shell_timeout,
                cancel_after=cancel_bash_after,
                evidence=evidence,
                cancel_event=cancel_event,
                allowed_paths=scope(),
            ),
        ),
    ]
    if catalog:
        definitions.append(
            ToolDefinition(
                "skill",
                "Read one project skill body or a resource inside that skill directory.",
                obj(
                    {
                        "name": {"type": "string", "minLength": 1},
                        "resource": {"type": "string", "minLength": 1},
                    },
                    ["name"],
                ),
                lambda args: read_skill(workspace, catalog, args),
                "safe",
                read_only=True,
            )
        )
    return definitions
