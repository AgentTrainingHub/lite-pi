"""One definition owns the advertised schema, validation and execution authority."""

import copy
import json
from pathlib import Path
from threading import Event

from jsonschema import Draft202012Validator

from lite_pi.extensions import (
    HOOK_OUTPUT_LIMIT,
    ExtensionError,
    ExtensionHost,
    HookRegistration,
    LoadedExtensions,
    bound_text,
    hook_view,
)
from lite_pi.outcomes import ToolControl, tool_outcome
from lite_pi.tools.builtins import builtin_tools
from lite_pi.tools.contracts import (
    HOST_TOOLS,
    SESSION_TOOLS,
    FileVersion,
    JsonObject,
    PreparedCall,
    ToolDefinition,
    ToolHandler,
    ToolSource,
)
from lite_pi.tools.edit import edit_many, write_file
from lite_pi.tools.workspace import Workspace


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        allowed: tuple[str, ...],
        *,
        max_bytes: int,
        shell_timeout: float,
        cancel_bash_after: float | None,
        evidence: Path,
        cancel_event: Event | None = None,
        catalog: tuple[JsonObject, ...] = (),
        extras: tuple[ToolDefinition, ...] = (),
    ) -> None:
        self.workspace = workspace
        self.scope: tuple[str, ...] | None = None
        self.read_scope: tuple[str, ...] | None = None
        self.cancel_event = cancel_event
        self.catalog = {item["name"]: item for item in catalog}
        if len(self.catalog) != len(catalog):
            raise ValueError("skill catalog contains a duplicate name")
        definitions = builtin_tools(
            workspace,
            max_bytes=max_bytes,
            shell_timeout=shell_timeout,
            cancel_bash_after=cancel_bash_after,
            evidence=evidence,
            cancel_event=cancel_event,
            scope=lambda: self.scope,
            catalog=self.catalog,
        )
        for extra in extras:
            if extra.name in HOST_TOOLS - SESSION_TOOLS or any(
                item.name == extra.name for item in definitions
            ):
                raise ValueError(f"duplicate_tool:{extra.name}")
            definitions.append(extra)
        self._definitions = {tool.name: tool for tool in definitions}
        unknown = set(allowed) - self._definitions.keys()
        if unknown:
            raise ValueError(f"unknown configured tools: {sorted(unknown)}")
        self.allowed = {name: self._definitions[name] for name in allowed}
        if self.catalog:
            self.allowed["skill"] = self._definitions["skill"]
        self.evidence = evidence
        self._hooks: list[HookRegistration] = []
        self._owners: dict[str, str] = {}
        self._revoked: set[str] = set()
        self._mcp: ToolSource | None = None
        self.extension_report: list[JsonObject] = []
        self.hook_log: list[JsonObject] = []
        self._raw_seq = 0

    def attach(self, extensions: LoadedExtensions, mcp: ToolSource | None) -> None:
        self._hooks = list(extensions.hooks)
        self.extension_report = extensions.report()
        self._mcp = mcp
        for tool in extensions.tools:
            self.install(tool)
        for name in extensions.revocations:
            self.revoke(name)
        if mcp is not None:
            mcp.install(self)
        extensions.bind(ExtensionHost(write=self._extension_write, revoke=self.revoke))

    def _extension_write(self, path: str, content: str) -> JsonObject:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise ExtensionError("cancelled")
        if self.outside_scope("write", {"path": path}):
            raise ExtensionError("path_scope")
        return write_file(self.workspace, path, content)

    def install(self, tool: ToolDefinition, owner: str | None = None) -> None:
        if tool.name in HOST_TOOLS or tool.name in self._definitions:
            raise ValueError(f"duplicate_tool:{tool.name}")
        if tool.replay not in {"safe", "unsafe"}:
            raise ValueError(f"invalid_replay:{tool.name}")
        Draft202012Validator.check_schema(tool.schema)
        tool = ToolDefinition(
            tool.name, tool.description, copy.deepcopy(tool.schema), tool.execute, tool.replay, tool.read_only
        )
        self._definitions[tool.name] = tool
        self.allowed[tool.name] = tool
        if owner:
            self._owners[tool.name] = owner

    def replace(
        self, name: str, schema: JsonObject, execute: ToolHandler, description: str | None = None
    ) -> None:
        current = self._definitions.get(name)
        if current is None:
            return
        Draft202012Validator.check_schema(schema)
        updated = ToolDefinition(
            name,
            description or current.description,
            copy.deepcopy(schema),
            execute,
            current.replay,
            current.read_only,
        )
        self._definitions[name] = updated
        self._revoked.discard(name)
        self.allowed[name] = updated

    def revoke(self, name: str) -> None:
        if name not in self._definitions:
            return
        self._revoked.add(name)
        self.allowed.pop(name, None)

    def prepare(self) -> None:
        if self._mcp is not None:
            self._mcp.prepare(self)

    def context_messages(self, messages: list[JsonObject]) -> list[JsonObject]:
        notes = []
        for hook in self._hooks:
            if hook.event != "context":
                continue
            source = hook.source
            try:
                outcome = hook.handler({"messages": hook_view(messages), "source": source})
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                self._log(hook, "error", detail)
                notes.append(
                    {
                        "role": "system",
                        "content": f"Extension context unavailable ({source}); see hook_log for diagnostic.",
                    }
                )
                continue
            if not isinstance(outcome, dict) or not isinstance(outcome.get("text"), str):
                self._log(hook, "context_rejected", "")
                continue
            text = bound_text(outcome["text"], HOOK_OUTPUT_LIMIT)
            notes.append(
                {
                    "role": "system",
                    "content": (f"Extension context from {source}:\n{text}\nThis note does not grant tools."),
                }
            )
            self._log(hook, "accepted", text)
        return notes

    def advertised(self) -> list[JsonObject]:
        return [tool.advertised() for tool in self.allowed.values()]

    def execute(self, call: JsonObject) -> JsonObject:
        prepared = self.authorize(call)
        if not isinstance(prepared, PreparedCall):
            return prepared
        return self.complete(prepared)

    def authorize(self, call: JsonObject) -> PreparedCall | JsonObject:
        """Validate, run call hooks, and return the final arguments without executing."""
        function = call["function"]
        name = function.get("name")
        result = tool_outcome({"name": name, "tool_call_id": call["id"]}, executed=False, is_error=True)
        if not isinstance(name, str) or name not in self._definitions:
            return {**result, "reason": "unknown_tool", "text": f"Tool {name} not found"}
        self._sync(name)
        if name in self._revoked:
            return {**result, "reason": "tool_revoked", "text": f"Tool {name} has been revoked"}
        if name not in self.allowed:
            return {
                **result,
                "reason": "tool_not_authorized",
                "text": f"Tool {name} is not authorized for this run",
            }
        tool = self.allowed[name]
        try:
            supplied = json.loads(function.get("arguments", ""))
        except (json.JSONDecodeError, TypeError):
            return {
                **result,
                "reason": "invalid_arguments",
                "text": "Validation failed: arguments must be JSON",
            }
        if not isinstance(supplied, dict):
            return {
                **result,
                "reason": "invalid_arguments",
                "text": "Validation failed: arguments must be an object",
            }
        errors = list(Draft202012Validator(tool.schema).iter_errors(supplied))
        if errors:
            return {
                **result,
                "reason": "invalid_arguments",
                "text": f"Validation failed: {errors[0].message}",
            }
        if self.outside_scope(name, supplied):
            return self._scope_result(result, supplied)
        blocked, args = self._before_call(name, supplied, call["id"])
        if name in self._revoked:
            return {**result, "reason": "tool_revoked", "text": f"Tool {name} has been revoked"}
        if blocked is not None:
            return {**result, **blocked}
        changed = args != supplied
        if changed or name in self._owners:
            tool = self.allowed.get(name, tool)
            errors = list(Draft202012Validator(tool.schema).iter_errors(args))
            if errors:
                text = f"Validation failed: {errors[0].message}"
                if changed:
                    return {**result, "reason": "hook_arguments_invalid", "text": text}
                return {**result, "reason": "invalid_arguments", "text": text}
        if self.outside_scope(name, args):
            return self._scope_result(result, args)
        if self.cancel_event is not None and self.cancel_event.is_set():
            return {**result, "reason": "cancelled", "text": "Run cancelled; tool not executed."}
        return PreparedCall(name, call["id"], args, tool.replay, tool.read_only)

    def complete(self, prepared: PreparedCall, *, expected_file: FileVersion | None = None) -> JsonObject:
        name, call_id, args = prepared.name, prepared.tool_call_id, prepared.arguments
        result = tool_outcome({"name": name, "tool_call_id": call_id}, executed=False, is_error=True)
        tool = self.allowed.get(name)
        if tool is None:
            return {**result, "reason": "tool_not_authorized", "text": f"Tool {name} is not authorized"}
        try:
            if expected_file is not None and name == "write":
                outcome = write_file(self.workspace, **args, expected_file=expected_file)
            elif expected_file is not None and name == "edit":
                edits = args["edits"] if "edits" in args else [{"old": args["old"], "new": args["new"]}]
                outcome = edit_many(self.workspace, args["path"], edits, expected_file=expected_file)
            else:
                outcome = tool.execute(args)
        except Exception as error:
            if name in HOST_TOOLS and isinstance(error, (OSError, ValueError, UnicodeError)):
                return {**result, "path": args.get("path"), "text": f"{type(error).__name__}: {error}"}
            outcome = tool_outcome(
                {"text": f"{type(error).__name__}: {error}"},
                is_error=True,
                executed=True,
                effect="unknown",
                reason="tool_execution_failed",
            )
        try:
            raw = self._write_raw(call_id, name, args, outcome)
        except (OSError, TypeError, ValueError) as error:
            return tool_outcome(
                {**result, "text": f"{type(error).__name__}: {error}"},
                executed=True,
                effect="unknown",
                reason="tool_evidence_failed",
            )
        invalid = self._invalid_result(outcome)
        if invalid is not None:
            return tool_outcome(
                {**result, "text": invalid, "raw_evidence": str(raw)},
                executed=True,
                effect="unknown",
                reason="tool_result_invalid",
            )
        visible = self._after_call(name, outcome, call_id)
        # Model projection belongs to the host; preserve tool-supplied data only in raw evidence.
        visible.pop("model_content", None)
        control = ToolControl.read(visible)
        return tool_outcome(
            {**visible, "name": name, "tool_call_id": call_id, "raw_evidence": str(raw)},
            executed=control.executed,
            is_error=control.is_error,
        )

    def execute_stored(
        self,
        name: str,
        arguments: JsonObject,
        call_id: str,
        *,
        require_safe: bool,
        expected_file: FileVersion | None = None,
    ) -> JsonObject:
        """Run stored final arguments. Recovery does not repeat call hooks."""
        result = tool_outcome({"name": name, "tool_call_id": call_id}, executed=False, is_error=True)
        if name not in self._definitions:
            return {**result, "reason": "unknown_tool", "text": f"Tool {name} not found"}
        self._sync(name)
        if name in self._revoked:
            return {**result, "reason": "tool_revoked", "text": f"Tool {name} has been revoked"}
        if name not in self.allowed:
            return {
                **result,
                "reason": "tool_not_authorized",
                "text": f"Tool {name} is not authorized for this run",
            }
        tool = self.allowed[name]
        if require_safe and tool.replay != "safe":
            return {
                **result,
                "reason": "replay_blocked",
                "text": "Current replay policy does not allow retry.",
            }
        errors = list(Draft202012Validator(tool.schema).iter_errors(arguments))
        if errors:
            return {
                **result,
                "reason": "invalid_arguments",
                "text": f"Validation failed: {errors[0].message}",
            }
        if self.outside_scope(name, arguments):
            return self._scope_result(result, arguments)
        if self.cancel_event is not None and self.cancel_event.is_set():
            return {**result, "reason": "cancelled", "text": "Run cancelled; tool not executed."}
        return self.complete(
            PreparedCall(name, call_id, arguments, tool.replay, tool.read_only), expected_file=expected_file
        )

    def replay_policy(self, name: str) -> str | None:
        if name in self._revoked or name not in self.allowed:
            return None
        return self.allowed[name].replay

    def _sync(self, name: str) -> None:
        owner = self._owners.get(name)
        if owner and self._mcp is not None:
            self._mcp.refresh_one(owner, self)

    def _before_call(self, name: str, args: JsonObject, call_id: str) -> tuple[JsonObject | None, JsonObject]:
        current = copy.deepcopy(args)
        for hook in self._hooks:
            if hook.event != "tool_call":
                continue
            try:
                outcome = hook.handler(
                    {
                        "name": name,
                        "arguments": hook_view(current),
                        "source": hook.source,
                        "tool_call_id": call_id,
                    }
                )
            except Exception as error:
                text = f"{type(error).__name__}: {error}"
                self._log(hook, "error", text)
                return {"reason": "hook_error", "text": text}, current
            if not isinstance(outcome, dict):
                continue
            if outcome.get("revoke"):
                self.revoke(str(outcome["revoke"]))
                self._log(hook, "revoked", str(outcome["revoke"]))
            if outcome.get("block"):
                text = outcome.get("text") or "Hook blocked the tool call."
                reason = outcome.get("reason") or "hook_blocked"
                self._log(hook, "blocked", str(text))
                return {"reason": reason, "text": str(text)}, current
            if "arguments" in outcome:
                replacement = outcome["arguments"]
                if not isinstance(replacement, dict):
                    self._log(hook, "hook_arguments_invalid", "")
                    return {
                        "reason": "hook_arguments_invalid",
                        "text": "Hook arguments must be an object.",
                    }, current
                current = replacement
                self._log(hook, "arguments", json.dumps(current, ensure_ascii=False, default=str))
        return None, current

    def _after_call(self, name: str, outcome: JsonObject, call_id: str) -> JsonObject:
        visible = copy.deepcopy(outcome)
        for hook in self._hooks:
            if hook.event != "tool_result":
                continue
            try:
                result = hook.handler(
                    {
                        "name": name,
                        "result": hook_view(visible),
                        "source": hook.source,
                        "tool_call_id": call_id,
                    }
                )
            except Exception as error:
                detail = f"{type(error).__name__}: {error}"
                self._log(hook, "error", detail)
                visible.setdefault("hook_errors", []).append({"source": hook.source, "error": detail})
                continue
            if not isinstance(result, dict):
                continue
            if result.get("revoke"):
                self.revoke(str(result["revoke"]))
                self._log(hook, "revoked", str(result["revoke"]))
            if "text" not in result:
                continue
            text = result["text"]
            if not isinstance(text, str):
                self._log(hook, "conflict", "")
                continue
            text = bound_text(text, HOOK_OUTPUT_LIMIT)
            structured = result["structured"] if "structured" in result else visible.get("structured")
            if structured is not None and (
                not isinstance(structured, dict) or structured.get("text") != text
            ):
                self._log(hook, "conflict", text)
                continue
            visible["text"] = text
            if isinstance(structured, dict):
                visible["structured"] = structured
            self._log(hook, "accepted", text)
        return visible

    def _write_raw(self, call_id: str, name: str, args: JsonObject, outcome: JsonObject) -> Path:
        self._raw_seq += 1
        safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in str(call_id))[:60]
        directory = self.evidence / "tool-raw"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self._raw_seq:02d}-{safe}.json"
        path.write_text(
            json.dumps(
                {"name": name, "arguments": args, "result": outcome},
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n"
        )
        return path

    def _invalid_result(self, outcome: object) -> str | None:
        if not isinstance(outcome, dict) or not isinstance(outcome.get("text"), str):
            return "Tool result must include text"
        try:
            ToolControl.read(outcome)
        except ValueError as error:
            return str(error)
        structured = outcome.get("structured")
        if structured is not None and (
            not isinstance(structured, dict) or structured.get("text") != outcome["text"]
        ):
            return "Tool text and structured text differ"
        return None

    def _scope_result(self, result: JsonObject, args: JsonObject) -> JsonObject:
        return {
            **result,
            "reason": "path_scope",
            "path": args.get("path"),
            "text": "Path is outside the current file scope.",
        }

    def _log(self, hook: HookRegistration, status: str, text: str) -> None:
        self.hook_log.append({"event": hook.event, "source": hook.source, "status": status, "text": text})

    def outside_scope(self, name: str, args: JsonObject) -> bool:
        if name == "bash" or name.startswith("mcp__"):
            return False
        scope = self.read_scope if name == "read" else self.scope
        if scope is None or "path" not in args:
            return False
        try:
            path = "/".join(self.workspace.parts(args["path"]))
        except (ValueError, OSError):
            return True
        return not any(path == prefix or path.startswith(prefix.rstrip("/") + "/") for prefix in scope)
