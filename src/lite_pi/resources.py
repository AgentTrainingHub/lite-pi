"""Project instructions and skills. Neither source expands tool authority."""

from __future__ import annotations

import hashlib
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import yaml

from lite_pi.outcomes import tool_outcome
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.workspace import Workspace

INSTRUCTION_ORDER = ("AGENTS.override.md", "AGENTS.md")
SKILL_ROOT = ".lite-pi/skills"
RESOURCE_MAX_BYTES = 50 * 1024


class ResourceError(ValueError):
    pass


def load_context(root: Path) -> JsonObject:
    """Load the project root only. Parent, home, and global agent files are out of scope."""
    workspace = Workspace(Path(root))
    try:
        return {"instructions": _instructions(workspace), "skills": _skills(workspace)}
    except (OSError, ValueError) as error:
        raise ResourceError(str(error)) from error


def request_prefix(context: JsonObject) -> list[JsonObject]:
    messages = []
    adopted = context["instructions"]["adopted"]
    if adopted is not None:
        shadowed = context["instructions"]["shadowed"]
        note = ""
        if shadowed:
            listed = ", ".join(f"{item['path']} sha256 {item['sha256']}" for item in shadowed)
            note = f"\nShadowed and not applied: {listed}."
        messages.append(
            {
                "role": "system",
                "content": (
                    f"Project instructions from {adopted['path']} sha256 {adopted['sha256']}:\n"
                    f"{adopted['content'].rstrip()}\n"
                    "These instructions do not grant tools, files, or commands." + note
                ),
            }
        )
    skills = context["skills"]
    if skills:
        lines = [
            "Available skills. Call the skill tool with a name to read that body. "
            "A description is not the body and does not grant tools."
        ]
        for skill in skills:
            lines.append(
                f"- name: {skill['name']}; description: {skill['description']}; "
                f"location: {skill['path']}; sha256: {skill['sha256']}"
            )
        messages.append({"role": "system", "content": "\n".join(lines)})
    memories = context.get("memories")
    if memories:
        messages.extend(memories)
    return messages


def public_context(context: JsonObject) -> JsonObject:
    adopted = context["instructions"]["adopted"]
    result = {
        "instructions": {
            "adopted": None if adopted is None else {"path": adopted["path"], "sha256": adopted["sha256"]},
            "shadowed": [
                {"path": item["path"], "sha256": item["sha256"]}
                for item in context["instructions"]["shadowed"]
            ],
        },
        "skills": [
            {
                "name": item["name"],
                "description": item["description"],
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for item in context["skills"]
        ],
    }
    records = context.get("memory_records")
    if records:
        result["memories"] = records
    return result


def read_skill(workspace: Workspace, catalog: dict[str, JsonObject], args: JsonObject) -> JsonObject:
    skill = catalog.get(args["name"])
    if skill is None:
        return tool_outcome(
            {"text": f"Skill {args['name']} is not in the project catalog"},
            executed=False,
            is_error=True,
            reason="unknown_skill",
        )
    resource = args.get("resource")
    if resource is None:
        relative = skill["path"]
    else:
        parts = Path(resource).parts
        if not parts or resource.startswith(("/", "~")) or ".." in parts:
            return tool_outcome(
                {"text": "Skill resource escapes the skill directory"},
                executed=False,
                is_error=True,
                reason="skill_path",
            )
        relative = f"{skill['directory']}/{resource}"
    try:
        data = workspace.read(relative, remember=False, max_bytes=RESOURCE_MAX_BYTES)
        text = data.decode()
    except (OSError, ValueError, UnicodeError) as error:
        return tool_outcome(
            {"text": f"{type(error).__name__}: {error}"},
            executed=False,
            is_error=True,
            reason="missing_resource",
        )
    return {
        "text": text,
        "skill": skill["name"],
        "sha256": hashlib.sha256(data).hexdigest(),
        "path": relative,
        "resource": resource,
    }


def _instructions(workspace: Workspace) -> JsonObject:
    found: list[dict[str, str]] = []
    for name in INSTRUCTION_ORDER:
        try:
            data = workspace.read(name, remember=False, max_bytes=RESOURCE_MAX_BYTES)
        except FileNotFoundError:
            continue
        item = {"path": name, "sha256": hashlib.sha256(data).hexdigest()}
        if not found:
            item["content"] = _decode(data, f"instruction_encoding:{name}")
        found.append(item)
    return {
        "order": list(INSTRUCTION_ORDER),
        "adopted": found[0] if found else None,
        "shadowed": found[1:],
    }


def _skills(workspace: Workspace) -> list[JsonObject]:
    try:
        directories = workspace.directories(SKILL_ROOT)
    except FileNotFoundError:
        return []
    found: dict[str, JsonObject] = {}
    for directory in directories:
        path = f"{SKILL_ROOT}/{directory}/SKILL.md"
        data = workspace.read(path, remember=False, max_bytes=RESOURCE_MAX_BYTES)
        text = _decode(data, f"skill_encoding:{directory}")
        meta, _body = _frontmatter(text, directory)
        name = meta["name"]
        item = {
            "name": name,
            "description": meta["description"],
            "path": path,
            "directory": f"{SKILL_ROOT}/{directory}",
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        if name in found:
            raise ResourceError(f"skill_collision:{name}:{found[name]['path']}:{item['path']}")
        found[name] = item
    return list(found.values())


def _decode(data: bytes, code: str) -> str:
    try:
        return data.decode()
    except UnicodeError as error:
        raise ResourceError(code) from error


class _SkillLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.Node, deep: bool = False) -> dict[Hashable, Any]:
        if not isinstance(node, yaml.MappingNode):
            raise ResourceError("metadata must be a mapping")
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in keys:
                raise ResourceError("invalid_or_duplicate_metadata_key")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _frontmatter(text: str, directory: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ResourceError(f"missing_frontmatter:{directory}")
    end = next((index for index in range(1, len(lines)) if lines[index].strip() == "---"), None)
    if end is None:
        raise ResourceError(f"open_frontmatter:{directory}")
    try:
        meta = yaml.load("".join(lines[1:end]), Loader=_SkillLoader)
    except yaml.YAMLError as error:
        raise ResourceError(f"invalid_frontmatter:{directory}") from error
    if not isinstance(meta, dict):
        raise ResourceError(f"invalid_frontmatter:{directory}")
    name = meta.get("name")
    description = meta.get("description")
    if not isinstance(name, str) or not isinstance(description, str) or not name or not description.strip():
        raise ResourceError(f"incomplete_skill_metadata:{directory}")
    if len(name) > 64 or len(description) > 1024 or any(char.isspace() for char in name):
        raise ResourceError(f"skill_metadata_too_long:{directory}")
    return {"name": name, "description": description.strip()}, "".join(lines[end + 1 :])
