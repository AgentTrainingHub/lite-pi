"""Independent artifact and approval checks for the integrated terminal task."""

import json
import re

from lite_pi.recovery import candidate_version
from lite_pi.tools.contracts import JsonObject


def citation_ids(runs: list[JsonObject]) -> set[str]:
    return {
        "cite:" + hit["chunk_id"]
        for run in runs
        for turn in run.get("transcript", [])
        for item in turn["results"]
        if item["name"] == "search_docs" and not item.get("is_error")
        for hit in item.get("hits", [])
        if hit["source"] == "docs/limits.md" and "QUERY_PAGE_CAP = 17" in hit["text"]
    }


def artifact_matches(text: str, format_name: str, citations: set[str]) -> bool:
    if format_name == "JSON":
        try:
            value = json.loads(text, object_pairs_hook=_unique_fields)
        except ValueError:
            return False
        return (
            isinstance(value, dict)
            and set(value) == {"limit", "citation"}
            and type(value["limit"]) is int
            and value["limit"] == 17
            and isinstance(value["citation"], str)
            and value["citation"] in citations
        )
    if format_name != "TEXT":
        return False
    match = re.fullmatch(r"limit=17\r?\n(cite:[A-Za-z0-9_-]+)(?:\r?\n)?", text)
    return bool(match and match[1] in citations)


def _unique_fields(pairs: list[tuple[str, object]]) -> JsonObject:
    fields: JsonObject = {}
    for key, value in pairs:
        if key in fields:
            raise ValueError("duplicate JSON field")
        fields[key] = value
    return fields


def approved_candidate(
    snapshot: JsonObject, format_name: str, citations: set[str], *, current_task: JsonObject
) -> JsonObject:
    pending: list[JsonObject] = [a for a in snapshot["approvals"] if a["decision"] == "pending"]
    if len(pending) != 1:
        raise AssertionError("expected exactly one pending approval")
    candidate = pending[0]
    intent = next(i for i in snapshot["intents"] if i["intent_id"] == candidate["intent_id"])
    scope = current_task["scope"]
    version = candidate_version(
        intent["name"],
        intent["arguments"],
        tuple(scope) if scope is not None else None,
        current_task["constraint"],
    )
    if (
        intent["name"] != "write"
        or intent["status"] != "awaiting_approval"
        or intent["arguments"].get("path") != "result.txt"
        or not artifact_matches(intent["arguments"].get("content", ""), format_name, citations)
        or current_task["workflow"] != snapshot["workflow"]
        or candidate["candidate_version"] != version
        or intent["candidate_version"] != version
    ):
        raise AssertionError("candidate differs from current authorized artifact")
    return candidate


def plan_verified(snapshot: JsonObject) -> bool:
    workflow = snapshot["workflow"]
    plan = workflow.get("plan")
    if not plan or len(plan["steps"]) != 3:
        return False
    steps = {step["id"]: step for step in plan["steps"]}
    definitions = {
        "sources": (["docs/limits.md"], [], {("read", "docs/limits.md"), ("search_docs", "QUERY_PAGE_CAP")}),
        "build": (["docs/limits.md"], ["sources"], {("read", "docs/limits.md"), ("write", "result.txt")}),
        "check": (["result.txt"], ["build"], {("read", "result.txt")}),
    }
    if steps.keys() != definitions.keys():
        return False
    intents = {intent["intent_id"]: intent for intent in snapshot["intents"]}
    if len(intents) != len(snapshot["intents"]):
        return False
    for identity, (inputs, dependencies, required) in definitions.items():
        step = steps[identity]
        if step["inputs"] != inputs or step["depends_on"] != dependencies:
            return False
        if not _step_verified(step, intents, required, workflow["requirement_version"]):
            return False
    return True


def _step_verified(
    step: JsonObject, intents: dict[str, JsonObject], required: set[tuple[str, str]], version: int
) -> bool:
    if step.get("valid") is not True or not step["checks"]:
        return False
    seen: set[str] = set()
    observed: set[tuple[str, str]] = set()
    versions: dict[str, str] = {}
    for check in step["checks"]:
        identity = check["intent_id"]
        intent = intents.get(identity)
        if identity in seen or intent is None:
            return False
        seen.add(identity)
        if not _receipt_matches(intent, check, step["evidence_epoch"], version):
            return False
        name, arguments, result = intent["name"], intent["arguments"], intent["result"]
        if name == "search_docs":
            if arguments.get("mode") != "hybrid" or arguments.get("query") != "QUERY_PAGE_CAP":
                return False
            observed.add((name, arguments["query"]))
        elif name in {"read", "write"}:
            path = arguments.get("path")
            digest = result.get("file_versions", {}).get(path)
            if (
                not isinstance(path, str)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                return False
            versions[path] = digest
            observed.add((name, path))
        else:
            return False
    return required <= observed and all(
        path in versions and versions[path] == step["input_versions"].get(path) for path in step["inputs"]
    )


def _receipt_matches(intent: JsonObject, check: JsonObject, epoch: int, version: int) -> bool:
    result = intent.get("result") or {}
    requirement = result.get("requirement_version")
    return (
        check.get("passed") is True
        and check["name"] == intent["name"]
        and intent["status"] == "settled"
        and result.get("executed") is True
        and result.get("is_error") is False
        and result.get("intent_id") == intent["intent_id"]
        and result.get("name") == intent["name"]
        and result.get("tool_call_id") == intent["call_id"]
        and type(requirement) is int
        and epoch <= requirement <= version
    )
