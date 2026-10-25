"""Real DeepSeek acceptance. Each invocation keeps immutable projects and observations."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Unpack, cast
from uuid import uuid4

from lite_pi.session import CODE_TOOLS, AgentSession, RunLimits
from lite_pi.spend.ledger import Ledger
from lite_pi.tools.contracts import JsonObject
from lite_pi.tools.shell import DOCKER_ENV, run_shell

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "baselines/coding/fixture"
FIX = "First run python3 check.py, read the failure, then fix the implementation. Do not change check.py. Run the check again and leave it passing."
FEATURE = "Add weightnote/report.py with line_count(lines), using split_line to count parsed lines. Keep check.py, invoice.py, total_grams unchanged. Run check.py and leave it passing."


class AcceptanceBlocked(RuntimeError):
    pass


def require_available(result: JsonObject) -> None:
    if result.get("status") == "blocked" or result.get("reason") == "isolation_unavailable":
        raise AcceptanceBlocked(result.get("reason") or "runtime unavailable")


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def snapshot(work: Path) -> dict[str, str]:
    values = {}
    for path in sorted(work.rglob("*")):
        if "__pycache__" in path.parts:
            continue
        name = str(path.relative_to(work))
        if path.is_symlink():
            values[name] = "symlink:" + os.readlink(path)
        elif path.is_file():
            values[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def results(run: JsonObject, name: str | None = None) -> list[JsonObject]:
    return [
        item for turn in run["transcript"] for item in turn["results"] if name is None or item["name"] == name
    ]


def output(item: JsonObject) -> str:
    return Path(item["output_path"]).read_text() if item.get("output_path") else str(item.get("text", ""))


def container_removed(item: JsonObject) -> bool:
    if not item.get("container"):
        return False
    observed = subprocess.run(
        ["docker", "inspect", item["container"]],
        capture_output=True,
        text=True,
        timeout=10,
        env=DOCKER_ENV,
    )
    return observed.returncode != 0 and "no such object" in observed.stderr.lower()


def cli(*args: str) -> JsonObject:
    proc = subprocess.run(
        [sys.executable, "-m", "lite_pi", *args],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            **({"HOME": os.environ["HOME"]} if "HOME" in os.environ else {}),
        },
        capture_output=True,
        text=True,
        timeout=180,
    )
    data = cast(JsonObject, json.loads(proc.stdout))
    if "--session" in args:
        session_path = Path(args[args.index("--session") + 1])
        evidence = session_path.parent / ("cli-" + uuid4().hex)
        evidence.mkdir()
        (evidence / "stdout.json").write_text(proc.stdout)
        (evidence / "stderr.txt").write_text(proc.stderr)
        save(evidence / "command.json", {"args": args, "returncode": proc.returncode})
    require_available(data)
    data["returncode"] = proc.returncode
    return data


def check(work: Path, code: str | None = None) -> JsonObject:
    command = "python3 check.py" if code is None else "python3 -c " + shlex.quote(code)
    observed = run_shell(work, command, evidence=work.parent / "verification", timeout=15, readonly=True)
    require_available(observed)
    return {
        "returncode": observed["exit_code"],
        "stdout": output(observed),
        "evidence": observed["output_path"],
    }


class Suite:
    def __init__(self, group: str):
        self.base = ROOT / ".artifacts/acceptance" / (group + "-" + uuid4().hex)
        self.base.mkdir(parents=True)
        self.records: list[JsonObject] = []
        save(self.base / "spend-before.json", Ledger().status())

    def project(self, name: str, *, fixture: bool = False) -> Path:
        base = self.base / name
        base.mkdir()
        work = base / "project"
        if fixture:
            shutil.copytree(FIXTURE, work)
        else:
            work.mkdir()
        return work

    def record(
        self, name: str, passed: bool, *, hit: bool = True, blocked: bool = False, **evidence: object
    ) -> None:
        item = {
            "name": name,
            "layer": "real_model",
            "status": "blocked"
            if blocked
            else "pass"
            if passed and hit
            else "fail"
            if hit
            else "not-covered",
            **evidence,
        }
        self.records.append(item)
        save(self.base / "cases.json", self.records)
        print(json.dumps(item, ensure_ascii=False), flush=True)

    def run(
        self, work: Path, task: str, *, tools: tuple[str, ...] = CODE_TOOLS, **limits: Unpack[RunLimits]
    ) -> JsonObject:
        before = snapshot(work)
        contents = {
            str(p.relative_to(work)): p.read_bytes()
            for p in work.rglob("*")
            if p.is_file() and not p.is_symlink()
        }
        observations = []
        callback = limits.pop("on_event", None)

        def observe(event: JsonObject) -> None:
            if event["type"] == "tool_result":
                item = event["result"]
                if item["name"] in {"edit", "write"} and not item["is_error"]:
                    name = item["path"]
                    identity = str((work / name).relative_to(work))
                    original = contents.get(identity, b"")
                    updated = (work / name).read_bytes()
                    expected = (
                        "".join(
                            difflib.unified_diff(
                                original.decode().splitlines(True),
                                updated.decode().splitlines(True),
                                fromfile=name,
                                tofile=name,
                            )
                        )
                        or "(no textual change)"
                    )
                    observations.append(
                        {
                            "call_id": item["tool_call_id"],
                            "path": name,
                            "before_hex": original.hex(),
                            "after_hex": updated.hex(),
                            "returned_diff": item["text"],
                            "matches": expected == item["text"],
                        }
                    )
                    contents[identity] = updated
            if callback:
                callback(event)

        limits["on_event"] = observe
        run = AgentSession(work, tools=tools).submit(task, **limits)
        require_available(run)
        after = snapshot(work)
        save(
            work.parent / (run["run_id"] + "-effects.json"),
            {"before": before, "after": after, "diffs": observations},
        )
        run["changed"] = sorted(
            path for path in before.keys() | after.keys() if before.get(path) != after.get(path)
        )
        run["diffs_match"] = all(item["matches"] for item in observations)
        return run

    def finish(self) -> int:
        groups = {item["effect"] for item in self.records if "effect" in item}
        effects = {
            group: sum(item["status"] == "pass" for item in self.records if item.get("effect") == group)
            for group in groups
        }
        boundaries = [item for item in self.records if "effect" not in item]
        passed = (
            all(count >= 2 for count in effects.values())
            and all(item.get("safe", True) for item in self.records)
            and all(item["status"] == "pass" for item in boundaries)
        )
        summary = {
            "status": "pass" if passed else "incomplete",
            "effects": effects,
            "cases": self.records,
            "evidence": str(self.base),
        }
        save(self.base / "summary.json", summary)
        save(self.base / "spend-after.json", Ledger().status())
        print(json.dumps({k: v for k, v in summary.items() if k != "cases"}), flush=True)
        return 0 if passed else 1
