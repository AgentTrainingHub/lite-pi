"""Deterministic context contracts. Scripted summaries are not live-model evidence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lite_pi.compaction import compact_if_needed, estimate_request
from lite_pi.journal import SUMMARY_PREFIX, Journal, JournalError
from lite_pi.provider import ModelResult
from lite_pi.resources import ResourceError, load_context, request_prefix
from lite_pi.session import AgentSession
from lite_pi.tools.registry import ToolRegistry
from lite_pi.tools.workspace import Workspace


def model(content: str, *, status: str = "stop", attempts: int = 0, error: str | None = None) -> ModelResult:
    return ModelResult(
        status,
        {"role": "assistant", "content": content},
        raw_stop_reason=status if status != "error" else None,
        actual_model="deterministic-fixture",
        attempts=attempts,
        error=error,
        usage={"prompt_tokens": 12, "completion_tokens": 4},
    )


def call(name: str, arguments: str, identity: str) -> dict:
    return {"id": identity, "type": "function", "function": {"name": name, "arguments": arguments}}


class ContextContracts(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-context-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()

    def test_override_hides_agents_and_ignores_unrelated_files(self):
        (self.work / "AGENTS.md").write_text("SHADOW-RULE do not apply\n")
        (self.work / "AGENTS.override.md").write_text("OVERRIDE-RULE apply this\n")
        (self.work / "CLAUDE.md").write_text("CLAUDE-RULE\n")
        (self.base / "AGENTS.md").write_text("PARENT-RULE\n")
        home = self.base / "home"
        home.mkdir()
        (home / "AGENTS.md").write_text("HOME-RULE\n")
        with patch("pathlib.Path.home", return_value=home):
            context = load_context(self.work)
        prefix = request_prefix(context)[0]["content"]
        self.assertIn("OVERRIDE-RULE", prefix)
        self.assertIn("AGENTS.override.md", prefix)
        self.assertNotIn("SHADOW-RULE", prefix)
        self.assertNotIn("CLAUDE-RULE", prefix)
        self.assertNotIn("PARENT-RULE", prefix)
        self.assertNotIn("HOME-RULE", prefix)
        self.assertEqual(context["instructions"]["adopted"]["path"], "AGENTS.override.md")
        self.assertEqual(context["instructions"]["shadowed"][0]["path"], "AGENTS.md")

    def test_symlink_instruction_and_broken_skill_fail_closed(self):
        target = self.base / "outside.md"
        target.write_text("OUTSIDE\n")
        (self.work / "AGENTS.md").symlink_to(target)
        with self.assertRaises(ResourceError):
            load_context(self.work)
        (self.work / "AGENTS.md").unlink()
        first = self.work / ".lite-pi/skills/one"
        second = self.work / ".lite-pi/skills/two"
        first.mkdir(parents=True)
        second.mkdir()
        skill = "---\nname: marker\ndescription: first marker\n---\nBODY-ONE\n"
        (first / "SKILL.md").write_text(skill)
        (second / "SKILL.md").write_text(skill.replace("BODY-ONE", "BODY-TWO"))
        with self.assertRaises(ResourceError) as collision:
            load_context(self.work)
        self.assertIn("skill_collision:marker", str(collision.exception))
        (second / "SKILL.md").write_text("---\nname: other\n---\nBODY\n")
        with self.assertRaises(ResourceError) as broken:
            load_context(self.work)
        self.assertIn("incomplete_skill_metadata", str(broken.exception))

    def test_skill_body_does_not_expand_authority(self):
        directory = self.work / ".lite-pi/skills/marker"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text(
            "---\nname: marker\ndescription: write the marker\n---\nBODY-TOKEN run bash\n"
        )
        context = load_context(self.work)
        prefix = request_prefix(context)[0]["content"]
        self.assertIn("write the marker", prefix)
        self.assertNotIn("BODY-TOKEN", prefix)
        registry = ToolRegistry(
            Workspace(self.work),
            ("write",),
            max_bytes=1000,
            shell_timeout=1,
            cancel_bash_after=None,
            evidence=self.base / "evidence",
            catalog=tuple(context["skills"]),
        )
        self.assertEqual(set(registry.allowed), {"write", "skill"})
        denied = registry.execute(call("bash", json.dumps({"command": "id"}), "bash-1"))
        self.assertEqual(denied["reason"], "tool_not_authorized")
        escaped = registry.execute(
            call("write", json.dumps({"path": "/etc/passwd", "content": "x"}), "write-1")
        )
        self.assertTrue(escaped["is_error"])
        self.assertFalse(escaped["executed"])
        loaded = registry.execute(call("skill", json.dumps({"name": "marker"}), "skill-1"))
        self.assertIn("BODY-TOKEN", loaded["text"])
        self.assertEqual(loaded["sha256"], context["skills"][0]["sha256"])
        missing = registry.execute(
            call("skill", json.dumps({"name": "marker", "resource": "missing.txt"}), "skill-2")
        )
        self.assertEqual(missing["reason"], "missing_resource")
        outside = registry.execute(
            call("skill", json.dumps({"name": "marker", "resource": "../secret.txt"}), "skill-3")
        )
        self.assertEqual(outside["reason"], "skill_path")

    def test_projection_keeps_original_tool_pair_and_replaces_old_turns(self):
        journal = Journal.create(self.base / "session.jsonl", self.work, tools=("read",))
        journal.append("user", {"content": "OLD-FACT"})
        journal.append(
            "assistant",
            {
                "content": "",
                "tool_calls": [
                    {"id": "read-1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
                ],
            },
        )
        journal.append("tool", {"tool_call_id": "read-1", "content": "file text"})
        kept = journal.append("user", {"content": "LATEST"})
        journal.append("compaction", {"summary": "OLD-FACT", "first_kept_id": kept["id"]})
        projected = journal.messages()
        self.assertEqual(projected[0]["content"], SUMMARY_PREFIX + "OLD-FACT")
        self.assertEqual([message["content"] for message in projected[1:]], ["LATEST"])
        raw = (self.base / "session.jsonl").read_text()
        self.assertIn("file text", raw)
        self.assertIn("read-1", raw)
        with self.assertRaises(JournalError) as error:
            journal.append("compaction", {"summary": " ", "first_kept_id": kept["id"]})
        self.assertEqual(error.exception.code, "invalid_compaction")

    def test_single_turn_compacts_between_complete_groups_and_reopens_with_the_user(self):
        journal = Journal.create(self.base / "split.jsonl", self.work, tools=("read",))
        user = journal.append("user", {"content": "Keep duplicates and the current two-item limit."})
        for index in range(4):
            journal.append(
                "assistant",
                {"content": "", "tool_calls": [call("read", '{"path":"source.py"}', f"r-{index}")]},
            )
            journal.append("tool", {"tool_call_id": f"r-{index}", "content": "observed data " * 160})
        original = journal.path.read_bytes()
        with patch("lite_pi.compaction.complete", return_value=model("Keep duplicates; checks passed.")):
            messages, info = compact_if_needed(
                journal.messages(),
                journal=journal,
                context_tokens=1200,
                reserve_tokens=64,
                prefix=[],
                definitions=[],
                run_id="split-contract",
            )
        self.assertFalse(info["failed"])
        self.assertTrue(info["adopted"])
        self.assertEqual(messages[1], {"role": "user", "content": user["body"]["content"]})
        calls = [c["id"] for m in messages for c in m.get("tool_calls", [])]
        self.assertEqual(calls, [m["tool_call_id"] for m in messages if m["role"] == "tool"])
        self.assertTrue(journal.path.read_bytes().startswith(original))
        reopened = Journal(journal.path)
        self.assertEqual(reopened.projection(), journal.projection())
        self.assertEqual(reopened.projection()[1][1], user["id"])
        self.assertEqual(journal.chain()[-1]["body"]["turn_user_id"], user["id"])

    def test_split_boundary_rejects_tool_results_and_a_different_user_turn(self):
        journal = Journal.create(self.base / "split-invalid.jsonl", self.work, tools=("read",))
        first = journal.append("user", {"content": "first"})
        journal.append("assistant", {"content": "done"})
        second = journal.append("user", {"content": "second"})
        assistant = journal.append("assistant", {"content": "", "tool_calls": [call("read", "{}", "r")]})
        tool = journal.append("tool", {"tool_call_id": "r", "content": "observed"})
        original = journal.path.read_bytes()
        for kept, turn_user in [(tool, second), (assistant, first)]:
            with self.subTest(kept=kept["kind"]), self.assertRaises(JournalError):
                journal.append(
                    "compaction",
                    {"summary": "short", "first_kept_id": kept["id"], "turn_user_id": turn_user["id"]},
                )
            self.assertEqual(journal.path.read_bytes(), original)

    def test_second_split_does_not_restore_previously_omitted_groups(self):
        journal = Journal.create(self.base / "split-twice.jsonl", self.work, tools=("read",))
        journal.append("user", {"content": "Keep duplicates."})
        for index in range(4):
            journal.append(
                "assistant",
                {"content": "", "tool_calls": [call("read", '{"path":"source.py"}', f"r-{index}")]},
            )
            journal.append("tool", {"tool_call_id": f"r-{index}", "content": "observed data " * 160})
        with patch("lite_pi.compaction.complete", return_value=model("checkpoint " * 80)):
            first, info = compact_if_needed(
                journal.messages(),
                journal=journal,
                context_tokens=1900,
                reserve_tokens=64,
                prefix=[],
                definitions=[],
                run_id="split-first",
            )
        self.assertTrue(info["adopted"])
        context = estimate_request(first[1:], [], 64) + 64 + 52
        with patch("lite_pi.compaction.complete", return_value=model("Keep duplicates.")):
            second, info = compact_if_needed(
                first,
                journal=journal,
                context_tokens=context,
                reserve_tokens=64,
                prefix=[],
                definitions=[],
                run_id="split-second",
            )
        self.assertFalse(info["failed"])
        self.assertLessEqual(estimate_request(second, [], 64) + 64, context)
        self.assertEqual(Journal(journal.path).messages(), second)

    def test_summary_failure_preserves_history_and_budget_failure_makes_no_request(self):
        journal = Journal.create(self.base / "session.jsonl", self.work, tools=())
        session = AgentSession(self.work, journal=journal)
        requests = []

        def respond(messages, **_kwargs):
            requests.append(messages)
            return model("noted")

        big = "FACT keep-me\n" + ("DROP-PAD " * 400)
        with (
            patch("lite_pi.loop.complete", side_effect=respond),
            patch(
                "lite_pi.compaction.complete",
                return_value=model("", status="error", error="HTTP 500", attempts=1),
            ),
        ):
            first = session.submit(big, max_turns=1, max_tokens=32, context_tokens=100000)
            second = session.submit("continue", max_turns=1, max_tokens=32, context_tokens=1000)
        self.assertEqual(first["status"], "stop")
        self.assertEqual(second["status"], "compaction_failed")
        self.assertEqual(second["reason"], "HTTP 500")
        self.assertEqual(len(requests), 1)
        self.assertFalse(any(record["kind"] == "compaction" for record in journal.records))
        self.assertIn("DROP-PAD", journal.path.read_text())
        self.assertIn("continue", journal.path.read_text())

        fresh = Journal.create(self.base / "short.jsonl", self.work, tools=())
        alone = AgentSession(self.work, journal=fresh)
        with (
            patch("lite_pi.loop.complete") as session_call,
            patch("lite_pi.compaction.complete") as summary_call,
        ):
            blocked = alone.submit("DROP-PAD " * 400, max_turns=1, max_tokens=32, context_tokens=200)
        self.assertEqual(blocked["reason"], "recent_exceeds_budget")
        session_call.assert_not_called()
        summary_call.assert_not_called()
        self.assertIn("DROP-PAD", fresh.path.read_text())

    def test_scripted_summary_drops_padding_and_keeps_the_latest_steer(self):
        journal = Journal.create(self.base / "session.jsonl", self.work, tools=())
        session = AgentSession(self.work, journal=journal)
        seen = []
        big = "FACT keep-me\n" + ("DROP-PAD " * 400)

        def respond(messages, **_kwargs):
            seen.append(messages)
            return model("noted", attempts=1)

        def summarize(messages, **kwargs):
            self.assertEqual(kwargs["purpose"], "compaction")
            self.assertIn("DROP-PAD", json.dumps(messages))
            return model("FACT keep-me", attempts=1)

        history = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": "noted"},
            {"role": "user", "content": "continue"},
            {"role": "user", "content": "STEER-KEEP"},
        ]
        reserve = 64
        # Leave room for the host identity while forcing the old padded turn out.
        context_tokens = 1000
        self.assertLess(context_tokens, estimate_request(history, [], reserve) + reserve)
        with (
            patch("lite_pi.loop.complete", side_effect=respond),
            patch("lite_pi.compaction.complete", side_effect=summarize),
        ):
            session.submit(big, max_turns=1, max_tokens=reserve, context_tokens=100000)
            session.steer("STEER-KEEP")
            result = session.submit(
                "continue", max_turns=1, max_tokens=reserve, context_tokens=context_tokens
            )
            session.resume(max_turns=1, max_tokens=reserve, context_tokens=100000)
        continuation = json.dumps(seen[1])
        self.assertIn("FACT keep-me", continuation)
        self.assertIn("STEER-KEEP", continuation)
        self.assertNotIn("DROP-PAD", continuation)
        self.assertNotIn("DROP-PAD", json.dumps(seen[2]))
        self.assertTrue(result["compactions"][-1]["adopted"])
        self.assertEqual(
            result["compactions"][-1]["deviation"], 12 - result["compactions"][-1]["continuation_estimate"]
        )
        self.assertIn("DROP-PAD", journal.path.read_text())
        self.assertNotIn("DROP-PAD", json.dumps(result["messages"]))
        self.assertTrue(os.path.isfile(journal.path))

    def test_repeated_user_text_compacts_by_record_identity(self):
        journal = Journal.create(self.base / "session.jsonl", self.work)
        journal.append("user", {"content": "continue"})
        journal.append("assistant", {"content": "padding " * 1000})
        session = AgentSession(self.work, journal=journal)
        with (
            patch("lite_pi.compaction.complete", return_value=model("old goal", attempts=1)),
            patch("lite_pi.loop.complete", return_value=model("done", attempts=1)),
        ):
            result = session.submit("continue", context_tokens=1000, max_tokens=64)
        self.assertEqual(result["status"], "stop")
        self.assertTrue(result["compactions"][0]["adopted"])
        kept = [r for r in journal.records if r["kind"] == "user"][-1]
        self.assertEqual(result["compactions"][0]["first_kept_id"], kept["id"])

    def test_summary_obeys_remaining_calls_and_cancellation(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                journal = Journal.create(self.base / f"session-{cancel}.jsonl", self.work)
                journal.append("user", {"content": "old " * 1000})
                session = AgentSession(self.work, journal=journal)

                def summarize(_messages, **kwargs):
                    self.assertEqual(kwargs["max_attempts"], 1)
                    self.assertTrue(callable(kwargs["stop"]))
                    if cancel:
                        session.cancel()
                        self.assertTrue(kwargs["stop"]())
                    return model("old goal", attempts=1)

                with (
                    patch("lite_pi.compaction.complete", side_effect=summarize),
                    patch("lite_pi.loop.complete") as continuation,
                ):
                    result = session.submit("next", context_tokens=1000, max_tokens=64, max_model_calls=1)
                self.assertEqual(result["status"], "aborted" if cancel else "budget")
                self.assertEqual(result["request_count"], 1)
                continuation.assert_not_called()
                if cancel:
                    self.assertFalse(any(r["kind"] == "compaction" for r in journal.records))

    def test_compaction_rejects_other_branch_boundary_before_writing(self):
        journal = Journal.create(self.base / "session.jsonl", self.work)
        first = journal.append("user", {"content": "branch A"})
        journal.branch(journal.records[0]["id"], "branch B")
        before = journal.path.read_bytes(), journal.leaf_id()
        with self.assertRaises(JournalError):
            journal.append("compaction", {"summary": "invalid", "first_kept_id": first["id"]})
        self.assertEqual((journal.path.read_bytes(), journal.leaf_id()), before)

    def test_compaction_cannot_hide_unpaired_tools(self):
        journal = Journal.create(self.base / "session.jsonl", self.work)
        journal.append("user", {"content": "old"})
        journal.append("assistant", {"content": "", "tool_calls": [call("read", "{}", "pending")]})
        kept = journal.append("user", {"content": "next"})
        before = journal.path.read_bytes()
        with self.assertRaises(JournalError):
            journal.append("compaction", {"summary": "invalid", "first_kept_id": kept["id"]})
        self.assertEqual(journal.path.read_bytes(), before)

    def test_resource_discovery_rejects_ancestor_symlinks_and_hardlinks(self):
        outside = self.base / "outside"
        (outside / "skills/marker").mkdir(parents=True)
        skill = outside / "skills/marker/SKILL.md"
        skill.write_text("---\nname: marker\ndescription: outside\n---\nSECRET\n")
        (self.work / ".lite-pi").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ResourceError):
            load_context(self.work)
        (self.work / ".lite-pi").unlink()
        os.link(skill, self.work / "AGENTS.md")
        with self.assertRaises(ResourceError):
            load_context(self.work)
        (self.work / "AGENTS.md").unlink()
        directory = self.work / ".lite-pi/skills/marker"
        directory.mkdir(parents=True)
        os.link(skill, directory / "SKILL.md")
        with self.assertRaises(ResourceError):
            load_context(self.work)

    def test_yaml_frontmatter_and_resource_size_limits(self):
        from lite_pi.resources import RESOURCE_MAX_BYTES, read_skill

        directory = self.work / ".lite-pi/skills/marker"
        directory.mkdir(parents=True)
        skill = directory / "SKILL.md"
        skill.write_bytes(
            b"---\r\nname: marker\r\ndescription: >-\r\n  Read a marker:\r\n  preserve its value.\r\n---\r\nbody\r\n"
        )
        context = load_context(self.work)
        self.assertEqual(context["skills"][0]["description"], "Read a marker: preserve its value.")
        (directory / "large.txt").write_bytes(b"x" * (RESOURCE_MAX_BYTES + 1))
        loaded = read_skill(
            Workspace(self.work),
            {"marker": context["skills"][0]},
            {"name": "marker", "resource": "large.txt"},
        )
        self.assertTrue(loaded["is_error"])
        self.assertIn("exceeds", loaded["text"])
        for bad in ("name: marker\nname: other\ndescription: text", "name: marker\ndescription: [text]"):
            skill.write_text("---\n" + bad + "\n---\nbody\n")
            with self.assertRaises(ResourceError):
                load_context(self.work)

    def test_second_compaction_merges_summary_once_and_branches_stay_isolated(self):
        journal = Journal.create(self.base / "session.jsonl", self.work)
        journal.append("user", {"content": "FIRST-GOAL"})
        kept = journal.append("user", {"content": "padding " * 1000})
        journal.append("compaction", {"summary": "FIRST-GOAL", "first_kept_id": kept["id"]})
        session = AgentSession(self.work, journal=journal)

        def summarize(messages, **_kwargs):
            self.assertEqual(json.dumps(messages).count("FIRST-GOAL"), 1)
            return model("FIRST-GOAL second checkpoint", attempts=1)

        with (
            patch("lite_pi.compaction.complete", side_effect=summarize),
            patch("lite_pi.loop.complete", return_value=model("done", attempts=1)),
        ):
            result = session.submit("continue", context_tokens=1000, max_tokens=64)
        self.assertEqual(result["status"], "stop")
        reopened = Journal(journal.path)
        self.assertEqual(reopened.messages(), result["messages"])
        summary_leaf = journal.leaf_id()
        journal.branch(journal.records[0]["id"], "unrelated branch")
        self.assertEqual(journal.messages(), [{"role": "user", "content": "unrelated branch"}])
        journal.select(summary_leaf)
        self.assertIn("FIRST-GOAL", journal.messages()[0]["content"])

    def test_summary_request_and_run_limits_respect_model_window(self):
        journal = Journal.create(self.base / "session.jsonl", self.work)
        session = AgentSession(self.work, journal=journal)
        before = journal.path.read_bytes()
        with self.assertRaises(ValueError):
            session.submit("next", context_tokens=128001)
        self.assertEqual(journal.path.read_bytes(), before)
        journal.append("user", {"content": "old " * 200000})
        with patch("lite_pi.compaction.complete") as summary, patch("lite_pi.loop.complete") as normal:
            result = session.submit("next", context_tokens=1000, max_tokens=64)
        self.assertEqual(result["reason"], "summary_input_exceeds_context")
        self.assertEqual(result["request_count"], 0)
        summary.assert_not_called()
        normal.assert_not_called()
        self.assertFalse(any(r["kind"] == "compaction" for r in journal.records))
