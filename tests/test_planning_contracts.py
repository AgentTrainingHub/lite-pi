"""Planning and clarification at the public durable Session seam; controlled HTTP only."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from deterministic_transport import response, transport

from lite_pi.cli import _run_result
from lite_pi.journal import Journal
from lite_pi.session import AgentSession


def call(name: str, arguments: dict, identity: str = "plan-1") -> dict:
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def steps() -> list[dict]:
    return [
        {
            "id": "inspect",
            "title": "查看输入",
            "depends_on": [],
            "deliverable": "读取 source.py",
            "inputs": ["source.py"],
        },
        {
            "id": "change",
            "title": "修改结果",
            "depends_on": ["inspect"],
            "deliverable": "生成 result.txt",
            "inputs": [],
        },
        {
            "id": "check",
            "title": "核对结果",
            "depends_on": ["change"],
            "deliverable": "读取结果并核对内容",
            "inputs": ["result.txt"],
        },
    ]


class PlanContracts(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="lite-pi-planning-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.work = self.base / "project"
        self.work.mkdir()
        (self.work / "source.py").write_text("value = 3\n")
        self.journal = Journal.create(self.base / "session.jsonl", self.work, tools=("read", "plan"))
        self.session = AgentSession.open(self.journal.path)

    def create(self, definitions: list[dict] | None = None) -> dict:
        with transport(
            self.base / "create",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {
                                "action": "create",
                                "expected_version": 0,
                                "reason": "开始任务",
                                "steps": definitions or steps(),
                            },
                        )
                    ],
                )
            ],
        ):
            return self.session.submit("先列计划", max_turns=1)

    def test_model_plan_is_visible_after_reopening_without_claiming_verified_work(self) -> None:
        with transport(
            self.base / "wire",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {
                                "action": "create",
                                "expected_version": 0,
                                "reason": "用户要求三步检查",
                                "steps": steps(),
                            },
                        )
                    ],
                )
            ],
        ):
            result = self.session.submit("先列计划", max_turns=1)
        plan = result["tasks"]["workflow"]["plan"]
        self.assertEqual(result["status"], "turn_limit")
        self.assertEqual([s["id"] for s in plan["steps"]], ["inspect", "change", "check"])
        self.assertTrue(all(not s["checks"] and s["claim"] is None for s in plan["steps"]))
        self.assertEqual(AgentSession.open(self.journal.path).inspect()["workflow"]["plan"], plan)

    def test_revision_invalidates_dependants_and_keeps_an_independent_step(self) -> None:
        definitions = steps()
        definitions[1]["depends_on"] = []
        definitions[2]["depends_on"] = ["inspect"]
        first = self.create(definitions)["tasks"]["workflow"]["plan"]
        with transport(
            self.base / "revision",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {
                                "action": "revise",
                                "expected_version": 1,
                                "reason": "输入要求改变",
                                "steps": definitions,
                                "invalidated_steps": ["inspect"],
                            },
                            "revise-1",
                        )
                    ],
                )
            ],
        ):
            changed = self.session.submit("保留独立步骤，只重做输入相关步骤", max_turns=1)
        plan = changed["tasks"]["workflow"]["plan"]
        self.assertEqual(plan["id"], first["id"])
        self.assertEqual(plan["revision"], 2)
        self.assertEqual([s["valid"] for s in plan["steps"]], [False, True, False])
        self.assertEqual(AgentSession.open(self.journal.path).inspect()["workflow"]["plan"], plan)

    def test_step_claim_is_separate_from_the_observed_tool_evidence(self) -> None:
        self.create()
        with transport(
            self.base / "read", [response("", calls=[call("read", {"path": "source.py"}, "read-1")])]
        ):
            read = self.session.submit("读取输入", max_turns=1)
        observed = read["transcript"][0]["results"][0]
        with transport(
            self.base / "record",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {
                                "action": "record",
                                "expected_version": 1,
                                "step_id": "inspect",
                                "claim": "已读取输入",
                                "evidence": [observed["intent_id"]],
                            },
                            "record-1",
                        )
                    ],
                )
            ],
        ):
            recorded = self.session.submit("记录已执行的工作", max_turns=1)
        step = recorded["tasks"]["workflow"]["plan"]["steps"][0]
        self.assertEqual(step["claim"], "已读取输入")
        self.assertTrue(step["checks"][0]["passed"])
        self.assertEqual(step["checks"][0]["intent_id"], observed["intent_id"])
        (self.work / "source.py").write_text("value = 9\n")
        with transport(self.base / "changed-input", [response("输入已变化")]):
            refreshed = AgentSession.open(self.journal.path).submit("继续", max_turns=1)
        changed = refreshed["tasks"]["workflow"]["plan"]
        self.assertEqual([s["valid"] for s in changed["steps"]], [False, False, False])
        self.assertEqual(changed["steps"][0]["claim"], "已读取输入")
        self.assertTrue(changed["steps"][0]["checks"][0]["passed"])

    def test_question_waits_without_polling_and_accepts_a_durable_user_answer(self) -> None:
        journal = Journal.create(self.base / "question.jsonl", self.work, tools=("question", "write"))
        session = AgentSession.open(journal.path)
        with transport(
            self.base / "ask",
            [
                response(
                    "",
                    calls=[
                        call(
                            "question",
                            {
                                "expected_version": 0,
                                "text": "输出 JSON 还是纯文本？",
                                "options": ["JSON", "纯文本"],
                                "paths": ["result.txt"],
                                "step_ids": [],
                            },
                            "ask-1",
                        )
                    ],
                )
            ],
        ) as wire:
            waiting = session.submit("按选定格式生成结果", max_turns=4, approvals=True)
        self.assertEqual(waiting["status"], "awaiting_question")
        self.assertEqual(wire.call_count, 1)
        state = waiting["tasks"]["workflow"]
        receipt, exit_code = _run_result(waiting)
        question = receipt["question"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(question["expected_version"], state["version"])
        self.assertNotEqual(question["expected_version"], state["question"]["requirement_version"])
        self.assertEqual(question["options"], ["JSON", "纯文本"])
        with transport(self.base / "waiting", []) as wire:
            paused = AgentSession.open(journal.path).resume()
        self.assertEqual(paused["status"], "awaiting_question")
        self.assertEqual(wire.call_count, 0)
        answer = AgentSession.open(journal.path).answer(
            question["id"], "JSON", expected_version=question["expected_version"]
        )
        self.assertEqual(answer["question"]["status"], "answered")
        self.assertEqual(answer["question"]["answer"], "JSON")
        self.assertFalse(AgentSession.open(journal.path).inspect()["approvals"])
        with self.assertRaises(ValueError):
            AgentSession.open(journal.path).answer(
                state["question"]["id"], "纯文本", expected_version=state["version"]
            )

    def test_host_revision_supersedes_an_approved_but_unexecuted_candidate(self) -> None:
        journal = Journal.create(self.base / "approval.jsonl", self.work, tools=("plan", "write"))
        session = AgentSession.open(journal.path)
        with transport(
            self.base / "candidate",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {"action": "create", "expected_version": 0, "reason": "开始", "steps": steps()},
                        )
                    ],
                ),
                response("", calls=[call("write", {"path": "result.txt", "content": "old"}, "write-1")]),
            ],
        ):
            pending = session.submit("写结果，先审批", approvals=True)
        approval = pending["tasks"]["approvals"][0]
        session.decide(approval["approval_id"], "approve")
        session.revise_plan(
            expected_version=1, reason="改为新结果", steps=steps(), invalidated_steps=("change",)
        )
        self.assertFalse((self.work / "result.txt").exists())
        self.assertEqual(session.decide(approval["approval_id"], "approve")["decision"], "superseded")
        with transport(self.base / "after-change", [response("需要新的候选")]):
            result = AgentSession.open(journal.path).resume()
        self.assertFalse((self.work / "result.txt").exists())
        self.assertEqual(result["tasks"]["workflow"]["plan"]["revision"], 2)

    def test_cancelling_a_question_does_not_supply_an_answer_or_resume_work(self) -> None:
        journal = Journal.create(self.base / "cancel.jsonl", self.work, tools=("question",))
        session = AgentSession.open(journal.path)
        with transport(
            self.base / "ask-cancel",
            [
                response(
                    "",
                    calls=[
                        call(
                            "question",
                            {
                                "expected_version": 0,
                                "text": "选哪个？",
                                "options": ["甲", "乙"],
                                "paths": [],
                                "step_ids": [],
                            },
                        )
                    ],
                )
            ],
        ):
            waiting = session.submit("需要确认的任务")
        state = waiting["tasks"]["workflow"]
        answered = session.answer(state["question"]["id"], "", expected_version=state["version"], cancel=True)
        self.assertEqual(answered["question"]["status"], "cancelled")
        self.assertIsNone(answered["question"]["answer"])
        with transport(self.base / "no-answer", []) as wire:
            stopped = AgentSession.open(journal.path).resume()
        self.assertEqual(stopped["status"], "awaiting_decision")
        self.assertEqual(wire.call_count, 0)

    def read_result(self, path: str, directory: str) -> dict:
        with transport(self.base / directory, [response("", calls=[call("read", {"path": path})])]):
            result = self.session.submit("读取证据", max_turns=1)
        return result["transcript"][0]["results"][0]

    def record_result(self, intent: str | list[str], version: int, step_id: str = "inspect") -> dict:
        with transport(
            self.base / f"record-{version}",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {
                                "action": "record",
                                "expected_version": version,
                                "step_id": step_id,
                                "claim": "模型声明完成",
                                "evidence": [intent] if isinstance(intent, str) else intent,
                            },
                        )
                    ],
                )
            ],
        ):
            return self.session.submit("记录证据", max_turns=1)

    def test_failed_evidence_invalidates_step_and_dependants(self) -> None:
        definitions = steps()
        definitions[0]["inputs"] = []
        self.create(definitions)
        failed = self.read_result("missing.txt", "missing")
        result = self.record_result(failed["intent_id"], 1)
        plan = result["tasks"]["workflow"]["plan"]
        self.assertFalse(plan["steps"][0]["checks"][0]["passed"])
        self.assertEqual([s["valid"] for s in plan["steps"]], [False, False, False])

    def test_pre_revision_evidence_cannot_restore_invalidated_work(self) -> None:
        self.create()
        old = self.read_result("source.py", "old-read")
        self.session.revise_plan(
            expected_version=1, reason="要求变化", steps=steps(), invalidated_steps=("inspect",)
        )
        result = self.record_result(old["intent_id"], 2)
        self.assertFalse(result["tasks"]["workflow"]["plan"]["steps"][0]["valid"])
        self.assertIn("evidence_predates_requirement", result["transcript"][0]["results"][0]["text"])

    def test_target_branch_inputs_are_checked_before_resume(self) -> None:
        self.create()
        read = self.read_result("source.py", "branch-read")
        self.record_result(read["intent_id"], 1)
        self.journal.reload()
        target = self.journal.leaf_id()
        self.session.branch(self.journal.records[0]["id"], "独立分支")
        (self.work / "source.py").write_text("value = 9\n")
        with transport(self.base / "branch-resume", [response("已恢复")]):
            result = self.session.resume(leaf_id=target, max_turns=1)
        self.assertFalse(result["tasks"]["workflow"]["plan"]["steps"][0]["valid"])

    def test_revision_journal_intent_is_adopted_after_interruption(self) -> None:
        self.create()
        self.journal.reload()
        args = {
            "action": "revise",
            "expected_version": 1,
            "reason": "中断前已提交的用户要求",
            "steps": steps(),
            "invalidated_steps": ["inspect"],
        }
        self.journal.append("user", {"content": args["reason"], "plan_revision": args})
        with transport(self.base / "revision-resume", [response("恢复修订")]):
            result = AgentSession.open(self.journal.path).resume(max_turns=1)
        self.assertEqual(result["tasks"]["workflow"]["plan"]["revision"], 2)
        with transport(self.base / "revision-again", [response("仍为同一修订")]):
            again = AgentSession.open(self.journal.path).resume(max_turns=1)
        self.assertEqual(again["tasks"]["workflow"]["plan"]["revision"], 2)

    def test_invalid_dependencies_and_stale_revision_leave_the_plan_unchanged(self) -> None:
        self.create()
        for index, kind in enumerate(("cycle", "unknown", "duplicate", "stale")):
            definitions = steps()
            if kind == "cycle":
                definitions[0]["depends_on"] = ["check"]
            elif kind == "unknown":
                definitions[0]["depends_on"] = ["absent"]
            elif kind == "duplicate":
                definitions[1]["id"] = "inspect"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.session.revise_plan(
                    expected_version=0 if kind == "stale" else 1,
                    reason="拒绝无效修订",
                    steps=definitions,
                    invalidated_steps=(),
                )
            self.assertEqual(self.session.inspect()["workflow"]["version"], 1)

    def test_waiting_question_allows_only_proven_independent_read_in_current_batch(self) -> None:
        journal = Journal.create(
            self.base / "independent.jsonl", self.work, tools=("read", "write", "plan", "question")
        )
        session = AgentSession.open(journal.path)
        definitions = steps()
        definitions[1]["depends_on"] = []
        definitions[1]["inputs"] = ["independent.txt"]
        (self.work / "independent.txt").write_text("保留资料")
        with transport(
            self.base / "independent-wire",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {
                                "action": "create",
                                "expected_version": 0,
                                "reason": "独立分析",
                                "steps": definitions,
                            },
                        )
                    ],
                ),
                response(
                    "",
                    calls=[
                        call(
                            "question",
                            {
                                "expected_version": 1,
                                "text": "确定输入？",
                                "options": [],
                                "paths": ["source.py"],
                                "step_ids": ["inspect"],
                            },
                            "ask",
                        ),
                        call("read", {"path": "source.py"}, "affected"),
                        call("read", {"path": "independent.txt"}, "independent"),
                        call("write", {"path": "new.txt", "content": "不应写入"}, "write"),
                    ],
                ),
            ],
        ) as wire:
            result = session.submit("确认之前可读取独立资料", max_turns=4)
        self.assertEqual(result["status"], "awaiting_question")
        self.assertEqual(wire.call_count, 2)
        outcomes = result["transcript"][1]["results"]
        self.assertEqual([r["executed"] for r in outcomes], [True, False, True, False])
        self.assertFalse((self.work / "new.txt").exists())

    def test_question_is_not_answerable_from_another_branch_or_after_revision(self) -> None:
        journal = Journal.create(self.base / "isolation.jsonl", self.work, tools=("question", "plan"))
        session = AgentSession.open(journal.path)
        with transport(
            self.base / "isolation-wire",
            [
                response(
                    "",
                    calls=[
                        call(
                            "plan",
                            {"action": "create", "expected_version": 0, "reason": "开始", "steps": steps()},
                        )
                    ],
                ),
                response(
                    "",
                    calls=[
                        call(
                            "question",
                            {
                                "expected_version": 1,
                                "text": "选择？",
                                "options": ["甲", "乙"],
                                "paths": ["source.py"],
                                "step_ids": ["inspect"],
                            },
                        )
                    ],
                ),
            ],
        ):
            pending = session.submit("需要选择")
        state = pending["tasks"]["workflow"]
        journal.reload()
        leaf = journal.leaf_id()
        session.branch(leaf, "新分支")
        with self.assertRaisesRegex(ValueError, "question_not_pending_on_branch"):
            session.answer(state["question"]["id"], "甲", expected_version=state["version"])
        with transport(self.base / "return-branch", []) as wire:
            session.resume(leaf_id=leaf)
        self.assertEqual(wire.call_count, 0)
        session.revise_plan(
            expected_version=2, reason="改要求", steps=steps(), invalidated_steps=("inspect",)
        )
        with self.assertRaises(ValueError):
            session.answer(state["question"]["id"], "乙", expected_version=2)

    def test_durable_answer_is_adopted_after_interrupted_host_submission(self) -> None:
        journal = Journal.create(self.base / "adopt-answer.jsonl", self.work, tools=("question",))
        session = AgentSession.open(journal.path)
        with transport(
            self.base / "adopt-wire",
            [
                response(
                    "",
                    calls=[
                        call(
                            "question",
                            {
                                "expected_version": 0,
                                "text": "选择？",
                                "options": [],
                                "paths": [],
                                "step_ids": [],
                            },
                        )
                    ],
                )
            ],
        ):
            result = session.submit("确认")
        state = result["tasks"]["workflow"]
        journal.reload()
        journal.append(
            "user",
            {
                "content": "自由答复",
                "question_id": state["question"]["id"],
                "question_version": state["version"],
                "question_cancelled": False,
            },
        )
        with transport(self.base / "adopt-resume", [response("已根据答复继续")]):
            resumed = AgentSession.open(journal.path).resume()
        self.assertEqual(resumed["tasks"]["workflow"]["question"]["answer"], "自由答复")
        self.assertEqual(resumed["tasks"]["workflow"]["version"], 2)

    def test_answer_rejects_an_incomplete_tool_batch_until_resume_repairs_it(self) -> None:
        from lite_pi.tasks import TaskBook

        journal = Journal.create(self.base / "interrupted-question.jsonl", self.work, tools=("question",))
        session = AgentSession.open(journal.path)
        with (
            transport(
                self.base / "interrupted-ask",
                [
                    response(
                        "",
                        calls=[
                            call(
                                "question",
                                {
                                    "expected_version": 0,
                                    "text": "确定？",
                                    "options": [],
                                    "paths": [],
                                    "step_ids": [],
                                },
                            )
                        ],
                    )
                ],
            ),
            patch.object(TaskBook, "commit", side_effect=RuntimeError("submission interrupted")),
        ):
            with self.assertRaises(RuntimeError):
                session.submit("确认")
        state = session.inspect()["workflow"]
        with self.assertRaisesRegex(ValueError, "resume_required_before_answer"):
            session.answer(state["question"]["id"], "已确认", expected_version=state["version"])
        with transport(self.base / "repair-question", []) as wire:
            repaired = AgentSession.open(journal.path).resume()
        self.assertEqual(wire.call_count, 0)
        self.assertEqual(repaired["status"], "awaiting_question")
        answer = AgentSession.open(journal.path).answer(state["question"]["id"], "已确认", expected_version=1)
        self.assertEqual(answer["question"]["status"], "answered")

    def test_process_crash_after_question_effect_recovers_without_a_second_question(self) -> None:
        import os
        import subprocess
        import sys

        from tui_pty import ROOT

        path = self.base / "crash.jsonl"
        command = [
            sys.executable,
            str(ROOT / "tests/fixtures/tui_driver.py"),
            "code",
            "--root",
            str(self.work),
            "--session",
            str(path),
            "--planning",
            "PLAN_QUESTION",
        ]
        crashed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                "HOME": os.environ["HOME"],
                "PATH": "/usr/bin:/bin",
                "LITE_PI_CRASH_AFTER": "effect",
            },
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(crashed.returncode, 86, crashed.stderr)
        before = AgentSession.open(path).inspect()["workflow"]
        self.assertEqual(before["question"]["status"], "pending")
        with transport(self.base / "crash-repair", []) as wire:
            resumed = AgentSession.open(path).resume()
        self.assertEqual(wire.call_count, 0)
        self.assertEqual(resumed["status"], "awaiting_question")
        self.assertEqual(resumed["tasks"]["workflow"], before)

    def test_model_receives_branch_local_execution_ids_for_plan_evidence(self) -> None:
        self.create()
        observed = self.read_result("source.py", "visible-read")
        with transport(self.base / "visible-context", [response("核对可用证据")]) as wire:
            self.session.submit("现在记录证据", max_turns=1)
        request = wire.call_args.args[0]
        context = "\n".join(m["content"] for m in request["messages"] if m["role"] == "system")
        self.assertIn(observed["intent_id"], context)
        self.assertIn("available_evidence", context)
        self.assertIn("Clarification asks for missing task requirements, never execution permission", context)
        self.assertIn("host will stop BEFORE its effect", context)
        self.assertIn("source.py", context)
        prefix = next(
            m["content"]
            for m in request["messages"]
            if m["role"] == "system" and "Current host workflow state" in m["content"]
        )
        catalog = json.loads(prefix.splitlines()[-1])["available_evidence"]
        evidence = next(item for item in catalog if item["intent_id"] == observed["intent_id"])
        self.assertEqual(evidence["input_paths"], ["source.py"])
        self.journal.reload()
        self.session.branch(self.journal.records[0]["id"], "独立分支")
        with transport(self.base / "invisible-other-branch", [response("没有祖先证据")]) as wire:
            self.session.resume(max_turns=1)
        request = wire.call_args.args[0]
        self.assertNotIn(observed["intent_id"], json.dumps(request["messages"]))

    def test_next_request_refreshes_plan_inputs_after_a_tool_changes_a_file(self) -> None:
        self.journal = Journal.create(
            self.base / "changed-input.jsonl", self.work, tools=("read", "write", "plan")
        )
        self.session = AgentSession.open(self.journal.path)
        self.create()
        observed = self.read_result("source.py", "input-read")
        self.record_result(observed["intent_id"], 1)
        with transport(
            self.base / "changed-input",
            [
                response(
                    "",
                    calls=[call("write", {"path": "source.py", "content": "value = 9\n", "overwrite": True})],
                ),
                response("输入已变化，需要重新核对依据"),
            ],
        ) as wire:
            result = self.session.submit("更新输入后继续", max_turns=2)
        self.assertEqual(result["status"], "stop")
        self.assertEqual((self.work / "source.py").read_text(), "value = 9\n")
        request = wire.call_args_list[1].args[0]
        prefix = next(
            m["content"]
            for m in request["messages"]
            if m["role"] == "system" and "Current host workflow state" in m["content"]
        )
        workflow = json.loads(prefix.splitlines()[-1])
        self.assertTrue(all(not step["valid"] for step in workflow["plan"]["steps"]))
        self.assertIn("version_changed", workflow["plan"]["reason"])

    def test_evidence_catalog_is_bounded_without_erasing_old_receipts(self) -> None:
        self.create()
        names = [f"entry-{i:02d}-" + "x" * 170 + ".txt" for i in range(24)]
        for name in names:
            (self.work / name).write_text("observed")
        with transport(
            self.base / "many-evidences",
            [
                response(
                    "", calls=[call("read", {"path": name}, f"read-{i}") for i, name in enumerate(names)]
                ),
                response("保留历史记录"),
            ],
        ) as wire:
            result = self.session.submit("读取资料", max_turns=2)
        prefix = [
            m["content"]
            for m in wire.call_args.args[0]["messages"]
            if m["role"] == "system" and "available_evidence" in m["content"]
        ][0]
        payload = json.loads(prefix.splitlines()[-1])
        self.assertLessEqual(len(payload["available_evidence"]), 12)
        self.assertTrue(payload["evidence_catalog"]["truncated"])
        self.assertTrue(all(len(e["target"]) <= 160 for e in payload["available_evidence"]))
        receipts = [i for i in self.session.inspect()["intents"] if i["name"] == "read"]
        self.assertEqual(len(receipts), 24)
        self.assertEqual(len(result["transcript"][0]["results"]), 24)

    def test_execution_receipt_is_visible_beside_original_tool_output(self) -> None:
        self.create()
        with transport(
            self.base / "receipt-body",
            [response("", calls=[call("read", {"path": "source.py"})]), response("已读取")],
        ) as wire:
            result = self.session.submit("读取输入", max_turns=2)
        outcome = result["transcript"][0]["results"][0]
        message = next(
            m
            for m in wire.call_args.args[0]["messages"]
            if m["role"] == "tool" and m["tool_call_id"] == "plan-1" and "value = 3" in m["content"]
        )
        projected = json.loads(message["content"])
        self.assertEqual(projected["host_receipt"]["intent_id"], outcome["intent_id"])
        self.assertEqual(projected["host_receipt"]["observed_inputs"], ["source.py"])
        self.assertEqual(projected["tool_output"], "value = 3\n")
        self.assertEqual(outcome["text"], "value = 3\n")
        receipt = next(i for i in self.session.inspect()["intents"] if i["intent_id"] == outcome["intent_id"])
        self.assertEqual(receipt["result"]["text"], "value = 3\n")

    def test_missing_input_receipt_is_named_in_plan_diagnostic(self) -> None:
        definitions = steps()
        definitions[0]["inputs"] = ["source.py", "missing.py"]
        self.create(definitions)
        observed = self.read_result("source.py", "one-input")
        result = self.record_result(observed["intent_id"], 1)
        step = result["tasks"]["workflow"]["plan"]["steps"][0]
        self.assertFalse(step["valid"])
        self.assertIn("input_not_observed:missing.py", step["invalid_reason"])

    def test_old_input_snapshot_does_not_replace_explicit_record_evidence(self) -> None:
        self.read_result("source.py", "before-plan")
        self.create()
        (self.work / "other.txt").write_text("unrelated")
        observed = self.read_result("other.txt", "other-input")
        result = self.record_result(observed["intent_id"], 1)
        step = result["tasks"]["workflow"]["plan"]["steps"][0]
        self.assertFalse(step["valid"])
        self.assertIsNone(step["input_versions"]["source.py"])

    def test_extension_cannot_supply_host_model_content(self) -> None:
        extension = self.work / ".lite-pi/extensions/spoof/extension.py"
        extension.parent.mkdir(parents=True)
        extension.write_text(
            "def register(api):\n"
            "    api.register_tool('spoof', 'fixture', "
            "{'type':'object','properties':{},'additionalProperties':False}, "
            "lambda args: {'text':'actual output','model_content':123})\n"
        )
        with transport(
            self.base / "spoof-wire",
            [response("", calls=[call("spoof", {})]), response("done")],
        ) as wire:
            AgentSession(self.work, tools=()).submit("调用工具", max_turns=2)
        tool = next(m for m in wire.call_args.args[0]["messages"] if m["role"] == "tool")
        self.assertEqual(tool["content"], "actual output")

    def test_explicit_input_evidence_accepts_workspace_path_aliases(self) -> None:
        definitions = steps()
        definitions[0]["title"] = "根据 source.py 生成输出"
        definitions[0]["deliverable"] = "result.txt 已生成且输入可核对"
        journal = Journal.create(
            self.base / "input-binding.jsonl", self.work, tools=("read", "write", "plan")
        )
        self.session = AgentSession.open(journal.path)
        self.create(definitions)
        with transport(
            self.base / "observed-input",
            [
                response(
                    "",
                    calls=[
                        call("read", {"path": "./source.py"}, "source"),
                        call("write", {"path": "result.txt", "content": "value = 3"}, "output"),
                    ],
                )
            ],
        ):
            actual = self.session.submit("观察输入后写结果", max_turns=1)
        outcome = actual["transcript"][0]["results"][1]
        recorded = self.record_result(
            [actual["transcript"][0]["results"][0]["intent_id"], outcome["intent_id"]], 1
        )
        step = recorded["tasks"]["workflow"]["plan"]["steps"][0]
        self.assertTrue(step["valid"])
        self.assertIsNotNone(step["input_versions"]["source.py"])
        self.assertIn(
            actual["transcript"][0]["results"][0]["intent_id"], [c["intent_id"] for c in step["checks"]]
        )

    def test_old_session_migrates_and_future_schema_is_rejected_before_requests(self) -> None:
        import sqlite3

        from lite_pi.tasks import task_db_path

        journal = Journal.create(self.base / "legacy.jsonl", self.work, tools=("read",))
        journal.append("user", {"content": "历史问题"})
        journal.append("assistant", {"content": "历史答复"})
        with transport(self.base / "legacy-wire", [response("正常续接")]):
            resumed = AgentSession.open(journal.path).resume()
        self.assertIsNone(resumed["tasks"]["workflow"]["plan"])
        with sqlite3.connect(task_db_path(journal)) as db:
            db.execute("PRAGMA user_version=99")
        with transport(self.base / "future-wire", []) as wire:
            with self.assertRaisesRegex(ValueError, "unknown_task_schema"):
                AgentSession.open(journal.path).resume()
        self.assertEqual(wire.call_count, 0)


class PlanningPtyContracts(unittest.TestCase):
    def test_waiting_answer_and_cancellation_survive_terminal_restart(self) -> None:
        from uuid import uuid4

        from tui_pty import ROOT, Terminal

        from lite_pi.spend.ledger import Ledger

        base = ROOT / ".artifacts/planning/pty" / uuid4().hex
        work = base / "project"
        work.mkdir(parents=True)
        before = Ledger().status()["request_count"]
        terminal = Terminal(work, base / "first", controlled=True, extra=("--planning",), rows=36)
        try:
            terminal.wait(lambda: "Enter" in terminal.text)
            terminal.send("PLAN_QUESTION\r")
            terminal.wait(lambda: "等待答复" in terminal.text)
            self.assertIn("输出格式？", terminal.text)
            self.assertIn("1. JSON", terminal.text)
            terminal.capture("pending")
            terminal.send("\x04")
            terminal.wait(lambda: terminal.process.poll() is not None, 10)
        finally:
            terminal.close()
        paths = list((work / ".artifacts/sessions").glob("*.jsonl"))
        self.assertEqual(len(paths), 1)
        pending = AgentSession.open(paths[0]).inspect()["workflow"]["question"]
        self.assertEqual(pending["status"], "pending")
        terminal = Terminal(work, base / "second", controlled=True, session=paths[0], rows=36)
        try:
            terminal.wait(lambda: "等待答复" in terminal.text)
            terminal.send("1\r")
            terminal.wait(lambda: "本轮结束" in terminal.text)
            terminal.capture("answered")
        finally:
            terminal.close()
        answered = AgentSession.open(paths[0]).inspect()["workflow"]["question"]
        self.assertEqual(answered["answer"], "JSON")
        terminal = Terminal(work, base / "cancel", controlled=True, extra=("--planning",), rows=36)
        try:
            terminal.wait(lambda: "Enter" in terminal.text)
            terminal.send("PLAN_QUESTION\r")
            terminal.wait(lambda: "等待答复" in terminal.text)
            terminal.send("\x1b")
            terminal.wait(lambda: "澄清已取消" in terminal.text)
            terminal.capture("cancelled")
        finally:
            terminal.close()
        self.assertEqual(Ledger().status()["request_count"], before)


if __name__ == "__main__":
    unittest.main()
