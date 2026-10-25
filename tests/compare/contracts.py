"""Result contracts shared by the two comparison arms and their judge."""

from typing import TypedDict

from lite_pi.tools.contracts import JsonObject


class ArmObservation(TypedDict):
    arm: str
    elapsed: float
    resumed: bool
    checked: JsonObject
    before: dict[str, str]
    after: dict[str, str]
    state_passed: bool


class HandwrittenObservation(ArmObservation):
    results: list[JsonObject]
    model: str | None


class GraphObservation(ArmObservation):
    run_id: str
    receipt: JsonObject
    model_node_runs: int
    tool_node_runs: int
