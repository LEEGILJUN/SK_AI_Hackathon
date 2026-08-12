"""도구 호출 계층 검증 — 프롬프트에서 재구성까지 (작업 16).

이 과제의 주안점은 "프롬프트로 재학습 행위가 되느냐"다. 그래서 확인할 것이
두 갈래다.

  1. 명령이 실제 실행으로 이어지는가
  2. **언어 모델이 시켜도 하면 안 되는 일은 안 되는가**

2번이 더 중요하다. 언어 모델은 틀릴 수 있고, 도구 계층이 마지막 방어선이다.
재구성이 답이 아닌 원인에서 모델이 재구성을 부르더라도 실행되면 안 된다.
"""

from __future__ import annotations

import json

import pytest

from agents.adapters.base import ToolCall
from agents.adapters.stub import StubAdapter
from agents.curate import CurationPlan, RemovalCandidate, plan_curation
from agents.diagnose import DiagnosisResult
from inspection.types import InferenceResult, NearestMatch, PatchRef
from agents.tools import (
    COMPARE_SPEC,
    DIAGNOSE_SPEC,
    PLAN_SPEC,
    REBUILD_SPEC,
    Tool,
    ToolRegistry,
    run_agent,
)


class ScriptedAdapter(StubAdapter):
    """실제 모델을 흉내 내는 테스트용 어댑터.

    StubAdapter 는 항상 is_stub=True 를 붙인다. 그 성질이 안전 장치이므로
    (스텁 응답은 근거가 될 수 없다) 건드리지 않고, 에이전트 루프를 돌리기
    위한 어댑터를 따로 둔다.
    """

    is_stub = False

    def chat(self, *args, **kwargs):
        response = super().chat(*args, **kwargs)
        response.is_stub = False
        return response


def _missed(bank_image: str) -> InferenceResult:
    """오염 후보를 만들기 위한 미검출 추론 결과."""
    match = NearestMatch(
        query=PatchRef("q.png", 1, 1, 9),
        bank=PatchRef(bank_image, 2, 2, 18),
        distance=2.0,
        bank_row_index=7,
    )
    return InferenceResult(
        image="q.png", score=1.5, max_patch_distance=1.5,
        grid_h=8, grid_w=8, matches=[match], bank_version="v3",
    )


def make_registry(record: dict) -> ToolRegistry:
    """진단 → 계획 → 재구성 흐름을 흉내 내는 도구 모음."""

    def diagnose_issue(line: str, object_name: str):
        record["diagnosed"] = (line, object_name)
        result = DiagnosisResult(
            cause=record["cause"],
            requires_bank_rebuild=record["rebuild"],
            confidence="high",
            needs_human=False,
        )
        record["diagnosis"] = result
        return result.to_dict()

    def plan_curation_tool():
        diagnosis = record.get("diagnosis")
        if diagnosis is None:
            raise RuntimeError("진단을 먼저 실행해야 한다")
        # 오염이면 역추적이 지목한 건이 있다고 가정한다. 실제 차단 로직을
        # 그대로 태우기 위해 계획을 손으로 만들지 않는다.
        plan = plan_curation(diagnosis, missed_results=[_missed("d0.png")] * 2)
        record["plan"] = plan
        return plan.to_dict()

    def rebuild_bank(confirm: bool):
        plan: CurationPlan | None = record.get("plan")
        if plan is None:
            raise RuntimeError("계획을 먼저 세워야 한다")
        if not confirm:
            return {"executed": False, "reason": "confirm 이 false 다"}
        if not plan.touches_bank:
            # 마지막 방어선. 모델이 불러도 여기서 막힌다.
            return {"executed": False, "reason": f"계획이 뱅크를 건드리지 않는다. {plan.reason}"}
        record["rebuilt"] = True
        return {"executed": True, "to_version": "v4"}

    return ToolRegistry(
        [
            Tool(DIAGNOSE_SPEC, diagnose_issue),
            Tool(PLAN_SPEC, plan_curation_tool),
            Tool(REBUILD_SPEC, rebuild_bank, mutates_bank=True),
        ]
    )


# ── 도구 실행 ──────────────────────────────────────────────────────────


def test_registry_executes_and_records():
    record = {"cause": "bank_contamination", "rebuild": True}
    registry = make_registry(record)

    result = registry.execute(
        ToolCall(id="1", name="diagnose_issue", arguments={"line": "line_02", "object_name": "capsules"})
    )

    assert result.ok is True
    assert result.output["cause"] == "bank_contamination"
    assert record["diagnosed"] == ("line_02", "capsules")
    assert len(registry.calls) == 1


def test_unknown_tool_is_reported_not_raised():
    """없는 도구를 불러도 죽지 않아야 한다. 모델이 고쳐 부를 수 있게."""
    registry = make_registry({"cause": "threshold", "rebuild": False})
    result = registry.execute(ToolCall(id="1", name="없는도구", arguments={}))

    assert result.ok is False
    assert "없는도구" in result.error
    assert "diagnose_issue" in result.error


def test_bad_arguments_are_reported_not_raised():
    registry = make_registry({"cause": "threshold", "rebuild": False})
    result = registry.execute(ToolCall(id="1", name="diagnose_issue", arguments={"엉뚱한인자": 1}))

    assert result.ok is False
    assert "인자가 맞지 않다" in result.error


def test_tool_result_serializes_for_the_model():
    registry = make_registry({"cause": "bank_contamination", "rebuild": True})
    result = registry.execute(
        ToolCall(id="abc", name="diagnose_issue", arguments={"line": "l", "object_name": "o"})
    )
    message = result.to_message("abc")

    assert message.role == "tool"
    assert message.tool_call_id == "abc"
    payload = json.loads(message.content)
    assert payload["ok"] is True


def test_mutating_tools_are_tracked():
    """뱅크를 바꾼 호출만 따로 셀 수 있어야 한다. 승인 요청에 실을 목록이다."""
    record = {"cause": "bank_contamination", "rebuild": True}
    registry = make_registry(record)

    for name, args in (
        ("diagnose_issue", {"line": "l", "object_name": "o"}),
        ("plan_curation", {}),
        ("rebuild_bank", {"confirm": True}),
    ):
        registry.execute(ToolCall(id=name, name=name, arguments=args))

    mutations = registry.mutations()
    assert len(mutations) == 1
    assert mutations[0].name == "rebuild_bank"


# ── 시켜도 하면 안 되는 일 ─────────────────────────────────────────────


def test_rebuild_is_refused_when_plan_blocks_it():
    """재구성이 답이 아닌 원인에서 모델이 재구성을 불러도 실행되면 안 된다.

    이것이 이 파일의 핵심이다. 언어 모델은 틀릴 수 있고 도구 계층이
    마지막 방어선이다.
    """
    record = {"cause": "normal_overlap", "rebuild": False}
    registry = make_registry(record)

    registry.execute(ToolCall(id="1", name="diagnose_issue", arguments={"line": "l", "object_name": "o"}))
    registry.execute(ToolCall(id="2", name="plan_curation", arguments={}))
    result = registry.execute(ToolCall(id="3", name="rebuild_bank", arguments={"confirm": True}))

    assert result.output["executed"] is False
    assert record.get("rebuilt") is None, "정상 분포 중첩인데 뱅크가 재구성됐다"


def test_rebuild_requires_a_plan_first():
    """순서를 건너뛰면 막힌다."""
    registry = make_registry({"cause": "bank_contamination", "rebuild": True})
    result = registry.execute(ToolCall(id="1", name="rebuild_bank", arguments={"confirm": True}))

    assert result.ok is False
    assert "계획을 먼저" in result.error


def test_confirm_false_does_not_execute():
    record = {"cause": "bank_contamination", "rebuild": True}
    registry = make_registry(record)
    registry.execute(ToolCall(id="1", name="diagnose_issue", arguments={"line": "l", "object_name": "o"}))
    registry.execute(ToolCall(id="2", name="plan_curation", arguments={}))
    result = registry.execute(ToolCall(id="3", name="rebuild_bank", arguments={"confirm": False}))

    assert result.output["executed"] is False
    assert record.get("rebuilt") is None


# ── 에이전트 루프 ──────────────────────────────────────────────────────


def test_agent_runs_tools_in_order_from_a_prompt():
    """자연어 명령이 진단 → 계획 → 재구성으로 이어져야 한다."""
    record = {"cause": "bank_contamination", "rebuild": True}
    registry = make_registry(record)

    adapter = ScriptedAdapter(
        scripted=["", "", "", "뱅크 오염으로 판단해 재구성했습니다."],
        tool_calls=[
            [ToolCall("1", "diagnose_issue", {"line": "line_02", "object_name": "capsules"})],
            [ToolCall("2", "plan_curation", {})],
            [ToolCall("3", "rebuild_bank", {"confirm": True})],
            [],
        ],
    )

    run = run_agent("2라인 캡슐 뱅크 다시 만들어줘", adapter, registry)

    assert [r.name for r in run.tool_results] == ["diagnose_issue", "plan_curation", "rebuild_bank"]
    assert record["rebuilt"] is True
    assert "재구성했습니다" in run.final_text


def test_agent_stops_when_model_is_stub():
    """모델이 없으면 도구를 부르지 않는다. 스텁이 재구성을 시키면 안 된다."""
    record = {"cause": "bank_contamination", "rebuild": True}
    registry = make_registry(record)

    run = run_agent("뱅크 다시 만들어줘", StubAdapter(), registry)

    assert run.tool_results == []
    assert record.get("rebuilt") is None
    assert "연결되지 않아" in run.stopped_reason


def test_agent_stops_at_max_steps():
    """같은 도구를 반복해 부르며 끝내지 못하는 경우를 막는다."""
    registry = make_registry({"cause": "bank_contamination", "rebuild": True})
    adapter = ScriptedAdapter(
        scripted=[""] * 20,
        tool_calls=[[ToolCall(str(i), "diagnose_issue", {"line": "l", "object_name": "o"})] for i in range(20)],
    )

    run = run_agent("돌려줘", adapter, registry, max_steps=3)

    assert run.steps == 3
    assert "넘겨 중단" in run.stopped_reason


def test_agent_survives_model_failure():
    class Broken(ScriptedAdapter):
        def chat(self, *args, **kwargs):
            raise ConnectionError("서버 없음")

    run = run_agent("돌려줘", Broken(), make_registry({"cause": "threshold", "rebuild": False}))

    assert "모델 호출 실패" in run.stopped_reason
    assert run.tool_results == []
