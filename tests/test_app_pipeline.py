"""웹 데모가 도구 호출 경로로 도는지 — 작업 11.

예전 구현은 함수를 순서대로 불렀다. 그러면 화면이 보여주는 것은 "에이전트가
판단하는 것"이 아니라 "파이프라인이 도는 것"이다. 여기서 지키려는 것은 셋이다.

  1. 언어 모델이 붙으면 **모델이 도구 순서를 정한다**
  2. 모델이 없으면 같은 도구를 고정 순서로 재생하되 **그렇다고 표시한다**
  3. 모델이 순서를 어기거나 금지된 조치를 부르면 **도구가 거부한다**

3번이 가장 중요하다. 언어 모델이 원인을 정하거나 재구성을 강행할 수 있으면
진단이 인상 평가가 된다.
"""

from __future__ import annotations

import pytest

from agents.adapters.base import ChatResponse, ModelAdapter, ToolCall
from agents.adapters.stub import StubAdapter
from app.pipeline import DemoFactory, run_pipeline


class ScriptedLLM(ModelAdapter):
    """도구 순서를 정하는 어댑터.

    도구 목록이 없는 호출(인테이크의 항목 추출 등)에는 JSON 을 돌려주고,
    도구 목록이 있는 호출에만 계획을 소비한다. 하나로 섞으면 추출 호출이
    계획을 한 칸씩 갉아먹어 무엇을 재는지 알 수 없게 된다.
    """

    is_stub = False

    def __init__(self, plan, extraction: str = "{}"):
        self.plan = list(plan)
        self.extraction = extraction
        self.index = 0

    def describe(self) -> str:
        return "scripted-llm"

    def chat(self, messages, tools=None, **kwargs) -> ChatResponse:
        if not tools:
            return ChatResponse(text=self.extraction, is_stub=False, model="scripted-llm")
        if self.index < len(self.plan):
            name, arguments = self.plan[self.index]
            self.index += 1
            return ChatResponse(
                text="",
                tool_calls=[ToolCall(id=f"call-{self.index}", name=name, arguments=dict(arguments))],
                is_stub=False,
                model="scripted-llm",
            )
        return ChatResponse(text="마쳤습니다. 배포 여부는 사람이 정합니다.",
                            is_stub=False, model="scripted-llm")


FULL_PLAN = [
    ("intake_issue", {}),
    ("run_checks", {}),
    ("diagnose_issue", {}),
    ("plan_curation", {}),
    ("rebuild_bank", {"confirm": True}),
    ("evaluate_gate", {}),
    ("shadow_compare", {}),
    ("prepare_release", {}),
]


@pytest.fixture(scope="module")
def factory():
    try:
        return DemoFactory()
    except RuntimeError as exc:  # 모델 가중치를 못 받는 환경
        pytest.skip(str(exc))


# ── 모델이 없을 때 ──────────────────────────────────────────────────────


def test_without_a_model_it_replays_a_fixed_order_and_says_so(factory):
    """모델이 없어도 끝까지 돌되, 순서를 모델이 정한 것처럼 보이면 안 된다."""
    outcome = run_pipeline(factory, patch_override="defect",
                           adapters=(StubAdapter(), StubAdapter()))

    assert outcome.finished, "고정 순서로도 승인 요청까지 가야 한다"
    assert outcome.driver == "fallback"
    assert "고정 순서" in outcome.driver_note
    assert "모델이 정한 것이 아닙니다" in outcome.driver_note
    assert [name for name, _ in outcome.tool_trace] == [n for n, _ in FULL_PLAN]


# ── 모델이 붙었을 때 ────────────────────────────────────────────────────


def test_the_model_drives_the_tool_order(factory):
    outcome = run_pipeline(
        factory, patch_override="defect",
        adapters=(ScriptedLLM(FULL_PLAN), StubAdapter()),
    )

    assert outcome.driver == "model"
    assert outcome.finished
    assert [name for name, _ in outcome.tool_trace] == [n for n, _ in FULL_PLAN]
    assert all(status == "성공" for _, status in outcome.tool_trace)


def test_out_of_order_calls_are_refused_with_a_reason(factory):
    """모델이 순서를 어기면 도구가 거부하고 무엇을 먼저 해야 하는지 알려준다.

    삼키고 진행하면 화면에 빈 칸이 생기고 왜 비었는지 알 수 없다.
    """
    outcome = run_pipeline(
        factory, patch_override="defect",
        adapters=(ScriptedLLM([("diagnose_issue", {}), ("plan_curation", {})]), StubAdapter()),
    )

    statuses = dict(outcome.tool_trace)
    assert "run_checks" in statuses["diagnose_issue"]
    assert "diagnose_issue" in statuses["plan_curation"]
    assert outcome.diagnosis is None
    assert not outcome.finished


def test_rebuild_is_refused_when_the_plan_forbids_it(factory):
    """재구성이 답이 아닌 원인에서 모델이 재구성을 불러도 실행되지 않는다.

    정량 목표 "재구성이 답이 아닌 케이스 전건 차단"이 여기 걸려 있다.
    """
    outcome = run_pipeline(
        factory, patch_override="normal",   # 최근접 패치가 진짜 정상품
        adapters=(ScriptedLLM(FULL_PLAN), StubAdapter()),
    )

    assert outcome.plan is not None
    if outcome.plan.touches_bank:
        pytest.skip("이 시연 데이터에서는 뱅크를 건드리는 원인이 나왔다")

    assert outcome.rebuild is None or not outcome.rebuild.executed
    assert not outcome.finished, "재구성이 막히면 승인 요청까지 가지 않는다"


# ── 판별 5번을 모델에 묻는 경로 ─────────────────────────────────────────


def test_asking_the_model_for_the_patch_verdict_withholds_when_unconnected(factory):
    """patch_override 가 없으면 모델에 묻고, 모델이 없으면 판정을 보류한다.

    독스트링만 그렇게 적혀 있고 분기가 없던 자리다. 지어낸 답이 근거로
    올라가면 안 되므로 보류가 옳은 동작이다.
    """
    outcome = run_pipeline(factory, patch_override=None,
                           adapters=(StubAdapter(), StubAdapter()))

    assert outcome.intake is not None
    stage = next(s for s in outcome.stages if s.key == "evidence")
    item5 = next(label for label, _ in stage.rows if label.startswith("5."))
    value = dict(stage.rows)[item5]
    assert value.startswith("×"), "모델이 없으면 판별 5번은 근거로 쓰이지 않아야 한다"


# ── 라인·품목이 코드에 박혀 있지 않은가 ─────────────────────────────────


def test_context_comes_from_the_caller_not_the_code(factory):
    """양식에서 받은 라인·품목이 그대로 인테이크에 들어가야 한다."""
    outcome = run_pipeline(
        factory, patch_override="defect",
        adapters=(StubAdapter(), StubAdapter()),
        context={"line": "line_07", "object_name": "pcb1", "defect_type": "scratch"},
    )

    assert outcome.intake is not None
    assert outcome.intake.report.line == "line_07"
    assert outcome.intake.report.object_name == "pcb1"
