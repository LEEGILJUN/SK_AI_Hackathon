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
from app.pipeline import DemoFactory, default_issue, run_pipeline


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
    ("lookup_mes", {}),
    ("run_inspection", {}),
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


def run(factory, **kwargs):
    """모델 없이 도는 기본 실행.

    이슈 원문에 제품명이 들어 있고, 언어 모델이 없으므로 양식 값이 쓰인다.
    실제 화면도 같은 조합이다.
    """
    kwargs.setdefault("adapters", (StubAdapter(), StubAdapter()))
    kwargs.setdefault("issue_text", default_issue(factory))
    kwargs.setdefault("context", {
        "line": "line_02", "object_name": "capsules",
        "defect_type": "dent", "product_id": factory.reported_product,
    })
    return run_pipeline(factory, **kwargs)


# ── 모델이 없을 때 ──────────────────────────────────────────────────────


def test_without_a_model_it_replays_a_fixed_order_and_says_so(factory):
    """모델이 없어도 끝까지 돌되, 순서를 모델이 정한 것처럼 보이면 안 된다."""
    outcome = run(factory, patch_override="defect")

    assert outcome.finished, "고정 순서로도 승인 요청까지 가야 한다"
    assert outcome.driver == "fallback"
    assert "고정 순서" in outcome.driver_note
    assert "모델이 정한 것이 아닙니다" in outcome.driver_note
    assert [name for name, _ in outcome.tool_trace] == [n for n, _ in FULL_PLAN]


# ── 모델이 붙었을 때 ────────────────────────────────────────────────────


def test_the_model_drives_the_tool_order(factory):
    outcome = run(factory, patch_override="defect",
                  adapters=(ScriptedLLM(FULL_PLAN), StubAdapter()))

    assert outcome.driver == "model"
    assert outcome.finished
    assert [name for name, _ in outcome.tool_trace] == [n for n, _ in FULL_PLAN]
    assert all(status == "성공" for _, status in outcome.tool_trace)


def test_out_of_order_calls_are_refused_with_a_reason(factory):
    """모델이 순서를 어기면 도구가 거부하고 무엇을 먼저 해야 하는지 알려준다.

    삼키고 진행하면 화면에 빈 칸이 생기고 왜 비었는지 알 수 없다.
    """
    outcome = run(factory, patch_override="defect",
                  adapters=(ScriptedLLM([("diagnose_issue", {}), ("plan_curation", {})]),
                            StubAdapter()))

    statuses = dict(outcome.tool_trace)
    assert "run_checks" in statuses["diagnose_issue"]
    assert "diagnose_issue" in statuses["plan_curation"]
    assert outcome.diagnosis is None
    assert not outcome.finished


def test_rebuild_is_refused_when_the_plan_forbids_it(factory):
    """재구성이 답이 아닌 원인에서 모델이 재구성을 불러도 실행되지 않는다.

    정량 목표 "재구성이 답이 아닌 케이스 전건 차단"이 여기 걸려 있다.
    """
    outcome = run(factory, patch_override="normal",   # 최근접 패치가 진짜 정상품
                  adapters=(ScriptedLLM(FULL_PLAN), StubAdapter()))

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
    outcome = run(factory, patch_override=None)

    assert outcome.intake is not None
    stage = next(s for s in outcome.stages if s.key == "evidence")
    item5 = next(label for label, _ in stage.rows if label.startswith("5."))
    value = dict(stage.rows)[item5]
    assert value.startswith("×"), "모델이 없으면 판별 5번은 근거로 쓰이지 않아야 한다"


# ── 라인·품목이 코드에 박혀 있지 않은가 ─────────────────────────────────


def test_context_comes_from_the_caller_not_the_code(factory):
    """양식에서 받은 라인·품목이 그대로 인테이크에 들어가야 한다."""
    outcome = run(factory, patch_override="defect",
                  context={"line": "line_07", "object_name": "pcb1",
                           "defect_type": "scratch", "product_id": "PCB1-07-x"})

    assert outcome.intake is not None
    assert outcome.intake.report.line == "line_07"
    assert outcome.intake.report.object_name == "pcb1"


# ── MES 조회와 품목별 뱅크 ──────────────────────────────────────────────


def test_bank_is_resolved_per_item_not_shared(factory):
    """품목마다 뱅크가 따로 있어야 한다.

    캡슐 정상 패치로 PCB 를 판정할 수 없다. 뱅크가 하나뿐이면 품목을 잘못
    고르는 실수가 드러나지 않는다.
    """
    versions = factory.bank_versions()
    assert len(set(versions.values())) == len(versions), "품목이 뱅크를 공유하고 있다"

    outcome = run(factory, patch_override="defect")
    stage = next(s for s in outcome.stages if s.key == "mes")
    used = dict(stage.rows)["품목 뱅크"]
    assert used == versions[("line_02", "capsules")]


def test_unknown_item_has_no_bank_and_stops(factory):
    """배포된 뱅크가 없는 품목이면 거기서 멈춘다.

    없는 품목에 다른 품목 뱅크를 물려 판정하면 결과가 전부 헛것이 된다.
    """
    outcome = run(factory, patch_override="defect",
                  context={"line": "line_09", "object_name": "없는품목",
                           "defect_type": "dent", "product_id": "X-1"})

    statuses = dict(outcome.tool_trace)
    assert "lookup_mes" in statuses
    assert "뱅크가 없다" in statuses["lookup_mes"]
    assert not outcome.finished


def test_mes_lookup_finds_images_from_the_product_name(factory):
    """이슈는 이미지가 아니라 제품명으로 온다. MES 가 이미지를 찾아야 한다."""
    outcome = run(factory, patch_override="defect")

    stage = next(s for s in outcome.stages if s.key == "mes")
    assert stage.status == "done"
    assert int(dict(stage.rows)["찾은 이미지"].rstrip("장")) > 0


def test_inspection_reports_missed_and_overkill(factory):
    """추론 결과가 미검·과검으로 갈려 화면에 떠야 한다. 사람은 보기만 한다."""
    outcome = run(factory, patch_override="defect")

    stage = next(s for s in outcome.stages if s.key == "inspect")
    assert "미검" in stage.headline and "과검" in stage.headline
    assert outcome.missed_records, "미검 건이 기록되어야 로트 집계가 된다"


# ── 리포트에 로트 집중도가 실리는가 ─────────────────────────────────────


def test_approval_document_reports_where_defects_cluster(factory):
    """결함이 한 로트에 몰려 있으면 승인 문서가 그것을 말해야 한다.

    자재나 설비 문제인데 뱅크부터 다시 만들면 증상만 덮는다. 승인하는 사람이
    그 판단을 하려면 집계가 문서에 있어야 한다.
    """
    outcome = run(factory, patch_override="defect")
    if not outcome.finished:
        pytest.skip("이 실행에서는 승인 문서까지 가지 않았다")

    assert outcome.distribution is not None
    assert outcome.distribution.total > 0

    document = outcome.approval_markdown
    assert "## 결함이 어디에 몰렸나" in document
    assert "## 대상 이미지" in document
    for record in outcome.missed_records[:1]:
        assert record.product_id in document
        assert (record.lot or "") in document


# ── 인테이크가 자연어를 먼저 보는가 ─────────────────────────────────────


def test_extraction_runs_before_the_form_is_consulted(factory):
    """언어 모델이 원문에서 뽑은 값이 양식보다 우선이다.

    양식이 다 채워져 있어도 모델이 뽑았으면 그쪽을 쓴다. 반대로 두면
    자연어 입력이 장식이 되고 언어 모델을 쓰는 의미가 사라진다.
    """
    extracted = '{"line": "line_03", "object_name": "macaroni1", "defect_type": "crack"}'
    outcome = run(
        factory, patch_override="defect",
        adapters=(ScriptedLLM(FULL_PLAN, extraction=extracted), StubAdapter()),
        context={"line": "line_02", "object_name": "capsules",
                 "defect_type": "dent", "product_id": factory.reported_product},
    )

    assert outcome.intake is not None
    assert outcome.intake.report.line == "line_03", "양식이 원문 추출을 덮어썼다"
    assert outcome.intake.report.object_name == "macaroni1"


def test_product_name_alone_is_enough_to_proceed(factory):
    """이미지 첨부가 없어도 제품명이 있으면 진행한다.

    현장 이슈는 "이 로트가 계속 빠집니다"로 오지 이미지를 첨부해서 오지
    않는다. MES 가 이미지를 찾아 주므로 첨부를 요구할 이유가 없다.
    """
    outcome = run(factory, patch_override="defect")

    assert outcome.intake is not None
    assert outcome.intake.verdict == "proceed"
    assert not outcome.intake.report.attachments, "첨부 없이 진행한 경로를 재고 있다"


# ── 라인 시뮬레이터 ─────────────────────────────────────────────────────


def test_shadow_keeps_every_case_not_only_disagreements(factory):
    """시뮬레이터가 흘려보내려면 통과한 것도 남아 있어야 한다.

    갈린 것만 남기면 "몇 장을 어떻게 통과시켰는가"를 보여줄 수 없다.
    """
    outcome = run(factory, patch_override="defect")
    if outcome.shadow is None:
        pytest.skip("이 실행에서는 섀도까지 가지 않았다")

    shadow = outcome.shadow
    assert len(shadow.cases) == shadow.total
    assert sum(1 for c in shadow.cases if c.agreed) == shadow.agreed
    assert sum(1 for c in shadow.cases if not c.agreed) == shadow.review_count


def test_simulator_replays_real_numbers_not_invented_ones(factory):
    """시뮬레이터의 집계가 섀도 결과와 정확히 같아야 한다.

    화면에서 숫자가 따로 놀면 보기 좋은 애니메이션일 뿐 근거가 아니다.
    """
    from app.view import render_page

    outcome = run(factory, patch_override="defect")
    if outcome.shadow is None:
        pytest.skip("이 실행에서는 섀도까지 가지 않았다")

    html = render_page(outcome, outcome.issue_text)
    assert "코어셋 검증 — 가상 라인" in html
    assert "코어셋 검증 중입니다" in html
    # 흘려보내는 자료가 섀도 사례 전부여야 한다.
    for case in outcome.shadow.cases:
        assert f"/image/{case.image}" in html


def test_simulator_is_absent_when_there_was_no_shadow_run(factory):
    """섀도까지 못 갔으면 시뮬레이터도 없어야 한다. 빈 벨트를 띄우지 않는다."""
    from app.view import render_page

    outcome = run(factory, patch_override="defect",
                  adapters=(ScriptedLLM([("intake_issue", {})]), StubAdapter()))
    html = render_page(outcome, outcome.issue_text)
    assert outcome.shadow is None
    assert "코어셋 검증 — 가상 라인" not in html
