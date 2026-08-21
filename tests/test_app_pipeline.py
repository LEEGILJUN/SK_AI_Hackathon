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

from pathlib import Path

import pytest

from agents.adapters.base import ChatResponse, ModelAdapter, ToolCall
from agents.adapters.stub import StubAdapter
from app.pipeline import (
    CONTAMINATED_ITEM,
    COVERAGE_ITEM,
    DemoFactory,
    default_issue,
    run_pipeline,
)


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
    """합성 이미지로 고정한다.

    `DemoFactory()` 를 그냥 부르면 저장소 아래 `VisA_20220922/` 를 찾아 실데이터로
    선다. 그러면 **VisA 원본을 받아 둔 기계에서만 테스트가 달라진다** — 뱅크를
    실제로 세우느라 8초가 11분이 되고, 합성 전제로 적어 둔 단언들이 깨진다.
    실데이터 경로는 `test_visa_*` 가 가짜 트리로 따로 검사한다.
    """
    try:
        return DemoFactory(visa_root=Path(__file__).parent / "_no_visa_here")
    except RuntimeError as exc:  # 모델 가중치를 못 받는 환경
        pytest.skip(str(exc))


def run(factory, **kwargs):
    """모델 없이 도는 기본 실행.

    이슈 원문에 제품명이 들어 있고, 언어 모델이 없으므로 양식 값이 쓰인다.
    실제 화면도 같은 조합이다.
    """
    kwargs.setdefault("adapters", (StubAdapter(), StubAdapter()))
    kwargs.setdefault("issue_text", default_issue(factory))
    line, object_name = CONTAMINATED_ITEM
    kwargs.setdefault("context", {
        "line": line, "object_name": object_name,
        "defect_type": "scratch", "product_id": factory.reported_product,
    })
    return run_pipeline(factory, **kwargs)


def run_line(factory, key, **kwargs):
    """지정한 라인의 이슈로 돌린다.

    **라인마다 심은 것이 다르다.** 기본 실행은 오염 라인 하나만 보므로,
    다른 라인이 실제로 다른 원인을 내는지는 이쪽으로 본다. 조회 계층도
    화면이 만드는 것과 같게 만든다 — 여기서 상수를 쓰면 시험이 통과해도
    화면에서는 다른 답이 나온다.
    """
    from app.pipeline import DEMO_THRESHOLDS
    from lookup import MockLookup

    line, object_name = key
    item = factory.items[key]
    product = factory._product_id(line, object_name, item.holdout_defect[0])
    number = line.split("_")[-1].lstrip("0") or "0"

    kwargs.setdefault("adapters", (StubAdapter(), StubAdapter()))
    kwargs.setdefault(
        "issue_text",
        f"{number}라인 기판이 검사에서 계속 양품으로 나옵니다. 제품 {product} 건입니다.",
    )
    kwargs.setdefault("context", {
        "line": line, "object_name": object_name,
        # 해결 이력에 있는 결함 유형을 쓰면 중복으로 끊긴다. 그것도 옳은
        # 동작이라 여기서는 이력에 없는 유형으로 둔다.
        "defect_type": "burnt", "product_id": product,
    })
    kwargs.setdefault("lookup", MockLookup(
        threshold=2.2, catalog=factory.catalog, banks=factory.bank_versions(),
        quality_provider=factory.quality_baseline,
        bank_profiles=factory.bank_profiles(), thresholds=DEMO_THRESHOLDS,
    ))
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


# ── 온톨로지 조회는 판정에 손대지 못한다 ────────────────────────────────


def test_the_model_can_read_the_cause_taxonomy_without_changing_the_verdict(factory):
    """모델이 온톨로지를 읽어도 원인은 그대로 규칙이 낸다.

    조회 결과에 여섯 원인의 정의가 다 들어 있으므로, 이것이 판정에 새면
    모델이 "설명이 그럴듯한 원인"을 고를 수 있게 된다. 순서 제약도 없어야
    한다 — 정의를 묻는 데 앞 단계가 필요할 이유가 없다.
    """
    with_lookup = [
        ("lookup_ontology", {}),                       # 아무것도 하기 전에 물어도 된다
        *FULL_PLAN[:4],
        ("lookup_ontology", {"cause": "normal_overlap"}),
        *FULL_PLAN[4:],
    ]
    read = run(factory, patch_override="defect",
               adapters=(ScriptedLLM(with_lookup), StubAdapter()))
    plain = run(factory, patch_override="defect",
                adapters=(ScriptedLLM(FULL_PLAN), StubAdapter()))

    statuses = dict(read.tool_trace)
    assert statuses["lookup_ontology"] == "성공"
    assert read.diagnosis is not None and plain.diagnosis is not None
    assert read.diagnosis.cause == plain.diagnosis.cause, (
        "온톨로지를 읽었다고 원인이 달라지면 진단이 유사도 맞히기가 된다"
    )
    assert read.diagnosis.requires_bank_rebuild == plain.diagnosis.requires_bank_rebuild


def test_the_fixed_replay_does_not_fake_an_ontology_lookup(factory):
    """모델이 없으면 온톨로지를 부르지 않는다.

    물어볼 주체가 없는데 재생 목록에 끼워 두면, 아무도 읽지 않은 조회가
    화면에 남아 모델이 확인한 것처럼 보인다.
    """
    outcome = run(factory, patch_override="defect")
    assert "lookup_ontology" not in dict(outcome.tool_trace)


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
    assert used == versions[CONTAMINATED_ITEM]


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
    assert "결함이 어디에 몰렸나" in document
    assert "대상 이미지" in document
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
    # **문구가 아니라 블록으로 잡는다.** 전에 제목 문자열을 그대로 걸어
    # 두었더니 화면 문안을 다듬는 것이 시험을 깨는 일이 되어, 고쳐야 할
    # 문장을 못 고치고 그대로 뒀다. 확인할 것은 "섀도 재생 블록이 그려지는가"
    # 이지 제목이 무엇인가가 아니다.
    assert 'id="block-simulator"' in html
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
    # **부정 단언이라 더 조심해야 한다.** 문구로 잡아 두면 제목을 다듬는
    # 순간 단언이 항상 참이 되어, 시뮬레이터가 떠도 시험이 통과한다.
    # 379행 짝과 같은 것을 보게 블록으로 잡는다.
    assert 'id="block-simulator"' not in html


# ── 온톨로지 · 진단 근거 · 조회 방식 ────────────────────────────────────


def test_issue_graph_returns_the_path_not_just_a_score(factory):
    """유사도 숫자만 돌려주면 왜 비슷한지 검증할 수 없다.

    그래프 검색이 블랙박스가 되면 중복 차단이라는 역할도 못 맡긴다.
    """
    from lookup import MockLookup

    found = MockLookup().find_similar_issues("line_02", "pcb1", "scratch")
    assert found, "이슈 이력 그래프가 비어 있다"

    top = found[0]
    assert top.path, "도달 경로가 없다"
    assert top.matched_on, "무엇이 겹쳐서 비슷하다고 봤는지가 없다"
    relations = {edge.relation for edge in top.path}
    assert {"진단_원인", "조치", "결과"} <= relations, (
        f"원인→조치→결과 사슬이 끊겼다: {relations}"
    )
    assert found == sorted(found, key=lambda i: -i.similarity), "유사도 순이 아니다"


def test_a_different_line_is_related_not_duplicate(factory):
    """다른 라인의 같은 증상은 중복이 아니다.

    라인마다 뱅크가 따로이므로 1라인 뱅크가 오염됐다고 2라인도 그렇다는 뜻이
    아니다. 유사도만 보고 끊으면 실제로 있는 문제를 "이미 해결된 건"으로 덮는다.
    """
    outcome = run(factory, patch_override="defect")

    assert outcome.intake is not None
    assert outcome.intake.similar, "유사 사례가 넘어오지 않았다"

    cross_line = [i for i in outcome.intake.similar
                  if i.line != outcome.intake.report.line and i.resolved]
    assert cross_line, "다른 라인 사례가 없어 이 규칙을 재지 못한다"
    assert max(i.similarity for i in cross_line) >= 0.8, "유사도가 낮으면 시험이 안 된다"

    assert outcome.intake.verdict == "proceed", "다른 라인 사례로 끊었다"
    assert outcome.diagnosis is not None and outcome.diagnosis.cause is not None


def test_retrieval_trace_labels_each_lookup_by_actual_mechanism(factory):
    """조회를 전부 "RAG"로 뭉뚱그리면 화면이 거짓말을 한다.

    이 구분 자체가 과제의 논거다 — 진단의 신뢰도는 벡터 검색이 아니라
    결정론적 조회에서 나온다.
    """
    outcome = run(factory, patch_override="defect")

    assert outcome.retrievals, "조회 기록이 비어 있다"
    kinds = {c["kind"] for c in outcome.retrievals}
    assert "unknown" not in kinds, (
        f"RETRIEVAL_KIND 에 등록 안 된 조회가 있다: "
        f"{[c['name'] for c in outcome.retrievals if c['kind'] == 'unknown']}"
    )

    joins = [c for c in outcome.retrievals if c["kind"] == "join"]
    graphs = [c for c in outcome.retrievals if c["kind"] == "graph"]
    assert len(joins) > len(graphs), "조인이 그래프보다 많아야 한다"
    assert all(c["name"] == "find_similar_issues" for c in graphs), (
        "이슈 이력 말고 다른 것이 그래프 검색으로 표시됐다"
    )


def test_diagnosis_panel_draws_the_traced_pair(factory):
    """역추적한 두 자리를 실제로 잘라 보여줘야 한다.

    "격자(6,5), 거리 0.0059" 라고 글로만 적으면 확인할 방법이 없다.
    """
    from app.view import render_page

    outcome = run(factory, patch_override="defect")
    if outcome.inference is None:
        pytest.skip("이 실행에서는 추론까지 가지 않았다")

    html = render_page(outcome, outcome.issue_text)
    assert "이상 점수 히트맵" in html
    assert "이슈 이력 그래프" in html
    assert 'id="block-retrieval"' in html  # 제목 문구가 아니라 블록으로

    top = outcome.inference.top_match
    grid_h, grid_w = outcome.grid
    # 두 크롭이 같은 격자 기준으로 걸려야 한다. 화면이 좌표를 따로 계산하면
    # 두 벌이 되고 한쪽만 틀어져 엉뚱한 자리를 자른다.
    assert f"row={top.query.row}&amp;col={top.query.col}" in html
    assert f"row={top.bank.row}&amp;col={top.bank.col}" in html
    assert f"grid_h={grid_h}&amp;grid_w={grid_w}" in html


# ── 전 구간 화면의 구조 ────────────────────────────────────────────────
#
# 아래 다섯은 **백본 가중치 없이도 돈다.** 위쪽 테스트들은 뱅크를 만들어야 해서
# 가중치가 캐시에 없으면 통째로 skip 되는데, 그 상태에서 화면 코드가 깨져도
# 아무도 모른다. 실제로 view.py 가 Python 3.11 에서 import 조차 안 되던 적이
# 있었고 이 조합 때문에 테스트가 못 잡았다.


def _stage_outcome():
    """가중치 없이 만들 수 있는 최소 결과. 화면 구조만 보기 위한 것이다."""
    from app.pipeline import RunOutcome, Stage

    return RunOutcome(
        issue_text="2라인 캡슐 표면 찍힘이 계속 빠집니다.",
        stages=[
            Stage(key="intake", title="1. 인테이크", status="done", rows=[("판정", "넘김")]),
            Stage(key="mes", title="2. MES 조회", status="done", rows=[("찾은 이미지", "14장")]),
            Stage(key="evidence", title="4. 판별 7항목", status="done",
                  rows=[("1. defect_visible", "×  unknown"), ("2. quality", "○  True")]),
            Stage(key="diagnose", title="5. 진단", status="done", rows=[("확신도", "high")]),
            Stage(key="release", title="10. 승인 요청", status="done", rows=[("패키지", "생성")]),
        ],
        approval_markdown="# 승인 요청",
    )


def test_navigation_points_at_sections_that_exist():
    """이동 바의 링크가 전부 실제 절을 가리켜야 한다.

    누르면 아무 데도 안 가는 링크는 시연 중에 그대로 드러난다.
    """
    import re

    from app.view import render_page

    html = render_page(_stage_outcome(), "이슈")
    nav = re.search(r'<nav class="nav">(.*?)</nav>', html, re.S)
    assert nav, "이동 바가 없다"

    anchors = re.findall(r'href="#([\w-]+)"', nav.group(1))
    ids = set(re.findall(r'id="([\w-]+)"', html))

    assert anchors, "이동 바에 링크가 하나도 없다"
    assert [a for a in anchors if a not in ids] == []
    assert "stage-diagnose" in anchors
    assert "doc-approval" in anchors


def test_key_stages_stay_open_and_others_collapse():
    """판별 항목과 진단은 접지 않는다. 근거 자체라 접으면 볼 것이 없다."""
    import re

    from app.view import render_page

    html = render_page(_stage_outcome(), "이슈")

    for key in ("evidence", "diagnose"):
        section = re.search(rf'id="stage-{key}".*?</section>', html, re.S).group(0)
        assert "<details>" not in section, f"{key} 단계가 접혔다"

    # 부차 단계는 접는다. 전 구간 화면이 세로로 너무 길어 시연에서 이동이 안 된다.
    mes = re.search(r'id="stage-mes".*?</section>', html, re.S).group(0)
    assert "<details>" in mes


def test_first_screen_has_no_navigation_bar():
    """결과가 없으면 이동할 자리도 없다. 빈 바를 띄우지 않는다."""
    from app.view import render_page

    assert '<nav class="nav">' not in render_page(None, "이슈 원문")


def test_nearest_patch_reads_as_a_sentence():
    """판별 4번이 파이썬 딕셔너리 원문으로 나오면 안 된다.

    진단의 핵심 근거인데 화면에서 가장 안 읽히는 자리가 된다.
    """
    from app.pipeline import _evidence_value

    text = _evidence_value({
        "source_image": "line_02/capsules/normal/normal_007.png",
        "row": 3, "col": 0, "bank_row_index": 47, "distance": 0.5697905421257019,
    })

    assert text == "normal_007.png · 격자 (3,0) · 거리 0.5698"
    assert "{" not in text


def test_unchecked_item_is_marked_apart():
    """확인하지 못한 항목이 눈에 띄어야 한다. 그리고 본문은 여전히 escape 된다."""
    from app.view import _mark

    assert 'class="mark no"' in _mark("×  unknown")
    assert 'class="mark yes"' in _mark("○  True")
    assert "&lt;script&gt;" in _mark("○  <script>")
    assert "<script>" not in _mark("○  <script>")


# ── VisA 실데이터로 서는 경로 ──────────────────────────────────────────
#
# 실제로 뱅크를 만들어 보는 것은 VisA 원본과 백본 가중치가 있는 기계의 몫이다.
# 여기서는 **어느 쪽으로 설지 고르는 판단**과 그 판단이 화면에 드러나는지만
# 본다. 둘 다 원본 없이 확인할 수 있고, 틀리면 조용히 무의미해지는 자리다.


def _fake_visa(root, categories):
    for category in categories:
        (root / category / "Data" / "Images" / "Normal").mkdir(parents=True)
        (root / category / "Data" / "Images" / "Anomaly").mkdir(parents=True)


def test_visa_is_used_only_when_every_item_is_there(tmp_path):
    """한 품목만 있으면 합성으로 떨어진다.

    절반은 실데이터 절반은 합성인 화면이 되면 무엇을 보고 있는지 흐려지고,
    품목마다 뱅크가 따로라는 것도 보여주지 못한다.
    """
    from app.pipeline import DEMO_ITEMS, visa_available

    categories = [category for _, _, category in DEMO_ITEMS]

    assert visa_available(tmp_path / "없음") is False

    partial = tmp_path / "일부"
    _fake_visa(partial, categories[:1])
    assert visa_available(partial) is False

    whole = tmp_path / "전부"
    _fake_visa(whole, categories)
    assert visa_available(whole) is True


def test_visa_never_runs_at_the_synthetic_resolution():
    """VisA 를 64/64 로 재면 실측표의 무작위 수준(0.526)보다도 아래다.

    화면은 멀쩡히 그려지고 숫자만 무의미해지므로 "돌아간다"와 "맞다"가
    갈리지 않는다. 두 설정이 섞이지 않게 못 박는다.
    """
    from app.pipeline import DEMO_CONFIG, VISA_CONFIG

    assert VISA_CONFIG.crop == 512
    assert VISA_CONFIG.backbone == "wide_resnet50_2"
    assert DEMO_CONFIG.crop != VISA_CONFIG.crop


def test_screen_says_which_data_it_stood_on():
    """합성으로 떨어진 것을 모르고 보면 수치를 실측으로 오해한다."""
    from app.view import _source_banner

    assert "VisA 실데이터" in _source_banner(True)
    assert "합성 이미지" in _source_banner(False)
    # 합성일 때만 주의 표시가 붙는다. VisA 로 섰는데 경고가 남으면 실측을 스스로 깎는다.
    assert "banner warn" in _source_banner(False)
    assert "banner warn" not in _source_banner(True)


# ── 판별 2번은 한 장이 아니라 구간으로 잰다 ─────────────────────────────


def test_quality_is_judged_over_the_lot_not_one_image(factory):
    """화질 이탈을 미검 이미지 한 장으로 판정하지 않는다.

    **한 번 여기서 진단이 통째로 바뀌었다.** `assess_quality([query])` 로
    한 장만 재고 있었는데, 미검 이미지는 정의상 결함이 있어서 지표가
    흔들린다. 그래서 **결함이 뚜렷할수록 설비·광학으로 잡히고**, 뱅크 오염
    시나리오가 `equipment_optics` 로 나왔다.

    설비 문제는 어느 시점부터 지속되는 현상이라 구간 비율로 봐야 한다
    (`data/quality_baseline.yaml` 의 `outlier_rule`). MES 가 가져온 로트
    전체를 넣고, 이탈 비율이 기준을 넘어야 의심한다.
    """
    outcome = run(factory, patch_override="defect")

    stage = next(s for s in outcome.stages if s.key == "evidence")
    label = next(name for name, _ in stage.rows if name.startswith("2."))
    assert dict(stage.rows)[label].startswith("○"), "판별 2번을 재지 못했다"
    assert "False" not in dict(stage.rows)[label], (
        "오염된 뱅크 시연에서 화질이 이탈로 잡혔다 — 한 장만 재고 있지 않은지 보라"
    )


def test_the_contaminated_demo_diagnoses_bank_contamination(factory):
    """혼입 이미지를 넣은 품목에서 뱅크 오염이 나온다.

    시연의 주 시나리오다. 다른 원인이 나오면 **원인 하나가 다른 것을 가리고
    있다**는 뜻이고, 재구성·게이트·섀도·승인까지 뒷단이 통째로 안 돈다.
    """
    outcome = run(factory, patch_override="defect")

    assert outcome.diagnosis is not None
    assert outcome.diagnosis.cause == "bank_contamination", (
        f"뱅크 오염 품목인데 {outcome.diagnosis.cause} 로 나왔다"
    )
    assert outcome.diagnosis.requires_bank_rebuild
    assert outcome.finished, "재구성이 답인 원인이면 승인 요청까지 가야 한다"


def test_the_quality_baseline_follows_the_item(factory):
    """화질 기준이 품목마다 다르다.

    목이 상수 하나를 돌려주고 있었다. 기준 분포는 원래 라인·품목마다
    따로이고, 상수로 두면 그 사실이 코드에서 사라진다. 실제로 그래서
    데모 품목을 바꾸자 멀쩡한 이미지가 전부 이탈로 잡혔다.
    """
    from app.pipeline import DEMO_ITEMS

    seen = [factory.quality_baseline(line, obj) for line, obj, _ in DEMO_ITEMS]
    assert all(s is not None for s in seen)
    means = [s["brightness"]["mean"] for s in seen]
    assert len(set(means)) > 1, "품목이 달라도 기준이 같다 — 상수를 돌려주고 있다"


# ── 로트가 크면 조회가 조용히 자른다 ────────────────────────────────────
#
# **4090 실측에서 터진 자리다.** `find_images` 의 기본 상한이 50 인데 가상
# 공장의 로트는 100장이다. 카탈로그는 정상을 먼저 넣으므로 잘리면 **결함이
# 먼저 사라진다.** 로트 74장 중 50장만 와서 결함 6장이 전부 빠졌고,
# 파이프라인은 "미검 없음"이라고 답했다. Mac 은 로트가 14장이라 안 걸렸다.


def _lot_with_defects_at_the_end(factory, size=74, defects=6):
    """정상을 앞에, 결함을 뒤에 둔 큰 로트 하나를 만든다. 실제 순서와 같다."""
    from dataclasses import replace

    line, obj, lot = "line_02", "pcb2", "LOT-BIG-001"
    seed_normal = next(r for r in factory.catalog
                       if r.object_name == obj and r.ground_truth == "pass")
    seed_defect = next(r for r in factory.catalog
                       if r.object_name == obj and r.ground_truth == "defect")

    rows = [
        replace(seed_normal, product_id=f"PCB2-BIG-N{i:03d}", lot=lot,
                line=line, split="operation")
        for i in range(size - defects)
    ]
    rows += [
        replace(seed_defect, product_id=f"PCB2-BIG-D{i:03d}", lot=lot,
                line=line, split="operation")
        for i in range(defects)
    ]
    return rows


def test_a_large_lot_is_not_silently_truncated(demo_factory):
    """로트가 크면 결함이 조용히 잘려 나가지 않는가.

    잘리면 파이프라인이 "미검 없음"이라고 답한다. 틀린 답인데 **에러가 아니라
    정상 응답으로 나와서** 아무도 눈치채지 못한다.
    """
    from agents.adapters import build_adapters
    from app.pipeline import _DemoSession
    from lookup import MockLookup

    rows = _lot_with_defects_at_the_end(demo_factory)
    lookup = MockLookup(catalog=rows + list(demo_factory.catalog),
                        banks=demo_factory.bank_versions(),
                        quality_provider=demo_factory.quality_baseline)
    session = _DemoSession(demo_factory, "x", {}, None, build_adapters(), 2.20, lookup)
    session.intake_issue(line="line_02", object_name="pcb2",
                         defect_type="스크래치", product_id="PCB2-BIG-D000")
    result = session.lookup_mes()

    assert result["images"] == 74, f"로트 74장이 다 와야 하는데 {result['images']}장"
    assert result["defects"] == 6, (
        f"결함 6장이 와야 하는데 {result['defects']}장 — 상한에 잘렸다"
    )


def test_hitting_the_scan_limit_is_written_on_the_screen(demo_factory):
    """상한에 닿으면 화면에 그 사실을 적는다.

    상한 자체는 남긴다 — 조건 없이 수만 장을 끌어오는 실수를 막아야 한다.
    다만 **조용히 자르면 "이게 전부"로 읽힌다.**
    """
    from agents.adapters import build_adapters
    from app.pipeline import LOT_SCAN_LIMIT, _DemoSession
    from lookup import MockLookup

    rows = _lot_with_defects_at_the_end(demo_factory, size=LOT_SCAN_LIMIT + 10)
    lookup = MockLookup(catalog=rows, banks=demo_factory.bank_versions(),
                        quality_provider=demo_factory.quality_baseline)
    session = _DemoSession(demo_factory, "x", {}, None, build_adapters(), 2.20, lookup)
    session.intake_issue(line="line_02", object_name="pcb2",
                         defect_type="스크래치", product_id="PCB2-BIG-D000")
    session.lookup_mes()

    stage = next(s for s in session.outcome.stages if s.key == "mes")
    found = dict(stage.rows)["찾은 이미지"]
    assert "잘렸을 수 있습니다" in found, f"상한에 닿았는데 화면에 안 적힌다: {found}"


def test_a_truncated_table_says_how_many_were_folded():
    """화면 표를 자를 때 남은 건수를 적는다.

    총계는 headline 에 있지만 **표만 보는 사람은 그것이 전부라고 읽는다.**
    특히 섀도 불일치는 사람이 직접 눈으로 볼 목록이라, 8건만 보여주고 나머지를
    안 알리면 "이만큼만 보면 된다"는 섀도의 논거 자체가 무너진다.
    """
    from app.pipeline import _sampled

    rows = [(f"img_{i}", "…") for i in range(8)]
    assert _sampled(rows, total=8, shown=8) == rows, "다 보여줬으면 덧붙이지 않는다"

    folded = _sampled(rows, total=31, shown=8)
    assert folded[-1] == ("…", "외 23건은 접었습니다")
    assert len(folded) == 9


# ── 판별 1번은 전체 이미지가 아니라 역추적 크롭을 본다 ──────────────────
#
# **4090 실측에서 진단이 통째로 멈췄다.** 전체 이미지를 그대로 주자 시각 언어
# 모델이 결함 이미지를 "normal" 이라 답했고, 판별 1번이 단독 차단 조건이라
# `decide()` 가 "접수 오류"로 끊었다.
#
# 우리 실측이 이미 답을 갖고 있었다(`docs/실험_역추적크롭.md`).
#
#     여유 24px (63×64)    0/10   전부 "무엇을 보는지 모르겠다"
#     여유 64px (143×144)  9/10
#
# 그 값이 코드에는 **판별 5번에만** 물려 있었다.


def _session_at_inspection(factory):
    """추론까지 마친 세션. 판별 항목 입력을 여기서 본다."""
    from agents.adapters import build_adapters
    from app.pipeline import _DemoSession

    line, object_name = CONTAMINATED_ITEM
    session = _DemoSession(
        factory, default_issue(factory),
        {"line": line, "object_name": object_name, "defect_type": "scratch",
         "product_id": factory.reported_product},
        None, build_adapters(), 2.20, None,
    )
    session.intake_issue(line=line, object_name=object_name,
                         defect_type="scratch",
                         product_id=factory.reported_product)
    session.lookup_mes()
    session.run_inspection()
    return session


def test_check_one_is_asked_about_a_crop_not_the_whole_image(demo_factory, monkeypatch):
    """판별 1번에 들어가는 것이 원본 경로가 아니라 잘라낸 조각이어야 한다."""
    from PIL import Image

    import app.pipeline as pipeline
    from agents.vision import VisionJudgment

    seen = {}

    def spy(adapter, image, reported_defect="", context_image=None):
        seen["image"] = image
        seen["context"] = context_image
        return VisionJudgment(verdict="defect", confidence=0.9, reason="",
                              model="spy", is_stub=False)

    monkeypatch.setattr(pipeline, "judge_defect_visible", spy)
    run(demo_factory)

    assert seen, "판별 1번이 불리지 않았다"
    assert isinstance(seen["image"], Image.Image), (
        "원본 경로가 그대로 들어간다 — 크롭이 안 물렸다"
    )
    assert seen["context"] is not None, "주변 맥락도 함께 줘야 판독이 안정된다"


def test_the_crop_is_enlarged_enough_to_read(demo_factory):
    """실측에서 정해진 크기로 확대한다. 작으면 전부 unknown 이 나온다."""
    session = _session_at_inspection(demo_factory)
    crop, context = session._query_crop(session.inference)

    assert crop is not None and context is not None
    assert min(crop.size) >= 512, f"확대가 안 됐다: {crop.size}"


def test_the_crop_comes_from_the_reported_image_not_the_bank(demo_factory):
    """판별 1번은 **접수 이미지**를, 판별 5번은 **뱅크 이미지**를 자른다.

    대상을 헷갈리면 "접수 이미지에 결함이 보이는가"를 뱅크 이미지에 묻게 된다.
    """
    session = _session_at_inspection(demo_factory)
    top = session.inference.top_match

    assert top.query.source_image != top.bank.source_image, (
        "질의와 뱅크가 같은 이미지면 이 시험이 구분하지 못한다"
    )
    assert session._query_crop(session.inference)[0] is not None


def test_a_missing_traceback_falls_back_to_the_whole_image(demo_factory):
    """자를 자리를 못 찾으면 전체 이미지로 되돌아간다.

    판독이 약해지지만 **멈추지는 않는다.** 못 봤다는 답이 나오면 그 사실이
    근거에 남는다.
    """
    session = _session_at_inspection(demo_factory)

    class NoMatch:
        top_match = None
        grid_h = grid_w = 0

    assert session._query_crop(NoMatch()) == (None, None)
    assert session._query_crop(None) == (None, None)


def test_뺄_이미지_표시가_윈도우_경로에서도_붙는다(factory):
    """경로 구분자가 달라도 뺄 이미지에 표시가 붙어야 한다.

    **맥에서 통과한 것이 증거가 되지 못한다.** 시연은 윈도우에서 돌고
    시험은 맥에서 돈다. 실제로 `str(Path(...))` 가 역슬래시를 내서 4090
    화면에만 표시가 하나도 안 붙었고(`figure class="bm"` 133개 중 `out`
    0개), 그런데 캡션은 "빨간 테두리가 뺄 2장입니다" 라고 적고 있었다.

    같은 함정에 두 번 걸렸다 — `agents/release.py` 의 `display_path` 도
    윈도우 경로를 한 조각으로 읽어 사용자명을 못 걸렀다. **경로를 문자열로
    견주거나 주소로 쓰는 자리는 전부 이 함정이 있다.**
    """
    from app.view import _bank_html

    outcome = run(factory, patch_override="defect")
    if outcome.plan is None or not outcome.plan.remove:
        pytest.skip("이 실행에서는 뺄 이미지가 없다")

    before = _bank_html(outcome, factory).count("bm out")
    assert before > 0, "맥 경로에서 표시가 붙어야 한다"

    for candidate in outcome.plan.remove:
        candidate.image = candidate.image.replace("/", "\\")
    assert _bank_html(outcome, factory).count("bm out") == before, (
        "역슬래시 경로에서도 같은 수가 붙어야 한다"
    )


def test_화면_주소에_역슬래시가_남지_않는다(factory):
    """`/image/` 주소에 역슬래시가 들어가면 윈도우 밖에서 404 다.

    윈도우 서버는 역슬래시를 받아 줘서 그 기계에서는 통했다. **그래서
    맥에서도, 4090 에서도 안 걸렸다.** 다른 환경으로 옮기면 그때 터진다.
    """
    from app.view import _bank_html, _factory_html

    outcome = run(factory, patch_override="defect")
    for html in (_bank_html(outcome, factory), _factory_html(factory)):
        assert "%5C" not in html, "주소에 역슬래시가 인코딩돼 남았다"
        assert "\\" not in html.split("<style")[0], "주소에 역슬래시가 그대로 남았다"


def test_뺄_이미지가_격자_앞쪽에_온다(factory):
    """**붙는 것만으로는 부족하다. 보여야 한다.**

    오늘 같은 자리에서 두 번 사고가 났다.

        표시가 안 붙음    윈도우 경로라 비교가 영영 안 맞았다
        붙었는데 안 보임  116·117 번째라 스크롤해야 나왔다

    캡션은 두 번 다 "빨간 테두리가 뺄 2장입니다" 라고 말했다. **화면이
    없는 것을 있다고 말하는 상태**가 원인만 바뀌어 반복됐다.

    그래서 셋을 함께 본다 — 몇 개라고 말하는가, 실제로 몇 개인가,
    그것이 앞쪽인가.
    """
    import re
    from app.view import _bank_html

    outcome = run(factory, patch_override="defect")
    if outcome.plan is None or not outcome.plan.remove:
        pytest.skip("이 실행에서는 뺄 이미지가 없다")

    html = _bank_html(outcome, factory)
    tiles = [m.group(1) for m in re.finditer(r'<figure class="(bm(?:\s+[a-z]+)*)"', html)]
    grid = [c for c in tiles if "big" not in c.split()]
    marked = [n for n, c in enumerate(grid) if "out" in c.split()]

    assert len(marked) == len(outcome.plan.remove), (
        f"뺄 것 {len(outcome.plan.remove)}장인데 표시는 {len(marked)}개다"
    )
    assert marked == list(range(len(marked))), (
        f"뺄 것이 격자 맨 앞에 와야 스크롤 없이 보인다. 지금 순번 {marked}"
    )


def test_커버리지_부족이_보충까지_실제로_간다(factory):
    """**진단만 되고 조치가 안 서던 자리다. 끝까지 도는지 본다.**

    이 경로는 세 군데가 끊겨 있었다.

        판별 6번이 "어느 조건이 비었는가"를 문장에만 남기고 버렸다
        파이프라인이 그것을 큐레이션에 안 넘겼다
        `DirectoryImageSource` 에 조건을 안 넘겨 보충이 언제나 0장이었다

    셋 다 이어야 뱅크가 실제로 바뀐다. 하나만 이으면 앞의 둘이 조용히
    막아서 "보충하겠다"고 말하고 아무 일도 안 일어난다. **뱅크 오염
    시연에서는 제거만 하므로 이 경로가 안 드러난다.**

    전에는 이 시험이 조회 계층에 없는 로트(`LOT-뱅크에-없는-것`)를 손으로
    끼워 넣어 판별 6번을 "없음"으로 만들었다. **그러면 배선만 재고 공장은
    안 잰다** — 실제 시연에서 그 조건이 정말 비어 있는지는 확인되지 않는다.
    지금은 야간 로트를 뱅크에 안 넣은 라인(`COVERAGE_ITEM`)을 그대로 돌린다.
    """
    outcome = run_line(factory, COVERAGE_ITEM, patch_override="genuine_normal")

    assert outcome.diagnosis is not None
    assert outcome.diagnosis.cause == "coverage_gap"

    item6 = outcome.diagnosis.evidence_by_item(6)
    assert item6 is not None and item6.extra.get("missing_conditions"), (
        "판별 6번이 어느 조건이 비었는지를 값으로 남겨야 한다"
    )

    assert outcome.plan is not None and outcome.plan.touches_bank, (
        "커버리지 부족은 보충이 답이다"
    )
    assert outcome.plan.add, "보충 요청이 서야 한다"

    assert outcome.rebuild is not None and outcome.rebuild.executed
    record = outcome.rebuild.record
    assert record is not None and record.added, (
        "보충이 0장이면 뱅크가 그대로다. 그것을 성공이라 하면 안 된다"
    )
    # 뱅크가 실제로 커졌는지. 숫자가 안 움직이면 아무 일도 안 일어난 것이다.
    assert len(outcome.rebuild.bank.images) == record.kept_count + len(record.added)
    assert record.kept_count > 0


# ── 라인마다 다른 것이 심겨 있는가 ──────────────────────────────────────


def test_라인마다_심은_것이_다르고_그대로_진단된다(factory):
    """**어느 라인을 눌러도 같은 이야기면 라인이 넷일 이유가 없다.**

    오염 하나만 심어 두었을 때는 2·3·4라인이 진단은 돌되 재구성까지 가지
    않았다. 그것이 맞는 동작이긴 해도 보여줄 것은 1라인 하나뿐이었다
    (`docs/4090_보고_시연구성.md` 1장).

    여기서 재는 것은 표에 적어 둔 `expected_cause` 가 실제로 나오는가다.
    **표만 고치고 공장이 안 따라오면 여기서 걸린다** — 야간 로트를 뱅크에
    넣는지, 혼입을 몇 장 섞는지가 전부 그 값과 맞물려 있다.
    """
    from app.pipeline import DEMO_LINES

    override = {"bank_contamination": "defect"}
    for spec in DEMO_LINES:
        outcome = run_line(
            factory, spec.key,
            patch_override=override.get(spec.expected_cause or "", "genuine_normal"),
        )
        assert outcome.diagnosis is not None, f"{spec.line}: 진단까지 못 갔다"
        if spec.expected_cause is None:
            assert not (outcome.plan and outcome.plan.touches_bank), (
                f"{spec.line}: 심은 것이 없는데 뱅크를 건드렸다 "
                f"(원인 {outcome.diagnosis.cause})"
            )
            continue
        assert outcome.diagnosis.cause == spec.expected_cause, (
            f"{spec.line}: {spec.expected_cause} 를 심었는데 "
            f"{outcome.diagnosis.cause} 가 나왔다"
        )


def test_뱅크_구성_이력이_실제_구성에서_나온다(factory):
    """판별 6번이 보는 조건이 **그 뱅크에 실제로 든 이미지**에서 나온다.

    전에는 목이 상수 하나를 돌려줬고, 거기 적힌 로트(`LOT-20260602-004`)는
    가상 공장에 없는 로트였다. **커버리지 판정이 공장 구성과 무관하게
    정해져 있었다는 뜻**이다. 화질 기준을 품목별로 돌린 것과 같은 자리다.
    """
    from app.pipeline import COVERAGE_ITEM, DEMO_LINE_BY_KEY

    for key, item in factory.items.items():
        profile = factory.bank_profile(*key)
        assert profile is not None
        assert profile.source_image_count == len(item.bank.images)

        lots = set(profile.conditions.get("lot") or [])
        in_bank = {str(p) for p in item.bank.images}
        for record in factory.catalog:
            if (record.line, record.object_name) != key:
                continue
            if str(record.path) in in_bank:
                assert record.lot in lots, f"{key}: 뱅크에 든 로트가 이력에 없다"

        night_lots = {
            record.lot for record in factory.catalog
            if (record.line, record.object_name) == key
            and str(record.path) in {str(p) for p in item.night_lot}
        }
        if key == COVERAGE_ITEM:
            assert not (night_lots & lots), (
                f"{key}: 야간 로트를 뱅크에 안 넣었는데 구성 이력에 들어 있다"
            )
        else:
            assert night_lots <= lots, (
                f"{key}: 야간 로트를 뱅크에 넣었는데 구성 이력에 없다. "
                f"그러면 이 라인까지 커버리지 부족으로 진단된다"
            )
        assert DEMO_LINE_BY_KEY[key].night_lot_in_bank == (key != COVERAGE_ITEM)


def test_품목마다_다른_임계값이_실제로_쓰인다(factory):
    """**판별 3번의 임계값은 조회에서 온다.** 실행 인자 하나가 아니다.

    4090 실측에서 과검 1% 지점이 candle 1.925 · capsules 2.560 으로 갈렸고,
    2.20 이 쓸 만했던 것은 pcb1 이 2.206 이라 우연히 맞은 것이다
    (`docs/4090_보고_시연구성.md` 3장).
    """
    from app.pipeline import COVERAGE_ITEM, DEMO_THRESHOLDS

    assert COVERAGE_ITEM in DEMO_THRESHOLDS, (
        "커버리지 부족 라인은 과검을 억누르려 임계값을 올려 둔 상태다. "
        "그 값이 없으면 미검이 안 생겨 이슈 자체가 성립하지 않는다"
    )
    outcome = run_line(factory, COVERAGE_ITEM, patch_override="genuine_normal")
    assert outcome.threshold == DEMO_THRESHOLDS[COVERAGE_ITEM], (
        f"조회가 준 값 {DEMO_THRESHOLDS[COVERAGE_ITEM]} 이 아니라 "
        f"{outcome.threshold} 로 판정했다"
    )


def test_보충에_결함_이미지가_섞이지_않는다(factory):
    """**미검 건은 설비가 양품이라 한 것이라 `verdict` 가 pass 다.**

    그것만 보고 정상으로 세면 보충이 결함을 뱅크에 넣는다 — 우리가 뱅크
    오염을 만드는 셈이다. 실제로 야간 로트 보충에 결함 6장이 함께 들어갔다.
    """
    outcome = run_line(factory, COVERAGE_ITEM, patch_override="genuine_normal")
    assert outcome.rebuild is not None and outcome.rebuild.record is not None

    defects = {
        str(record.path) for record in factory.catalog
        if record.ground_truth == "defect"
    }
    added = {str(name) for name in outcome.rebuild.record.added}
    assert not (added & defects), (
        f"보충에 결함이 섞였다: {sorted(added & defects)[:3]}"
    )


def test_사람이_적어_넣은_임계값이_품목별_값을_이긴다(factory):
    """**화면의 임계값 칸이 장식이 되면 안 된다.**

    그 칸은 "임계값 조절로 풀리는 문제가 아니다"를 만져 보게 하려고 열어
    둔 것이다. 품목별 값을 조회에서 받게 하면서 그것이 사람이 적은 값을
    덮으면, 숫자를 바꿔도 판정이 안 움직인다.
    """
    from app.pipeline import DEMO_THRESHOLDS

    forced = run_line(factory, COVERAGE_ITEM, patch_override="genuine_normal",
                      threshold=1.10)
    assert forced.threshold == 1.10, (
        f"사람이 1.10 을 적었는데 {forced.threshold} 로 판정했다"
    )
    assert DEMO_THRESHOLDS[COVERAGE_ITEM] != 1.10, "시험이 뜻을 잃었다"
