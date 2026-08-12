"""진단 판정 검증 — 작업 13.

정량 목표가 두 개다. 원인 분류 정확도 80% 이상, 그리고 **뱅크 재구성이
답이 아닌 케이스의 전건 차단.** 후자가 진짜 시험대다. 재학습을 해야 할 때
하는 것보다 하지 말아야 할 때 멈추는 것이 어렵다.

그래서 이 파일은 세 가지를 본다.

  1. 여섯 원인에 각각 제대로 도달하는가
  2. 근거가 모자라면 판정을 보류하는가 — 특히 판별 5번이 없을 때
  3. 판정 순서가 지켜지는가 — 화질이 먼저, 역추적이 임계값보다 먼저
"""

from __future__ import annotations

import pytest

from agents.diagnose import REBUILD_REQUIRED, Evidence, collect_evidence, decide
from agents.vision import VisionJudgment
from inspection.quality import QualityAssessment
from inspection.sweep import FeasibilityVerdict
from inspection.types import InferenceResult, NearestMatch, PatchRef
from lookup.base import BankProfile, CriteriaRule, ThresholdRecord
from lookup.mock import resolved_duplicate


# ── 근거 조립 helper ───────────────────────────────────────────────────


def vision(verdict: str, stub: bool = False) -> VisionJudgment:
    return VisionJudgment(
        verdict=verdict, confidence=0.9, reason=f"{verdict} 판독", model="test", is_stub=stub
    )


def quality(ok: bool) -> QualityAssessment:
    return QualityAssessment(
        image_count=30,
        deviations=[],
        outlier_ratio={},
        within_baseline=ok,
        reason="기준 안" if ok else "sharpness z=-4.10 (이탈 62%) 이탈",
    )


def inference(score: float, bank_image: str = "normal/normal_003.png") -> InferenceResult:
    match = NearestMatch(
        query=PatchRef("query.png", 1, 2, 10),
        bank=PatchRef(bank_image, 4, 5, 37),
        distance=score,
        bank_row_index=114,
    )
    return InferenceResult(
        image="query.png",
        score=score,
        max_patch_distance=score,
        grid_h=8,
        grid_w=8,
        matches=[match],
        bank_version="v3",
    )


def threshold(value: float = 2.20) -> ThresholdRecord:
    return ThresholdRecord("line_02", "capsules", "v3", value)


def criteria(defect_area: float = 150.0) -> CriteriaRule:
    return CriteriaRule("CR-1", "line_02", "capsules", "dent", defect_area, 90.0)


def bank_profile(dates: list[str] | None = None) -> BankProfile:
    return BankProfile(
        bank_version="v3",
        line="line_02",
        object_name="capsules",
        source_image_count=12,
        patch_count=800,
        conditions={"date": dates if dates is not None else ["2026-06-01", "2026-06-02"]},
    )


def build(
    *,
    visible="defect",
    quality_ok=True,
    score=1.0,
    thr=2.20,
    patch=None,
    coverage_date="2026-06-01",
    bank_dates=None,
    area=200.0,
    defect_area=150.0,
) -> list[Evidence]:
    return collect_evidence(
        defect_visible=vision(visible),
        quality=quality(quality_ok),
        inference=inference(score),
        threshold=threshold(thr),
        patch_judgment=vision(patch) if patch else None,
        bank_profile=bank_profile(bank_dates),
        condition_key="date",
        condition_value=coverage_date,
        criteria=criteria(defect_area),
        defect_area=area,
    )


# ── 여섯 원인에 도달하는가 ─────────────────────────────────────────────


def test_bank_contamination():
    """되짚은 패치가 결함이면 뱅크 오염."""
    result = decide(build(patch="defect", score=1.0))

    assert result.cause == "bank_contamination"
    assert result.requires_bank_rebuild is True
    assert result.needs_human is False
    assert "rebuild_bank" in result.recommended_actions
    assert "lower_threshold" in result.forbidden_actions


def test_normal_overlap_with_sweep():
    """패치가 진짜 정상품이고 임계값으로 해결 안 되면 정상 분포 중첩."""
    sweep = FeasibilityVerdict(
        achievable=False,
        target_detection=1.0,
        max_acceptable_fpr=0.05,
        required_threshold=0.45,
        resulting_fpr=0.70,
        resulting_detection=1.0,
        auroc=0.617,
        reason="과검률 70.0%가 되어 임계값 조정으로는 해결되지 않는다.",
    )
    result = decide(build(patch="normal", score=0.8), sweep=sweep)

    assert result.cause == "normal_overlap"
    assert result.requires_bank_rebuild is False
    assert "rebuild_bank" in result.forbidden_actions
    assert "70.0%" in result.reasoning


def test_threshold_with_sweep():
    """같은 조건에서 스윕이 해결 가능이라고 하면 임계값 문제."""
    sweep = FeasibilityVerdict(
        achievable=True,
        target_detection=1.0,
        max_acceptable_fpr=0.05,
        required_threshold=1.9,
        resulting_fpr=0.0,
        resulting_detection=1.0,
        auroc=1.0,
        reason="과검률 0.0%로 달성한다.",
    )
    result = decide(build(patch="normal", score=2.1), sweep=sweep)

    assert result.cause == "threshold"
    assert result.requires_bank_rebuild is False
    assert "rebuild_bank" in result.forbidden_actions


def test_coverage_gap():
    """현재 조건의 정상 패치가 뱅크에 없으면 커버리지 부족."""
    result = decide(build(patch="normal", coverage_date="2026-06-20", bank_dates=["2026-06-01"]))

    assert result.cause == "coverage_gap"
    assert result.requires_bank_rebuild is True
    assert "add_normal_images_for_condition" in result.recommended_actions


def test_equipment_optics():
    """화질이 기준을 벗어나면 설비·광학."""
    result = decide(build(quality_ok=False, patch="defect"))

    assert result.cause == "equipment_optics"
    assert result.requires_bank_rebuild is False
    assert "request_equipment_check" in result.recommended_actions


def test_criteria_problem():
    """검출은 했는데 기준상 양품이면 기준 문제."""
    result = decide(build(score=3.0, thr=2.20, area=50.0, defect_area=150.0))

    assert result.cause == "criteria"
    assert result.requires_bank_rebuild is False
    assert result.recommended_actions == ["redefine_criteria"]


# ── 모르면 찍지 않는가 ─────────────────────────────────────────────────


def test_missing_patch_judgment_withholds_verdict():
    """판별 5번이 없으면 뱅크 오염과 정상 분포 중첩을 가를 수 없다.

    조치가 정반대라 찍으면 절반은 반대 조치를 지시하게 된다.
    이 테스트가 이 파일에서 제일 중요하다.
    """
    result = decide(build(patch=None))

    assert result.cause is None
    assert result.requires_bank_rebuild is None
    assert result.needs_human is True
    assert set(result.candidate_causes) >= {"bank_contamination", "normal_overlap"}
    assert "정반대" in result.blocking_reason
    assert result.recommended_actions == []


def test_stub_vlm_does_not_count_as_judgment():
    """스텁이 낸 판정은 근거가 아니다. 보류로 떨어져야 한다."""
    evidence = collect_evidence(
        defect_visible=vision("defect", stub=True),
        quality=quality(True),
        inference=inference(1.0),
        threshold=threshold(),
        patch_judgment=vision("defect", stub=True),  # 스텁이 결함이라고 말해도
        bank_profile=bank_profile(),
        condition_key="date",
        condition_value="2026-06-01",
        criteria=criteria(),
        defect_area=200.0,
    )
    result = decide(evidence)

    assert result.cause is None, "스텁 응답으로 뱅크 오염을 확정하면 안 된다"
    assert result.needs_human is True


def test_unknown_patch_judgment_withholds():
    result = decide(build(patch="unknown"))
    assert result.cause is None
    assert result.needs_human is True


def test_missing_trace_withholds():
    """역추적이 없으면 더 갈 수 없다."""
    evidence = collect_evidence(
        defect_visible=vision("defect"),
        quality=quality(True),
        inference=None,
        threshold=threshold(),
        bank_profile=bank_profile(),
        condition_key="date",
        condition_value="2026-06-01",
    )
    result = decide(evidence)

    assert result.cause is None
    assert "판별 4번" in result.blocking_reason


def test_defect_not_visible_stops_before_diagnosis():
    """결함이 안 보이면 미검출이 아니라 접수 오류일 수 있다."""
    result = decide(build(visible="normal", patch="defect"))

    assert result.cause is None
    assert "접수 오류" in result.blocking_reason
    assert result.recommended_actions == ["request_correct_image"]


def test_no_sweep_lowers_confidence():
    """스윕 없이 내린 판정은 확신도가 낮고 사람 확인이 붙어야 한다."""
    with_sweep = decide(
        build(patch="normal", score=0.8),
        sweep=FeasibilityVerdict(False, 1.0, 0.05, 0.45, 0.7, 1.0, 0.6, "겹친다"),
    )
    without = decide(build(patch="normal", score=0.8))

    assert with_sweep.cause == without.cause == "normal_overlap"
    assert with_sweep.confidence == "high"
    assert without.confidence == "medium"
    assert without.needs_human is True


# ── 판정 순서 ──────────────────────────────────────────────────────────


def test_quality_is_checked_before_trace():
    """화질이 나가면 최근접 패치가 결함으로 보여도 설비를 먼저 본다.

    화질이 오염된 상태에서 나온 역추적 결과는 신뢰할 수 없다.
    """
    result = decide(build(quality_ok=False, patch="defect"))
    assert result.cause == "equipment_optics"


def test_trace_is_checked_before_threshold():
    """점수가 임계값 근처여도 되짚은 패치가 결함이면 뱅크 오염이다.

    실측으로 확인한 것 — 오염된 뱅크에서도 스윕은 '임계값을 다시 잡으면
    된다'고 답한다. 순서를 바꾸면 오염이 임계값 문제로 분류되고, 증상만
    덮은 채 다음 로트에서 재발한다.
    """
    sweep = FeasibilityVerdict(True, 1.0, 0.05, 1.9, 0.0, 1.0, 1.0, "해결된다")
    result = decide(build(patch="defect", score=2.1), sweep=sweep)

    assert result.cause == "bank_contamination", "임계값 문제로 분류되면 안 된다"


# ── 중복 차단 ──────────────────────────────────────────────────────────


def test_resolved_duplicate_stops_diagnosis():
    """이미 해결된 동일 건이면 진단을 진행하지 않는다."""
    result = decide(build(patch="defect"), similar_issues=[resolved_duplicate()])

    assert result.duplicate_of == "MOCK-ISS-0042"
    assert result.cause is None
    assert result.recommended_actions == ["review_past_issue"]


def test_unresolved_similar_issue_does_not_block():
    """해결되지 않은 유사 건은 차단 근거가 아니다."""
    issue = resolved_duplicate()
    issue.resolved = False
    result = decide(build(patch="defect"), similar_issues=[issue])

    assert result.duplicate_of is None
    assert result.cause == "bank_contamination"


def test_low_similarity_does_not_block():
    issue = resolved_duplicate(similarity=0.40)
    result = decide(build(patch="defect"), similar_issues=[issue])

    assert result.duplicate_of is None
    assert result.cause == "bank_contamination"


# ── 재구성 차단 ────────────────────────────────────────────────────────


def test_rebuild_blocked_for_four_of_six_causes():
    """여섯 중 넷은 뱅크 재구성이 답이 아니다. 정량 목표의 근거다."""
    blocked = [c for c, needs in REBUILD_REQUIRED.items() if not needs]

    assert set(blocked) == {"threshold", "normal_overlap", "equipment_optics", "criteria"}
    assert len(blocked) == 4


@pytest.mark.parametrize(
    "cause,evidence_kwargs,sweep",
    [
        ("equipment_optics", {"quality_ok": False, "patch": "defect"}, None),
        ("criteria", {"score": 3.0, "area": 50.0}, None),
        (
            "normal_overlap",
            {"patch": "normal", "score": 0.8},
            FeasibilityVerdict(False, 1.0, 0.05, 0.45, 0.7, 1.0, 0.6, "겹친다"),
        ),
        (
            "threshold",
            {"patch": "normal", "score": 2.1},
            FeasibilityVerdict(True, 1.0, 0.05, 1.9, 0.0, 1.0, 1.0, "해결된다"),
        ),
    ],
)
def test_no_rebuild_recommended_when_rebuild_is_not_the_answer(cause, evidence_kwargs, sweep):
    """재구성이 답이 아닌 원인에서 재구성을 권고하면 안 된다."""
    result = decide(build(**evidence_kwargs), sweep=sweep)

    assert result.cause == cause
    assert result.requires_bank_rebuild is False
    assert "rebuild_bank" not in result.recommended_actions


# ── 근거가 리포트에 남는가 ─────────────────────────────────────────────


def test_all_seven_items_appear_even_when_missing():
    """확인하지 못한 항목도 목록에 남아야 한다."""
    evidence = collect_evidence(inference=inference(1.0))

    assert [e.item_no for e in evidence] == [1, 2, 3, 4, 5, 6, 7]
    assert sum(1 for e in evidence if not e.usable) >= 5


def test_evidence_sources_are_tagged():
    """어떤 근거가 모델에서 왔고 어떤 것이 조회에서 왔는지 구분돼야 한다."""
    evidence = build(patch="defect")
    by_item = {e.item_no: e.source for e in evidence}

    assert by_item[1] == "vlm"
    assert by_item[5] == "vlm"
    assert by_item[4] == "trace"
    assert by_item[2] == "compute"
    assert by_item[3] == by_item[6] == by_item[7] == "lookup"

    # 시각 언어 모델에 걸린 항목은 둘뿐이어야 한다
    assert sum(1 for s in by_item.values() if s == "vlm") == 2


def test_result_serializes_for_handoff():
    result = decide(build(patch="defect"))
    payload = result.to_dict()

    assert payload["cause"] == "bank_contamination"
    assert payload["cause_label"] == "뱅크 오염"
    assert len(payload["evidence"]) == 7
