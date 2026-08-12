"""조회 계층 계약 테스트 — 이동현용.

`lookup/factory.py` 에 구현을 만들면 **이 테스트가 자동으로 그 구현도
검사합니다.** 통과하면 목을 빼고 끼울 수 있고, 진단 코드는 고치지 않습니다.

실행:
    .venv/bin/python -m pytest tests/test_lookup_contract.py -q

아직 구현이 없으면 목만 검사하고 "구현 없음"으로 건너뜁니다. 정상입니다.

## 만드는 법

`lookup/factory.py` 를 만들고 아래 중 하나로 이름 지으세요.
자동으로 찾습니다.

    class FactoryLookup: ...
    class Lookup: ...

생성자는 인자 없이 만들 수 있어야 합니다. 데이터 경로가 필요하면 기본값을
주세요. 예:

    class FactoryLookup:
        def __init__(self, manifest_path="data/manifest.csv"):
            ...

## 이 테스트가 요구하는 것

1. 함수 5개가 있고 이름·인자가 맞을 것
2. 못 찾으면 **예외가 아니라 None** 을 돌려줄 것
3. 돌려주는 값이 정해진 자료형일 것
4. `get_criteria(at=...)` 가 그 시점의 기준을 줄 것
5. `BankProfile.covers()` 가 동작할 것

하나씩 초록으로 만들면 됩니다. 한 번에 다 맞추려 하지 마세요.
"""

from __future__ import annotations

from datetime import date

import pytest

from lookup.base import (
    BankProfile,
    CriteriaRule,
    PastIssue,
    QualityBaselineRecord,
    ThresholdRecord,
)
from lookup.mock import MockLookup

REQUIRED_METHODS = (
    "get_threshold",
    "get_quality_baseline",
    "get_criteria",
    "get_bank_profile",
    "find_similar_issues",
)


def _load_implementations():
    """검사할 구현을 모은다. 목은 항상, 실구현은 있으면."""
    found = [pytest.param(MockLookup(), id="mock")]

    try:
        import lookup.factory as module
    except ImportError:
        return found

    for name in ("FactoryLookup", "Lookup"):
        cls = getattr(module, name, None)
        if cls is None:
            continue
        try:
            found.append(pytest.param(cls(), id=f"factory:{name}"))
        except Exception as exc:
            found.append(
                pytest.param(
                    None,
                    id=f"factory:{name}",
                    marks=pytest.mark.xfail(
                        reason=f"{name}() 을 인자 없이 만들지 못했다: {exc}", strict=False
                    ),
                )
            )
        break
    return found


@pytest.fixture(params=_load_implementations())
def lookup(request):
    if request.param is None:
        pytest.skip("구현을 만들지 못했다")
    return request.param


# ── 1. 함수가 다 있는가 ────────────────────────────────────────────────


def test_all_required_methods_exist(lookup):
    """함수 5개가 있어야 진단이 판별 항목을 채울 수 있다."""
    missing = [m for m in REQUIRED_METHODS if not callable(getattr(lookup, m, None))]
    assert not missing, (
        f"함수가 없습니다: {missing}\n"
        f"lookup/base.py 의 LookupLayer 를 보고 이름과 인자를 맞춰 주세요."
    )


# ── 2. 못 찾으면 None (예외가 아니라) ──────────────────────────────────


def test_unknown_key_returns_none_not_exception(lookup):
    """없는 값을 물었을 때 예외를 던지면 진단이 멈춘다.

    진단은 "근거를 얻지 못했다"를 하나의 상태로 다룬다. 조회 실패로
    전체가 죽으면 안 된다.
    """
    try:
        threshold = lookup.get_threshold("없는라인", "없는품목", "없는뱅크")
        baseline = lookup.get_quality_baseline("없는라인", "없는품목")
        criteria = lookup.get_criteria("없는라인", "없는품목")
        profile = lookup.get_bank_profile("없는뱅크버전")
    except Exception as exc:
        pytest.fail(
            f"없는 값을 물었더니 예외가 났습니다: {type(exc).__name__}: {exc}\n"
            f"찾지 못하면 None 을 돌려주세요."
        )

    for name, value, kind in (
        ("get_threshold", threshold, ThresholdRecord),
        ("get_quality_baseline", baseline, QualityBaselineRecord),
        ("get_criteria", criteria, CriteriaRule),
        ("get_bank_profile", profile, BankProfile),
    ):
        assert value is None or isinstance(value, kind), (
            f"{name} 이 {type(value).__name__} 을 돌려줬습니다. "
            f"{kind.__name__} 또는 None 이어야 합니다."
        )


def test_find_similar_issues_returns_list_never_none(lookup):
    """유사 사례는 없으면 빈 목록이다. None 이 아니다."""
    result = lookup.find_similar_issues("없는라인", "없는품목")
    assert isinstance(result, list), (
        f"find_similar_issues 가 {type(result).__name__} 을 돌려줬습니다. "
        f"list 여야 하고, 없으면 빈 리스트입니다."
    )
    assert all(isinstance(i, PastIssue) for i in result), "항목은 PastIssue 여야 합니다."


# ── 3. 자료형이 맞는가 ─────────────────────────────────────────────────


def test_threshold_shape(lookup):
    """판별 3번 — 임계값."""
    record = lookup.get_threshold("line_02", "capsules", "v3")
    if record is None:
        pytest.skip("line_02/capsules/v3 데이터가 아직 없다")

    assert isinstance(record, ThresholdRecord)
    assert isinstance(record.value, (int, float)), "value 는 숫자여야 합니다."
    assert record.value > 0, "임계값이 0 이하입니다. 이상 점수와 비교할 값입니다."


def test_quality_baseline_shape(lookup):
    """판별 2번 — 화질 기준 분포.

    stats 는 inspection.quality.assess_quality 에 그대로 넘어갑니다.
    지표마다 mean 과 std 가 있어야 하고, std 가 0 이면 z 점수를 못 냅니다.
    """
    record = lookup.get_quality_baseline("line_02", "capsules")
    if record is None:
        pytest.skip("line_02/capsules 데이터가 아직 없다")

    assert isinstance(record, QualityBaselineRecord)
    assert record.stats, "stats 가 비어 있습니다."

    for metric, values in record.stats.items():
        assert "mean" in values and "std" in values, (
            f"{metric} 에 mean 또는 std 가 없습니다. "
            f"{{'mean': ..., 'std': ...}} 형태여야 합니다."
        )
        assert values["std"] > 0, (
            f"{metric} 의 std 가 {values['std']} 입니다. "
            f"0 이면 이탈 여부를 계산할 수 없습니다."
        )


def test_criteria_verdict_works(lookup):
    """판별 7번 — 면적이 판정으로 옮겨져야 한다."""
    rule = lookup.get_criteria("line_02", "capsules", "dent")
    if rule is None:
        pytest.skip("판정 기준 데이터가 아직 없다")

    assert isinstance(rule, CriteriaRule)
    assert rule.defect_area > 0, "defect_area 가 0 이하입니다."

    assert rule.verdict_for(rule.defect_area) == "defect"
    assert rule.verdict_for(rule.defect_area * 2) == "defect"
    assert rule.verdict_for(0) == "pass"

    if rule.review_area is not None:
        assert rule.review_area <= rule.defect_area, (
            "review_area 가 defect_area 보다 큽니다. "
            "review 는 불량 기준보다 낮은 경계 구간입니다."
        )


def test_criteria_respects_as_of_date(lookup):
    """기준은 덮어쓰지 않고 쌓인다.

    과거 이슈를 지금 기준으로 판정하면 "기준 문제" 시나리오를 영영 찾지
    못한다. at 인자를 받는지만 확인한다.
    """
    try:
        lookup.get_criteria("line_02", "capsules", "dent", at=date(2026, 6, 1))
    except TypeError as exc:
        pytest.fail(
            f"get_criteria 가 at 인자를 받지 못합니다: {exc}\n"
            f"시그니처: get_criteria(line, object_name, defect_type=None, at=None)"
        )


def test_bank_profile_coverage(lookup):
    """판별 6번 — 현재 조건이 뱅크 구성에 있는가."""
    profile = lookup.get_bank_profile("v3")
    if profile is None:
        pytest.skip("뱅크 v3 데이터가 아직 없다")

    assert isinstance(profile, BankProfile)
    assert profile.conditions, (
        "conditions 가 비어 있습니다. 커버리지 부족을 판정할 수 없습니다.\n"
        '예: {"date": ["2026-06-01", ...], "lot": [...]}'
    )

    for key, values in profile.conditions.items():
        assert isinstance(values, list), f"conditions['{key}'] 는 리스트여야 합니다."

    # covers() 가 실제로 동작하는가
    some_key = next(iter(profile.conditions))
    known = profile.conditions[some_key]
    if known:
        assert profile.covers(some_key, known[0]) is True
    assert profile.covers(some_key, "이런값은없다__zzz") is False
    assert profile.covers("이런조건은없다__zzz", "아무거나") is False


def test_estimated_history_is_marked(lookup):
    """폴더 스캔으로 역추정한 이력은 추정으로 표시해야 한다.

    담당자 확인 후에만 확정으로 승격한다는 원칙이 있다. 값 자체를 강제하지는
    않고 필드가 있는지만 본다.
    """
    profile = lookup.get_bank_profile("v3")
    if profile is None:
        pytest.skip("뱅크 v3 데이터가 아직 없다")
    assert isinstance(profile.is_estimated, bool)


# ── 4. 진단에 실제로 물려서 도는가 ─────────────────────────────────────


def test_diagnosis_runs_with_this_lookup(lookup):
    """이 조회 계층으로 진단이 끝까지 도는가.

    개별 함수가 다 맞아도 조합해서 안 도는 경우가 있다. 마지막 확인이다.
    """
    from agents.diagnose import collect_evidence, decide
    from inspection.types import InferenceResult, NearestMatch, PatchRef

    threshold = lookup.get_threshold("line_02", "capsules", "v3")
    profile = lookup.get_bank_profile("v3")
    criteria = lookup.get_criteria("line_02", "capsules", "dent")

    match = NearestMatch(
        query=PatchRef("q.png", 1, 2, 10),
        bank=PatchRef("n.png", 4, 5, 37),
        distance=2.5,
        bank_row_index=1,
    )
    inference = InferenceResult(
        image="q.png", score=1.9, max_patch_distance=1.9,
        grid_h=8, grid_w=8, matches=[match], bank_version="v3",
    )

    evidence = collect_evidence(
        inference=inference,
        threshold=threshold,
        bank_profile=profile,
        condition_key="date",
        condition_value="2026-06-01",
        criteria=criteria,
        defect_area=200.0,
    )
    result = decide(evidence)

    assert len(evidence) == 7, "판별 항목이 7개가 아닙니다."
    # 시각 언어 모델 근거가 없으므로 판정은 보류되는 것이 정상이다.
    assert result.cause is None or result.cause in {
        "threshold", "bank_contamination", "coverage_gap",
        "normal_overlap", "equipment_optics", "criteria",
    }
