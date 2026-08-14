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

1. 함수 8개가 있고 이름·인자가 맞을 것
2. 못 찾으면 **예외가 아니라 None** 을 돌려줄 것
3. 돌려주는 값이 정해진 자료형일 것
4. `get_criteria(at=...)` 가 그 시점의 기준을 줄 것
5. `BankProfile.covers()` 가 동작할 것
6. MES 쪽 셋이 동작할 것 — `resolve_bank` `find_images` `defect_distribution`

하나씩 초록으로 만들면 됩니다. 한 번에 다 맞추려 하지 마세요.

## 나중에 추가된 MES 쪽 셋

접수된 이슈는 이미지가 아니라 **제품명이나 로트로** 옵니다. 그것을 이미지로
바꾸고 그 품목의 뱅크를 찾는 앞 단계가 필요해서 늘었습니다.

    resolve_bank(line, object_name)        이 품목은 어느 뱅크로 판정하는가
    find_images(...)                        이 제품·로트의 이미지가 무엇인가
    defect_distribution(...)                결함이 특정 라인·로트에 몰렸는가

**셋 다 조인으로 답합니다. 임베딩하지 마세요.** "3라인 A-217 로트 캡슐 이미지
목록"은 정확히 답할 문제고, 벡터 검색을 쓰면 비슷한 로트를 섞어 옵니다.

`find_images` 는 없으면 **빈 목록**, `defect_distribution` 은 없으면
**total=0 인 빈 집계**입니다(None 아님). "몰린 곳이 없다"와 "못 셌다"를
부르는 쪽이 구분할 수 있어야 하기 때문입니다.
"""

from __future__ import annotations

from datetime import date

import pytest

from lookup.base import (
    BankProfile,
    CriteriaRule,
    DefectDistribution,
    ImageRecord,
    PastIssue,
    QualityBaselineRecord,
    ThresholdRecord,
)
from lookup.mock import MockLookup

#: 시험이 물어볼 (라인, 품목). **출처는 `data/build_factory.py` 의
#: `VALID_LINES` 다.** 전에는 `line_02/capsules` 를 박아 뒀는데, 매핑이
#: pcb 로 옮겨간 뒤 그 조합이 없어져 **형태 시험 셋이 전부 건너뛰어졌다.**
#: 초록인데 아무것도 안 재고 있었다.
DEMO_ITEM = ("line_01", "pcb1")


def bank_version_of(lookup) -> str | None:
    """이 구현이 그 품목에 쓰는 뱅크 이름.

    **버전 문자열을 시험에 박지 않는다.** 목은 `v3`, 공장 구현은
    `pcb1-01-v1` 을 쓴다 — 이름 규칙은 구현의 자유이고 계약이 아니다.
    박아 두면 한쪽에서만 시험이 건너뛰어진다(실제로 그랬다).
    """
    profile = lookup.resolve_bank(*DEMO_ITEM)
    return profile.bank_version if profile else None

DEMO_DEFECT = "scratch"

REQUIRED_METHODS = (
    "get_threshold",
    "get_quality_baseline",
    "get_criteria",
    "get_bank_profile",
    "find_similar_issues",
    # MES 쪽. 판별 항목 앞 단계다.
    "resolve_bank",
    "find_images",
    "defect_distribution",
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
    """함수 8개가 있어야 접수부터 진단까지가 이어진다."""
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
    record = lookup.get_threshold(*DEMO_ITEM, bank_version_of(lookup) or "")
    if record is None:
        pytest.skip(f"{DEMO_ITEM} 임계값이 아직 없다")

    assert isinstance(record, ThresholdRecord)
    assert isinstance(record.value, (int, float)), "value 는 숫자여야 합니다."
    assert record.value > 0, "임계값이 0 이하입니다. 이상 점수와 비교할 값입니다."


def test_quality_baseline_shape(lookup):
    """판별 2번 — 화질 기준 분포.

    stats 는 inspection.quality.assess_quality 에 그대로 넘어갑니다.
    지표마다 mean 과 std 가 있어야 하고, std 가 0 이면 z 점수를 못 냅니다.
    """
    record = lookup.get_quality_baseline(*DEMO_ITEM)
    if record is None:
        pytest.skip(f"{DEMO_ITEM} 화질 기준이 아직 없다")

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
    rule = lookup.get_criteria(*DEMO_ITEM, DEMO_DEFECT)
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
        lookup.get_criteria(*DEMO_ITEM, DEMO_DEFECT, at=date(2026, 6, 1))
    except TypeError as exc:
        pytest.fail(
            f"get_criteria 가 at 인자를 받지 못합니다: {exc}\n"
            f"시그니처: get_criteria(line, object_name, defect_type=None, at=None)"
        )


def test_bank_profile_coverage(lookup):
    """판별 6번 — 현재 조건이 뱅크 구성에 있는가."""
    version = bank_version_of(lookup)
    profile = lookup.get_bank_profile(version) if version else None
    if profile is None:
        pytest.skip(f"{DEMO_ITEM} 뱅크 프로파일이 아직 없다")

    assert isinstance(profile, BankProfile)
    assert profile.conditions, (
        "conditions 가 비어 있습니다. 커버리지 부족을 판정할 수 없습니다.\n"
        '예: {"date": ["2026-06-01", ...], "lot": [...]}'
    )

    for key, values in profile.conditions.items():
        assert isinstance(values, list), f"conditions['{key}'] 는 리스트여야 합니다."

    # covers() 는 셋을 구분한다. **"기록하지 않은 축"과 "값이 없는 축"은
    # 다르다** — 모르는 것을 없다고 답하면 뱅크 프로파일이 그 축을 안
    # 담았다는 이유만으로 커버리지 부족으로 오진한다.
    some_key = next(iter(profile.conditions))
    known = profile.conditions[some_key]
    if known:
        assert profile.covers(some_key, known[0]) is True, "있는 값을 못 찾는다"
    assert profile.covers(some_key, "이런값은없다__zzz") is False, (
        "기록된 축에서 값을 못 찾으면 False 여야 한다"
    )
    assert profile.covers("이런조건은없다__zzz", "아무거나") is None, (
        "기록하지 않은 축은 None 이어야 한다. False 로 답하면 "
        "'모른다'가 '없다'가 되어 커버리지 부족으로 오진한다"
    )


def test_estimated_history_is_marked(lookup):
    """폴더 스캔으로 역추정한 이력은 추정으로 표시해야 한다.

    담당자 확인 후에만 확정으로 승격한다는 원칙이 있다. 값 자체를 강제하지는
    않고 필드가 있는지만 본다.
    """
    version = bank_version_of(lookup)
    profile = lookup.get_bank_profile(version) if version else None
    if profile is None:
        pytest.skip(f"{DEMO_ITEM} 뱅크 프로파일이 아직 없다")
    assert isinstance(profile.is_estimated, bool)


# ── 4. 진단에 실제로 물려서 도는가 ─────────────────────────────────────


def test_diagnosis_runs_with_this_lookup(lookup):
    """이 조회 계층으로 진단이 끝까지 도는가.

    개별 함수가 다 맞아도 조합해서 안 도는 경우가 있다. 마지막 확인이다.
    """
    from agents.diagnose import collect_evidence, decide
    from inspection.types import InferenceResult, NearestMatch, PatchRef

    version = bank_version_of(lookup) or ""
    threshold = lookup.get_threshold(*DEMO_ITEM, version)
    profile = lookup.get_bank_profile(version)
    criteria = lookup.get_criteria(*DEMO_ITEM, DEMO_DEFECT)

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


# ── 6. MES 쪽 셋 ───────────────────────────────────────────────────────


def test_resolve_bank_returns_profile_or_none(lookup):
    """품목마다 뱅크가 다르다. 없는 품목이면 None 이어야 한다.

    None 은 "그 품목에는 배포된 모델이 없다"는 뜻이고, 예외가 아니다.
    캡슐 뱅크를 PCB 에 돌려주는 것보다 없다고 말하는 편이 낫다.
    """
    try:
        missing = lookup.resolve_bank("없는라인", "없는품목")
    except Exception as exc:
        pytest.fail(
            f"없는 품목을 물었더니 예외가 났습니다: {type(exc).__name__}: {exc}\n"
            f"찾지 못하면 None 을 돌려주세요."
        )
    assert missing is None or isinstance(missing, BankProfile), (
        f"resolve_bank 가 {type(missing).__name__} 을 돌려줬습니다. "
        f"BankProfile 또는 None 이어야 합니다."
    )

    known = lookup.resolve_bank(*DEMO_ITEM)
    if known is not None:
        assert (known.line, known.object_name) == DEMO_ITEM, (
            "물어본 품목과 다른 프로파일이 왔습니다. "
            "뱅크가 하나뿐인 전제가 남아 있지 않은지 보세요."
        )


def test_find_images_returns_list_never_none(lookup):
    """못 찾으면 빈 목록이다. None 이 아니다."""
    try:
        result = lookup.find_images(line="없는라인", object_name="없는품목")
    except Exception as exc:
        pytest.fail(f"find_images 에서 예외가 났습니다: {type(exc).__name__}: {exc}")

    assert isinstance(result, list), (
        f"find_images 가 {type(result).__name__} 을 돌려줬습니다. list 여야 합니다."
    )
    assert all(isinstance(r, ImageRecord) for r in result), "항목은 ImageRecord 여야 합니다."


def test_find_images_without_any_filter_returns_nothing(lookup):
    """조건을 하나도 안 주면 전체를 훑어 오지 않는다.

    조건 없는 호출은 실수일 가능성이 높다. 공장 전체 이미지를 돌려주면
    그다음 단계가 통째로 막힌다.
    """
    assert lookup.find_images() == [], (
        "조건 없이 불렀는데 결과가 왔습니다. 빈 목록을 돌려주세요."
    )


def test_defect_distribution_is_never_none(lookup):
    """집계할 것이 없어도 빈 집계다. None 이 아니다.

    "몰린 곳이 없다"와 "못 셌다"는 다르다. 부르는 쪽이 구분할 수 있어야 한다.
    """
    try:
        dist = lookup.defect_distribution(line="없는라인", object_name="없는품목")
    except Exception as exc:
        pytest.fail(f"defect_distribution 에서 예외가 났습니다: {type(exc).__name__}: {exc}")

    assert isinstance(dist, DefectDistribution), (
        f"defect_distribution 이 {type(dist).__name__} 을 돌려줬습니다. "
        f"DefectDistribution 이어야 하며, 없으면 total=0 인 빈 집계입니다."
    )
    assert dist.total == 0
    assert dist.concentrated_in() == {}, "집계가 비었는데 몰린 곳이 나왔습니다."
    assert isinstance(dist.describe(), str)
