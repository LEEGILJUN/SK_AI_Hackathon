"""임계값 스윕 검증.

이 파일이 지켜야 하는 것은 두 가지다.

  1. 곡선 자체가 정확한가 — 검출률·과검률 계산이 손으로 센 것과 맞는가
  2. **판정이 두 방향으로 갈리는가** — 분리된 분포에서는 "임계값으로 해결된다",
     겹친 분포에서는 "해결되지 않는다"가 나와야 한다

2번이 본질이다. 늘 "해결된다"고 답하면 정상 분포 중첩을 놓치고 해결되지 않을
재학습을 반복하게 된다. 늘 "안 된다"고 답하면 간단한 임계값 문제를 키운다.
"""

from __future__ import annotations

import pytest

from inspection import (
    assess_threshold_feasibility,
    format_curve,
    sweep_from_results,
    sweep_thresholds,
)
from inspection.types import InferenceResult


# ── 곡선 계산 ──────────────────────────────────────────────────────────


def test_curve_counts_match_hand_calculation():
    """손으로 셀 수 있는 작은 예로 계산을 검증한다."""
    normal = [0.1, 0.2, 0.3]
    defect = [0.4, 0.5, 0.6]
    curve = sweep_thresholds(normal, defect)

    # 임계값 0.4 → 불량 3건 모두 검출, 양품 과검 0
    point = curve.at_threshold(0.4)
    assert point.detected == 3 and point.missed == 0
    assert point.false_positives == 0 and point.true_negatives == 3
    assert point.detection_rate == 1.0
    assert point.false_positive_rate == 0.0

    # 임계값 0.25 → 불량 전건 검출, 양품 1건(0.3) 과검
    point = curve.at_threshold(0.25)
    assert point.detection_rate == 1.0
    assert point.false_positives == 1
    assert point.false_positive_rate == pytest.approx(1 / 3)

    # 임계값 0.55 → 불량 1건(0.6)만 검출
    point = curve.at_threshold(0.55)
    assert point.detected == 1
    assert point.detection_rate == pytest.approx(1 / 3)


def test_points_are_ordered_from_strict_to_loose():
    """임계값 내림차순이어야 threshold_for_detection 이 최소 과검을 고른다."""
    curve = sweep_thresholds([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    thresholds = [p.threshold for p in curve.points]
    assert thresholds == sorted(thresholds, reverse=True)

    # 느슨해질수록 검출률과 과검률은 줄지 않는다
    detection = [p.detection_rate for p in curve.points]
    fpr = [p.false_positive_rate for p in curve.points]
    assert detection == sorted(detection)
    assert fpr == sorted(fpr)


def test_auroc_separated_and_overlapped():
    """완전히 갈린 분포는 1.0, 완전히 겹친 분포는 0.5 부근."""
    separated = sweep_thresholds([0.1, 0.2, 0.3], [0.7, 0.8, 0.9])
    assert separated.auroc() == pytest.approx(1.0)

    identical = sweep_thresholds([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert identical.auroc() == pytest.approx(0.5)


def test_threshold_for_detection_picks_lowest_false_positive():
    """목표 검출률을 만족하는 지점 중 과검이 가장 적은 것을 골라야 한다."""
    curve = sweep_thresholds([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    point = curve.threshold_for_detection(1.0)

    assert point is not None
    assert point.detection_rate == 1.0
    assert point.false_positive_rate == 0.0
    # 0.4 보다 더 내려갈 이유가 없다
    assert point.threshold == pytest.approx(0.4)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="양품과 불량 점수가 모두"):
        sweep_thresholds([], [0.4])


# ── 판정: 임계값 문제인가 정상 분포 중첩인가 ───────────────────────────


def test_separated_distribution_is_solvable_by_threshold():
    """분포가 갈려 있으면 임계값 조정으로 해결된다고 판정해야 한다."""
    curve = sweep_thresholds([0.10, 0.15, 0.20, 0.18, 0.12], [0.80, 0.85, 0.90])
    verdict = assess_threshold_feasibility(curve, target_detection=1.0, max_acceptable_fpr=0.05)

    assert verdict.achievable is True
    assert verdict.resulting_fpr == 0.0
    assert verdict.required_threshold is not None
    # **어미까지 붙잡지 않는다.** 문안 말투를 다듬는 것이 시험을 깨는
    # 일이 되면, 고쳐야 할 문장을 못 고치고 그대로 두게 된다.
    assert "임계값 조정으로 해결" in verdict.reason


def test_overlapped_distribution_is_not_solvable_by_threshold():
    """겹친 분포에서는 해결되지 않는다고 판정하고, 대가를 숫자로 제시해야 한다.

    정상 분포 중첩 시나리오의 핵심이다. 여기서 achievable=True 가 나오면
    해결되지 않을 재학습을 승인하게 된다.
    """
    # 불량 점수가 양품 분포 한가운데 묻혀 있다
    normal = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90]
    defect = [0.45, 0.52, 0.58]
    curve = sweep_thresholds(normal, defect)

    verdict = assess_threshold_feasibility(curve, target_detection=1.0, max_acceptable_fpr=0.05)

    assert verdict.achievable is False
    # 전건을 잡으려면 과검을 크게 감수해야 한다는 사실이 숫자로 나와야 한다
    assert verdict.resulting_fpr is not None and verdict.resulting_fpr > 0.05
    assert "해결되지 않" in verdict.reason
    assert f"{verdict.resulting_fpr:.1%}" in verdict.reason


def test_unreachable_target_reports_clearly():
    """어떤 임계값으로도 목표에 못 닿는 경우."""
    # 불량 최저점이 양품 최저점보다 낮아, 전건 검출은 전건 과검을 동반한다
    curve = sweep_thresholds([0.50, 0.60], [0.10, 0.90])
    verdict = assess_threshold_feasibility(curve, target_detection=1.0, max_acceptable_fpr=0.01)

    assert verdict.achievable is False
    assert verdict.resulting_fpr is not None and verdict.resulting_fpr == 1.0


def test_acceptable_fpr_changes_the_verdict():
    """허용 과검률은 현업이 정하는 값이고, 그에 따라 판정이 뒤집힌다."""
    normal = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    defect = [0.55, 0.75, 0.85]
    curve = sweep_thresholds(normal, defect)

    strict = assess_threshold_feasibility(curve, 1.0, max_acceptable_fpr=0.05)
    lenient = assess_threshold_feasibility(curve, 1.0, max_acceptable_fpr=0.60)

    assert strict.achievable is False
    assert lenient.achievable is True
    # 필요한 임계값 자체는 허용치와 무관하게 같아야 한다
    assert strict.required_threshold == lenient.required_threshold


# ── 추론 결과와의 연결 ─────────────────────────────────────────────────


def _fake_result(image: str, score: float, raw: float) -> InferenceResult:
    return InferenceResult(
        image=image, score=score, max_patch_distance=raw, grid_h=2, grid_w=2
    )


def test_sweep_from_results_selects_score_field():
    """보정 점수와 원 거리 중 무엇으로 스윕할지 고를 수 있어야 한다."""
    normals = [_fake_result("n0.png", 0.2, 0.9), _fake_result("n1.png", 0.3, 0.95)]
    defects = [_fake_result("d0.png", 0.8, 1.0)]

    weighted = sweep_from_results(normals, defects)
    raw = sweep_from_results(normals, defects, use_raw_distance=True)

    assert weighted.score_field == "score"
    assert weighted.defect_scores == [0.8]
    assert raw.score_field == "max_patch_distance"
    assert raw.defect_scores == [1.0]


def test_format_curve_is_readable_and_bounded():
    """리포트에 실을 표. 행 수가 제한되어야 길어지지 않는다."""
    curve = sweep_thresholds([0.1, 0.2, 0.3], [0.4, 0.5, 0.6], current_threshold=0.5)
    text = format_curve(curve, rows=4)

    assert "임계값" in text and "검출률" in text and "과검률" in text
    assert "← 현재" in text
    assert len(text.splitlines()) <= 6  # 머리글 2줄 + 최대 4행
