"""임계값 스윕 — 검출률과 과검률의 관계를 산출한다.

미검출이 났을 때 가장 먼저 나오는 말은 "임계값을 낮추면 되지 않나"다.
이 파일은 그 말이 맞는지 틀린지를 숫자로 답한다.

임계값을 낮춰가며 두 값을 함께 본다.
  검출률   불량 중 몇 %를 잡았는가
  과검률   양품 중 몇 %를 불량으로 잘못 판정했는가

둘은 같은 방향으로 움직인다. 그래서 "이 결함을 잡으려면 과검률이 얼마가
되는가"가 곧 임계값 조정의 대가다. 그 대가가 감당 가능하면 임계값 문제이고,
감당할 수 없으면 정상 분포 중첩이다. 후자는 데이터를 더 넣어도 해결되지
않으므로, 재학습을 시작하기 전에 여기서 걸러내야 한다.

이 판단이 없으면 해결되지 않을 재학습을 반복하게 된다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from .types import InferenceResult


@dataclass(frozen=True)
class SweepPoint:
    """임계값 한 지점에서의 성능."""

    threshold: float
    detection_rate: float  # 불량 중 검출된 비율
    false_positive_rate: float  # 양품 중 불량으로 판정된 비율
    detected: int
    missed: int
    false_positives: int
    true_negatives: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ThresholdCurve:
    """임계값 스윕 결과 전체.

    points 는 임계값 내림차순이다. 앞쪽이 엄격(검출률 낮고 과검 적음),
    뒤쪽이 느슨(검출률 높고 과검 많음)이다.
    """

    points: list[SweepPoint]
    normal_scores: list[float]
    defect_scores: list[float]
    current_threshold: float | None = None
    score_field: str = "score"

    # ── 조회 ────────────────────────────────────────────────────────────

    def at_threshold(self, threshold: float) -> SweepPoint:
        """임의의 임계값에서의 성능을 계산한다."""
        return _measure(np.asarray(self.normal_scores), np.asarray(self.defect_scores), threshold)

    def threshold_for_detection(self, target: float = 1.0) -> SweepPoint | None:
        """목표 검출률을 만족하는 지점 중 과검률이 가장 낮은 것.

        points 가 임계값 내림차순이므로, 조건을 만족하는 첫 지점이 곧
        가장 엄격한(=과검이 가장 적은) 지점이다.
        """
        for point in self.points:
            if point.detection_rate >= target - 1e-9:
                return point
        return None

    def auroc(self) -> float:
        """양품과 불량의 점수 분포가 얼마나 갈리는가. 1.0 이면 완전 분리.

        임계값과 무관한 값이라, 낮으면 임계값을 어디에 두어도 안 된다는
        뜻이 된다. 정상 분포 중첩의 신호다.
        """
        normal = np.asarray(self.normal_scores, dtype=np.float64)
        defect = np.asarray(self.defect_scores, dtype=np.float64)
        if normal.size == 0 or defect.size == 0:
            return float("nan")

        # 순위 기반(Mann-Whitney U). 동점은 평균 순위로 처리한다.
        combined = np.concatenate([defect, normal])
        order = combined.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, combined.size + 1, dtype=np.float64)

        sorted_vals = combined[order]
        start = 0
        for i in range(1, combined.size + 1):
            if i == combined.size or sorted_vals[i] != sorted_vals[start]:
                if i - start > 1:
                    ranks[order[start:i]] = ranks[order[start:i]].mean()
                start = i

        defect_rank_sum = ranks[: defect.size].sum()
        u = defect_rank_sum - defect.size * (defect.size + 1) / 2
        return float(u / (defect.size * normal.size))

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_field": self.score_field,
            "current_threshold": self.current_threshold,
            "auroc": self.auroc(),
            "normal_count": len(self.normal_scores),
            "defect_count": len(self.defect_scores),
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class FeasibilityVerdict:
    """임계값 조정으로 해결되는가에 대한 판정.

    진단 에이전트가 임계값 문제와 정상 분포 중첩을 가를 때 쓰는 근거다.
    reason 은 리포트에 그대로 실을 수 있게 한 문장으로 쓴다.
    """

    achievable: bool
    target_detection: float
    max_acceptable_fpr: float
    required_threshold: float | None
    resulting_fpr: float | None
    resulting_detection: float | None
    auroc: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 계산 ────────────────────────────────────────────────────────────────


def _measure(normal: np.ndarray, defect: np.ndarray, threshold: float) -> SweepPoint:
    """임계값 하나에서 검출률과 과검률을 센다. 판정은 점수 >= 임계값이면 불량."""
    detected = int((defect >= threshold).sum())
    missed = int(defect.size - detected)
    false_positives = int((normal >= threshold).sum())
    true_negatives = int(normal.size - false_positives)

    return SweepPoint(
        threshold=float(threshold),
        detection_rate=float(detected / defect.size) if defect.size else float("nan"),
        false_positive_rate=float(false_positives / normal.size) if normal.size else float("nan"),
        detected=detected,
        missed=missed,
        false_positives=false_positives,
        true_negatives=true_negatives,
    )


def sweep_thresholds(
    normal_scores: Sequence[float],
    defect_scores: Sequence[float],
    current_threshold: float | None = None,
    steps: int | None = None,
    score_field: str = "score",
) -> ThresholdCurve:
    """양품·불량 점수로 임계값 곡선을 만든다.

    steps
        None 이면 실제 점수값을 후보 임계값으로 쓴다. 곡선이 꺾이는 지점이
        정확히 그 값들이므로 근사가 아닌 정확한 곡선이 나온다.
        숫자를 주면 최소~최대 구간을 그만큼 균등 분할한다. 그래프를 그릴 때만
        쓰고, 판정에는 기본값을 쓴다.
    """
    normal = np.asarray(normal_scores, dtype=np.float64)
    defect = np.asarray(defect_scores, dtype=np.float64)
    if normal.size == 0 or defect.size == 0:
        raise ValueError("양품과 불량 점수가 모두 있어야 곡선을 만들 수 있다.")

    if steps is None:
        combined = np.concatenate([normal, defect])
        candidates = np.unique(combined)
        # 가장 높은 점수보다 위쪽 한 점을 더해 검출률 0 지점을 남긴다.
        candidates = np.append(candidates, candidates[-1] + max(1e-9, abs(candidates[-1]) * 1e-6))
    else:
        combined = np.concatenate([normal, defect])
        candidates = np.linspace(combined.min(), combined.max(), steps)

    points = [_measure(normal, defect, t) for t in candidates[::-1]]

    return ThresholdCurve(
        points=points,
        normal_scores=[float(v) for v in normal],
        defect_scores=[float(v) for v in defect],
        current_threshold=current_threshold,
        score_field=score_field,
    )


def sweep_from_results(
    normal_results: Sequence[InferenceResult],
    defect_results: Sequence[InferenceResult],
    current_threshold: float | None = None,
    use_raw_distance: bool = False,
    steps: int | None = None,
) -> ThresholdCurve:
    """추론 결과 목록에서 바로 곡선을 만든다.

    use_raw_distance
        가중 보정 전의 원 거리로 스윕한다. 보정은 뱅크 밀도에 따라 점수를
        깎으므로, 뱅크가 바뀌는 상황을 비교할 때는 보정 전 값이 척도로
        일관적이다.
    """
    field = "max_patch_distance" if use_raw_distance else "score"
    pick = lambda r: getattr(r, field)  # noqa: E731
    return sweep_thresholds(
        [pick(r) for r in normal_results],
        [pick(r) for r in defect_results],
        current_threshold=current_threshold,
        steps=steps,
        score_field=field,
    )


def assess_threshold_feasibility(
    curve: ThresholdCurve,
    target_detection: float = 1.0,
    max_acceptable_fpr: float = 0.05,
) -> FeasibilityVerdict:
    """임계값 조정만으로 목표 검출률에 닿을 수 있는지 판정한다.

    이것이 임계값 문제와 정상 분포 중첩을 가르는 지점이다.
      닿는다   → 임계값 재조정으로 해결. 뱅크 재구성 불필요
      못 닿는다 → 점수 분포가 겹쳐 있다는 뜻. 데이터를 더 넣어도 해결되지
                 않으므로 기준 재정의·전용 판별 로직·촬영 개선으로 넘긴다

    max_acceptable_fpr 은 현업이 정하는 값이다. 기본값 5%는 자리표시이며
    라인별 기준은 판정 기준 테이블에서 가져와야 한다.
    """
    auroc = curve.auroc()
    point = curve.threshold_for_detection(target_detection)

    if point is None:
        return FeasibilityVerdict(
            achievable=False,
            target_detection=target_detection,
            max_acceptable_fpr=max_acceptable_fpr,
            required_threshold=None,
            resulting_fpr=None,
            resulting_detection=None,
            auroc=auroc,
            reason=(
                f"어떤 임계값으로도 검출률 {target_detection:.0%}에 닿지 못합니다. "
                f"점수 분포가 겹쳐 있어(AUROC {auroc:.3f}) 임계값 조정으로는 해결되지 않습니다."
            ),
        )

    achievable = point.false_positive_rate <= max_acceptable_fpr
    if achievable:
        reason = (
            f"임계값을 {point.threshold:.4f}로 내리면 검출률 {point.detection_rate:.0%}를 "
            f"과검률 {point.false_positive_rate:.1%}로 달성합니다. "
            f"허용 과검률 {max_acceptable_fpr:.1%} 안이므로 임계값 조정으로 해결됩니다."
        )
    else:
        reason = (
            f"검출률 {target_detection:.0%}를 달성하려면 임계값을 {point.threshold:.4f}까지 "
            f"내려야 하고, 그때 과검률이 {point.false_positive_rate:.1%}가 됩니다. "
            f"허용 과검률 {max_acceptable_fpr:.1%}를 넘으므로 임계값 조정으로는 해결되지 않습니다. "
            f"(AUROC {auroc:.3f})"
        )

    return FeasibilityVerdict(
        achievable=achievable,
        target_detection=target_detection,
        max_acceptable_fpr=max_acceptable_fpr,
        required_threshold=point.threshold,
        resulting_fpr=point.false_positive_rate,
        resulting_detection=point.detection_rate,
        auroc=auroc,
        reason=reason,
    )


def format_curve(curve: ThresholdCurve, rows: int = 10) -> str:
    """곡선을 사람이 읽을 표로. 리포트와 콘솔 확인용."""
    points = curve.points
    if len(points) > rows:
        idx = np.linspace(0, len(points) - 1, rows).round().astype(int)
        points = [points[i] for i in dict.fromkeys(idx.tolist())]

    lines = [
        f"  {'임계값':>10}  {'검출률':>7}  {'과검률':>7}   검출/미검  과검",
        f"  {'-' * 10}  {'-' * 7}  {'-' * 7}   ---------  ----",
    ]
    for p in points:
        mark = ""
        if curve.current_threshold is not None and abs(p.threshold - curve.current_threshold) < 1e-9:
            mark = "  ← 현재"
        lines.append(
            f"  {p.threshold:>10.4f}  {p.detection_rate:>6.1%}  {p.false_positive_rate:>6.1%}   "
            f"{p.detected:>4}/{p.missed:<4}  {p.false_positives:>4}{mark}"
        )
    return "\n".join(lines)
