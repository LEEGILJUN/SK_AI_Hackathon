"""화질 지표 계산 — 판별 항목 2번.

설비·광학 원인을 가르는 유일한 정량 근거다. 어느 시점부터 이미지가 흐려지거나
어두워졌다면 뱅크를 다시 만들 일이 아니라 설비를 봐야 한다. 그 구분을 여기서
계산한 값과 기준 분포의 대조로 한다.

지표 정의는 data/quality_baseline.yaml 과 짝을 이룬다. 그쪽이 "무엇을 재는가"의
단일 출처이고 여기는 계산 구현이다. 정의가 바뀌면 양쪽을 함께 고쳐야 한다.

판정을 이미지 한 장으로 하지 않는 것이 중요하다. 설비 문제는 어느 시점부터
지속되는 현상이므로 구간 단위로 본다. 한 장이 튀는 것은 늘 있는 일이다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

#: quality_baseline.yaml 의 metrics 키와 일치해야 한다.
METRIC_KEYS = ("brightness", "contrast", "sharpness", "noise")


@dataclass(frozen=True)
class QualityMetrics:
    """이미지 한 장의 화질 지표."""

    brightness: float
    contrast: float
    sharpness: float
    noise: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)


def _to_gray(image_path: str | Path) -> np.ndarray:
    with Image.open(image_path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32)


def compute_metrics(image_path: str | Path) -> QualityMetrics:
    """이미지 한 장의 화질 지표를 계산한다.

    brightness  그레이스케일 픽셀 평균 (0-255)
    contrast    그레이스케일 픽셀 표준편차
    sharpness   라플라시안 응답의 분산. 흐려지면 떨어진다
    noise       평탄한 영역의 국소 표준편차 중앙값

    noise 는 아직 잠정 정의다. quality_baseline.yaml 에 산출식이 TODO 로
    남아 있으며, 도메인 담당이 확정하면 여기를 그 정의로 맞춘다. 지금 구현은
    "무늬가 없는 곳에서도 남아 있는 흔들림"을 재는 통상적인 방식이다.
    """
    gray = _to_gray(image_path)

    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    # 국소 표준편차 지도를 만들고, 무늬가 적은 영역(하위 30%)의 중앙값을 쓴다.
    # 무늬가 강한 곳은 표준편차가 커서 노이즈와 구분되지 않기 때문이다.
    kernel = (7, 7)
    local_mean = cv2.blur(gray, kernel)
    local_sq_mean = cv2.blur(gray * gray, kernel)
    local_var = np.clip(local_sq_mean - local_mean * local_mean, 0.0, None)
    local_std = np.sqrt(local_var)
    flat_cutoff = np.percentile(local_std, 30)
    flat = local_std[local_std <= flat_cutoff]
    noise = float(np.median(flat)) if flat.size else float(np.median(local_std))

    return QualityMetrics(
        brightness=brightness, contrast=contrast, sharpness=sharpness, noise=noise
    )


@dataclass
class MetricDeviation:
    """지표 하나의 기준 대비 이탈."""

    metric: str
    value: float
    baseline_mean: float
    baseline_std: float
    z: float
    out_of_range: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityAssessment:
    """구간 단위 화질 판정 — 판별 항목 2번의 답.

    within_baseline 이 False 면 설비·광학을 의심할 근거가 된다.
    이미지 한 장이 아니라 구간의 이탈 비율로 판정한다.
    """

    image_count: int
    deviations: list[MetricDeviation]
    outlier_ratio: dict[str, float]
    within_baseline: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_count": self.image_count,
            "within_baseline": self.within_baseline,
            "outlier_ratio": self.outlier_ratio,
            "deviations": [d.to_dict() for d in self.deviations],
            "reason": self.reason,
        }


def assess_quality(
    image_paths: Sequence[str | Path],
    baseline: dict[str, dict[str, float]],
    tolerance_sigma: float = 3.0,
    outlier_ratio_threshold: float = 0.30,
    min_images: int = 1,
    metrics_required: int = 1,
) -> QualityAssessment:
    """이미지 구간의 화질이 기준 분포 안에 있는지 판정한다.

    baseline
        지표별 {"mean": ..., "std": ...}. quality_baseline.yaml 의 stats 를
        그대로 넣으면 된다.
    tolerance_sigma
        개별 이미지를 이탈로 볼 z 경계
    outlier_ratio_threshold
        구간 내 이탈 비율이 이 값을 넘으면 그 지표는 이탈로 본다
    metrics_required
        몇 개 지표가 동시에 이탈해야 설비 의심으로 판정할지

    이미지가 min_images 에 못 미치면 판정하지 않고 within_baseline=True 로
    둔다. 표본이 모자란 것을 이상으로 읽으면 멀쩡한 설비를 세우게 된다.
    """
    if not image_paths:
        raise ValueError("화질을 잴 이미지가 없다.")

    measured = [compute_metrics(p) for p in image_paths]

    outlier_ratio: dict[str, float] = {}
    deviations: list[MetricDeviation] = []

    for key in METRIC_KEYS:
        stats = baseline.get(key)
        if not stats or not stats.get("std"):
            continue

        mean = float(stats["mean"])
        std = float(stats["std"])
        values = np.array([m[key] for m in measured], dtype=np.float64)
        z_scores = (values - mean) / std

        outliers = np.abs(z_scores) > tolerance_sigma
        outlier_ratio[key] = float(outliers.mean())

        # 구간 대표값은 평균 z 로 남긴다. 리포트에 실을 값이다.
        deviations.append(
            MetricDeviation(
                metric=key,
                value=float(values.mean()),
                baseline_mean=mean,
                baseline_std=std,
                z=float(z_scores.mean()),
                out_of_range=outlier_ratio[key] > outlier_ratio_threshold,
            )
        )

    if len(measured) < min_images:
        return QualityAssessment(
            image_count=len(measured),
            deviations=deviations,
            outlier_ratio=outlier_ratio,
            within_baseline=True,
            reason=(
                f"이미지가 {len(measured)}장뿐이라 구간 판정을 하지 않았다 "
                f"(최소 {min_images}장 필요). 화질을 근거로 쓰지 않는다."
            ),
        )

    breached = [d for d in deviations if d.out_of_range]
    within = len(breached) < metrics_required

    if within:
        reason = "화질 지표가 기준 분포 안에 있다. 설비·광학 원인으로 볼 근거가 없다."
    else:
        detail = ", ".join(
            f"{d.metric} z={d.z:+.2f} (이탈 {outlier_ratio[d.metric]:.0%})" for d in breached
        )
        reason = f"화질 지표가 기준 분포를 벗어났다 — {detail}. 설비 점검이 필요하다."

    return QualityAssessment(
        image_count=len(measured),
        deviations=deviations,
        outlier_ratio=outlier_ratio,
        within_baseline=within,
        reason=reason,
    )


def compute_baseline(image_paths: Sequence[str | Path]) -> dict[str, dict[str, float]]:
    """기준 분포를 산출한다. quality_baseline.yaml 의 stats 에 넣을 형태.

    초기 뱅크 구간의 정상 이미지로만 돌려야 한다. 열화가 주입된 운영 구간이
    섞이면 기준 자체가 오염되어 설비 문제를 영영 잡지 못한다.
    """
    if not image_paths:
        raise ValueError("기준 분포를 낼 이미지가 없다.")

    measured = [compute_metrics(p) for p in image_paths]
    stats: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.array([m[key] for m in measured], dtype=np.float64)
        stats[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "p01": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
        }
    return stats
