"""PatchCore 추론과 최근접 패치 역추적.

진단 에이전트가 쓰는 도구 계층이다. 판별 항목 3번(이상 점수의 위치)과
4번(최근접 정상 패치가 무엇인가)을 여기서 계산한다.

사용 예:

    from inspection import PatchEmbedder, build_bank, score_image

    embedder = PatchEmbedder()
    bank = build_bank(normal_image_paths, embedder, bank_version="v3")
    bank.save("banks/line_02/v3")

    result = score_image(defect_image, bank, embedder)
    print(result.score, result.score_position(threshold=1.8))
    print(result.top_match.bank.source_image)   # 최근접 정상 패치의 출처
"""

from .bank import MemoryBank, build_bank, greedy_coreset
from .crop import crop_patch, crop_with_context, patch_box
from .device import available_devices, describe, pick_device
from .features import FeatureConfig, PatchEmbedder
from .quality import QualityAssessment, QualityMetrics, assess_quality, compute_baseline, compute_metrics
from .isolation import (
    IsolationScore,
    contamination_amplification,
    image_isolation,
    patch_isolation,
    suspect_images,
)
from .shadow import Disagreement, ShadowReport, shadow_compare
from .sweep import (
    FeasibilityVerdict,
    SweepPoint,
    ThresholdCurve,
    assess_threshold_feasibility,
    format_curve,
    sweep_from_results,
    sweep_thresholds,
)
from .trace import anomaly_map, bank_contribution, score_image, score_images
from .types import InferenceResult, NearestMatch, PatchRef

__all__ = [
    "MemoryBank",
    "crop_patch",
    "crop_with_context",
    "patch_box",
    "build_bank",
    "greedy_coreset",
    "available_devices",
    "describe",
    "pick_device",
    "FeatureConfig",
    "PatchEmbedder",
    "QualityAssessment",
    "QualityMetrics",
    "assess_quality",
    "compute_baseline",
    "compute_metrics",
    "IsolationScore",
    "contamination_amplification",
    "image_isolation",
    "patch_isolation",
    "suspect_images",
    "Disagreement",
    "ShadowReport",
    "shadow_compare",
    "FeasibilityVerdict",
    "SweepPoint",
    "ThresholdCurve",
    "assess_threshold_feasibility",
    "format_curve",
    "sweep_from_results",
    "sweep_thresholds",
    "anomaly_map",
    "bank_contribution",
    "score_image",
    "score_images",
    "InferenceResult",
    "NearestMatch",
    "PatchRef",
]
