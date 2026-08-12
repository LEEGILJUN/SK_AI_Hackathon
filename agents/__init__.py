"""에이전트 계층.

adapters  언어 모델·시각 언어 모델 실행 환경 (설정으로 교체)
vision    시각 언어 모델을 쓰는 두 판별 — 항목 1번과 5번
"""

from .adapters import ModelAdapter, build_adapters, load_config
from .diagnose import (
    CAUSE_LABEL_KO,
    REBUILD_REQUIRED,
    DiagnosisResult,
    Evidence,
    collect_evidence,
    decide,
    narrate,
)
from .vision import VisionJudgment, cause_from_patch_judgment, judge_bank_patch, judge_defect_visible

__all__ = [
    "ModelAdapter",
    "CAUSE_LABEL_KO",
    "REBUILD_REQUIRED",
    "DiagnosisResult",
    "Evidence",
    "collect_evidence",
    "decide",
    "narrate",
    "build_adapters",
    "load_config",
    "VisionJudgment",
    "cause_from_patch_judgment",
    "judge_bank_patch",
    "judge_defect_visible",
]
