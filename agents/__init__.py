"""에이전트 계층.

adapters  언어 모델·시각 언어 모델 실행 환경 (설정으로 교체)
vision    시각 언어 모델을 쓰는 두 판별 — 항목 1번과 5번
ontology  원인·판별·조치 체계를 모델이 조회하게. 판정 권한은 없다
"""

from .adapters import ModelAdapter, build_adapters, load_config
from .curate import AdditionRequest, CurationPlan, RemovalCandidate, plan_curation
from .rebuild import (
    DirectoryImageSource,
    ImageSource,
    RebuildRecord,
    RebuildResult,
    compare_banks,
    execute_rebuild,
)
from .gate import (
    CheckResult,
    GateCriteria,
    GateResult,
    ReproducibilityResult,
    check_reproducibility,
    evaluate_gate,
)
from .release import ReleasePackage, prepare_release, write_approval_document
from .tools import AgentRun, Tool, ToolRegistry, ToolResult, run_agent
from .diagnose import (
    CAUSE_LABEL_KO,
    REBUILD_REQUIRED,
    DiagnosisResult,
    Evidence,
    collect_evidence,
    decide,
    narrate,
)
from .ontology import (
    CAUSES,
    CHECKS,
    action_label,
    describe_cause,
    describe_check,
    lookup_ontology,
    overview,
)
from .vision import VisionJudgment, cause_from_patch_judgment, judge_bank_patch, judge_defect_visible

__all__ = [
    "ModelAdapter",
    "AdditionRequest",
    "CurationPlan",
    "RemovalCandidate",
    "plan_curation",
    "DirectoryImageSource",
    "ImageSource",
    "RebuildRecord",
    "RebuildResult",
    "compare_banks",
    "execute_rebuild",
    "CheckResult",
    "GateCriteria",
    "GateResult",
    "ReproducibilityResult",
    "check_reproducibility",
    "evaluate_gate",
    "ReleasePackage",
    "prepare_release",
    "write_approval_document",
    "AgentRun",
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "run_agent",
    "CAUSE_LABEL_KO",
    "REBUILD_REQUIRED",
    "DiagnosisResult",
    "Evidence",
    "collect_evidence",
    "decide",
    "narrate",
    "CAUSES",
    "CHECKS",
    "action_label",
    "describe_cause",
    "describe_check",
    "lookup_ontology",
    "overview",
    "build_adapters",
    "load_config",
    "VisionJudgment",
    "cause_from_patch_judgment",
    "judge_bank_patch",
    "judge_defect_visible",
]
