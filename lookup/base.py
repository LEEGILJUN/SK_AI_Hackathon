"""결정론적 조회 계층 — 인터페이스 (작업 5).

진단의 신뢰도는 벡터 검색이 아니라 여기서 나온다. 뱅크 구성 프로파일, 화질
기준 분포, 임계값, 판정 기준은 조인과 집계로 정확한 값을 얻어야 하는 대상이며
임베딩하면 오히려 정확도가 떨어진다.

**이 파일은 명세다.** 진단 에이전트는 LookupLayer 만 보고 동작하고, 실제
구현이 무엇인지 알지 못한다. 지금은 목 구현(mock.py)으로 루프를 닫아 두고,
이동현이 가상 공장 데이터에 붙은 구현으로 갈아끼운다. 그때 진단 코드는
고치지 않는다.

각 함수가 어느 판별 항목에 대응하는지 적어 둔다. 이 대응이 흐려지면
"이 조회가 왜 필요한가"를 나중에 아무도 답하지 못한다.

    판별 2번  화질 기준 분포     get_quality_baseline
    판별 3번  임계값             get_threshold
    판별 6번  뱅크 구성 이력     get_bank_profile
    판별 7번  판정 기준          get_criteria

판별 1·5번은 시각 언어 모델(agents/vision.py), 4번은 뱅크 역추적
(inspection/trace.py)이 맡는다. 여기서 다루지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Protocol, runtime_checkable


# ── 조회 결과 ───────────────────────────────────────────────────────────


@dataclass
class ThresholdRecord:
    """판별 3번 — 현재 운영 중인 이상 점수 임계값.

    임계값은 뱅크에 들어 있지 않다. 운영 설정이며 뱅크와 별개로 바뀐다.
    그래서 어느 뱅크 버전에 대해 언제부터 쓰인 값인지를 함께 남긴다.
    """

    line: str
    object_name: str
    bank_version: str
    value: float
    effective_from: date | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "effective_from": self.effective_from.isoformat() if self.effective_from else None
        }


@dataclass
class QualityBaselineRecord:
    """판별 2번 — 라인·객체별 화질 기준 분포.

    stats 는 지표별 {"mean", "std", ...}. inspection.quality.assess_quality 에
    그대로 넘길 수 있는 형태여야 한다.
    """

    line: str
    object_name: str
    stats: dict[str, dict[str, float]]
    computed_from: dict[str, Any] = field(default_factory=dict)
    tolerance_sigma: float = 3.0
    outlier_ratio_threshold: float = 0.30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriteriaRule:
    """판별 7번 — 마스크 면적 기반 판정 기준.

    기준은 덮어쓰지 않고 쌓인다. 과거 이슈를 다시 볼 때 그 시점의 기준으로
    판정해야 하므로 유효 기간이 붙는다.
    """

    rule_id: str
    line: str
    object_name: str
    defect_type: str | None
    defect_area: float
    review_area: float | None = None
    effective_from: date | None = None
    effective_to: date | None = None

    def verdict_for(self, area: float) -> str:
        """면적을 판정으로 옮긴다. defect | review | pass."""
        if area >= self.defect_area:
            return "defect"
        if self.review_area is not None and area >= self.review_area:
            return "review"
        return "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
        }


@dataclass
class BankProfile:
    """판별 6번 — 뱅크가 무엇으로 만들어졌는가.

    커버리지 부족을 가리는 근거다. 지금 문제가 난 조건(로트, 일자, 설비)의
    정상 이미지가 뱅크 구성에 들어갔는지를 여기서 확인한다.

    conditions
        뱅크를 구성한 이미지들이 가진 조건의 집합. 예를 들어
        {"date": ["2026-06-01", ...], "lot": [...], "equipment": [...]}
        진단은 "지금 조건이 이 집합에 있는가"만 물으면 된다.
    """

    bank_version: str
    line: str
    object_name: str
    source_image_count: int
    patch_count: int
    conditions: dict[str, list[str]] = field(default_factory=dict)
    built_at: date | None = None
    is_estimated: bool = False  # 폴더 스캔으로 역추정한 이력인가

    def covers(self, key: str, value: str) -> bool:
        """그 조건이 뱅크 구성에 포함됐는가."""
        values = self.conditions.get(key)
        if values is None:
            return False
        return value in values

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "built_at": self.built_at.isoformat() if self.built_at else None
        }


@dataclass
class PastIssue:
    """그래프 검색 결과 — 유사 사례 한 건.

    역할은 중복 작업 차단 하나다. 진단 근거로 쓰지 않는다. 진단의 근거는
    결정론적 조회에서 나오고, 이것은 "이미 해결된 건 아닌가"를 묻는 용도다.
    """

    issue_id: str
    line: str
    object_name: str
    cause: str
    action: str
    resolved: bool
    similarity: float
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 인터페이스 ──────────────────────────────────────────────────────────


@runtime_checkable
class LookupLayer(Protocol):
    """진단 에이전트가 부르는 조회 함수 전체.

    구현이 값을 찾지 못하면 예외를 내지 말고 None 을 돌려준다. 진단은
    "근거를 얻지 못했다"를 하나의 상태로 다뤄야 하며, 조회 실패로 멈추면
    안 된다. 근거가 비어 있는 것과 진단이 죽는 것은 다르다.
    """

    def get_threshold(
        self, line: str, object_name: str, bank_version: str
    ) -> ThresholdRecord | None:
        """판별 3번 — 현재 임계값."""
        ...

    def get_quality_baseline(
        self, line: str, object_name: str
    ) -> QualityBaselineRecord | None:
        """판별 2번 — 화질 기준 분포."""
        ...

    def get_criteria(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        at: date | None = None,
    ) -> CriteriaRule | None:
        """판별 7번 — 그 시점에 유효한 판정 기준.

        at 을 주면 그 날짜에 유효했던 기준을 돌려준다. 과거 이슈를 다시 볼 때
        지금 기준으로 판정하면 "기준 문제"를 영영 찾지 못한다.
        """
        ...

    def get_bank_profile(self, bank_version: str) -> BankProfile | None:
        """판별 6번 — 뱅크 구성 이력."""
        ...

    def find_similar_issues(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        limit: int = 5,
    ) -> list[PastIssue]:
        """유사 사례. 중복 차단 전용이며 진단 근거가 아니다."""
        ...
