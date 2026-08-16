"""결정론적 조회 계층 (작업 5).

진단의 신뢰도는 벡터 검색이 아니라 여기서 나온다. 인터페이스는 base.py 에
있고, 실제 구현은 데이터 담당이 가상 공장 데이터에 붙여 만든다. mock.py 는
그때까지 루프를 닫아 두기 위한 임시 대체물이며 시연에 쓰지 않는다.
"""

from .base import (
    BankProfile,
    CriteriaRule,
    LookupLayer,
    PastIssue,
    QualityBaselineRecord,
    ThresholdRecord,
)
from .mock import MockLookup, resolved_duplicate

__all__ = [
    "BankProfile",
    "CriteriaRule",
    "LookupLayer",
    "PastIssue",
    "QualityBaselineRecord",
    "ThresholdRecord",
    "MockLookup",
    "resolved_duplicate",
]
