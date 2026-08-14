"""학습 이력 인덱서 — 폴더를 훑어 뱅크 구성을 복원한다.

배포된 뱅크가 무엇으로 만들어졌는지를 아무도 안 적어 두는 것이 현장의
기본값이다. 그 상태에서 **커버리지 부족**을 판정하려면 "이 조건의 정상
패치가 뱅크에 있었나"를 답할 수 있어야 하고, 그러려면 구성 이력이 필요하다.

`scan` 하나만 알면 된다.

    from indexer import scan_history

    history = scan_history("banks/")
    history.records          복원한 뱅크들
    history.diff("a", "b")   두 버전 사이에 무엇이 들고 났는가
"""

from .scan import BankRecord, ScanResult, VersionDiff, scan_history

__all__ = ["BankRecord", "ScanResult", "VersionDiff", "scan_history"]
