"""예약 실행 스케줄러 — 지정 시각에 쌓인 것을 스스로 처리한다. 작업 21.

**과제 이름이 "자율 운영"인데 사람이 버튼을 눌러야만 돌면 자율이 아니다.**
야간에 쌓인 생산분을 아침에 보고받는 그림이 이 모듈이다.

    from scheduler import due, run_nightly

    if due(now, at=time(7, 0), last_run=어제):
        report = run_nightly(factory, lookup, adapters)

**배포는 하지 않는다.** 진단·큐레이션·재구성 후보 생성·게이트·섀도까지
자동으로 가고 **승인 요청 문서에서 멈춘다.** 품질 검사 설비의 특성상
의도적으로 배제한 경계이며, 이 경계 자체가 제안의 설득 근거다.
"""

from .nightly import NightlyReport, PendingIssue, run_nightly, scan_pending
from .schedule import due, next_run

__all__ = [
    "NightlyReport",
    "PendingIssue",
    "due",
    "next_run",
    "run_nightly",
    "scan_pending",
]
