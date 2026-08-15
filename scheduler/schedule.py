"""지금이 돌 차례인가 — 시각 판정만 한다.

**데몬을 만들지 않는다.** 백그라운드로 도는 프로세스를 두면 시연 중에 무엇이
언제 돌았는지 아무도 모르고, 재현성 검사도 할 수 없다. 여기는 "돌 차례인가"를
답하는 순수 함수이고, 실제로 부르는 것은 바깥이다(cron, 화면 버튼, 테스트).

시각을 인자로 받는 이유도 같다. `datetime.now()` 를 안에서 부르면 **시험이
시각에 따라 달라진다.**
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta


def due(now: datetime, at: time, last_run: datetime | None = None) -> bool:
    """지금 돌아야 하는가.

    at
        매일 이 시각 이후면 돌 차례다. 야간 누적을 아침에 처리하는 그림이라
        보통 이른 아침을 넣는다.
    last_run
        마지막으로 돈 시각. **같은 날 이미 돌았으면 안 돈다.**

    지연 실행을 허용한다 — 7시에 돌기로 했는데 9시에 깨어나도 그날 몫을
    처리한다. 놓친 날을 건너뛰면 그날 쌓인 것이 영영 처리되지 않는다.
    """
    if now.time() < at:
        return False
    if last_run is not None and last_run.date() >= now.date():
        return False
    return True


def next_run(now: datetime, at: time) -> datetime:
    """다음 실행 예정 시각. 화면에 "다음 실행 ○○"으로 띄운다."""
    today = datetime.combine(now.date(), at)
    return today if now < today else today + timedelta(days=1)
