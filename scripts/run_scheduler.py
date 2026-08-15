"""예약 실행 스케줄러를 한 번 돌린다 — 작업 21.

    .venv/bin/python scripts/run_scheduler.py            # 지금이 돌 차례면 돈다
    .venv/bin/python scripts/run_scheduler.py --now      # 시각 무시하고 즉시
    .venv/bin/python scripts/run_scheduler.py --at 07:00

**데몬이 아니다.** 백그라운드로 도는 프로세스를 두면 시연 중에 무엇이 언제
돌았는지 아무도 모르고 재현성 검사도 못 한다. cron 이나 화면 버튼이 이것을
부른다.

**배포하지 않는다.** 진단·큐레이션·재구성 후보·게이트·섀도까지 자동으로 가고
승인 요청 문서에서 멈춘다. 실제 장비 반영은 사람이 한다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.adapters import build_adapters  # noqa: E402
from app.pipeline import DemoFactory  # noqa: E402
from scheduler import due, next_run, run_nightly  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="야간 누적분 자동 점검")
    parser.add_argument("--at", default="07:00", help="매일 이 시각 이후 실행")
    parser.add_argument("--now", action="store_true", help="시각을 무시하고 즉시")
    args = parser.parse_args()

    hour, minute = (int(v) for v in args.at.split(":"))
    at, now = time(hour, minute), datetime.now()

    # 마지막으로 돈 시각을 파일에 남긴다. **이것이 없으면 "같은 날 두 번은 안
    # 돈다"가 시험에만 있고 실제로는 안 걸려서**, cron 이 매시 깨울 때마다
    # 같은 이슈가 다시 접수된다.
    stamp = REPO_ROOT / "runs" / "scheduler_last_run.txt"
    last_run = None
    if stamp.exists():
        try:
            last_run = datetime.fromisoformat(stamp.read_text().strip())
        except ValueError:
            pass  # 손상됐으면 안 돈 것으로 본다 — 건너뛰는 것보다 낫다.

    if not args.now and not due(now, at, last_run):
        when = "오늘은 이미 돌았습니다." if last_run and last_run.date() >= now.date() else ""
        print(f"아직 돌 차례가 아닙니다. {when} 다음 실행 {next_run(now, at):%Y-%m-%d %H:%M}")
        return 0

    factory = DemoFactory()
    try:
        from lookup.factory import FactoryLookup

        lookup = FactoryLookup()
        if lookup.manifest.empty:
            raise FileNotFoundError
    except (ImportError, FileNotFoundError):
        print("공장 데이터가 없어 목으로 돕니다 — 시연 결과로 쓰지 마세요.")
        from lookup import MockLookup

        lookup = MockLookup(catalog=factory.catalog, banks=factory.bank_versions())

    report = run_nightly(factory, lookup, build_adapters(), now=now)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(now.isoformat())
    print(report.describe())
    print(f"\n다음 실행 {next_run(now, at):%Y-%m-%d %H:%M}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
