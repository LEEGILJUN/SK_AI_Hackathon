"""배포된 뱅크가 무엇으로 만들어졌는지 되짚는다 — 작업 10.

    .venv/bin/python scripts/scan_bank_history.py <폴더>
    .venv/bin/python scripts/scan_bank_history.py <폴더> --diff v1 v2

폴더 구조를 전제하지 않는다. 아래를 전부 걸어 뱅크 이력 파일과 벡터 파일이
있는 자리를 찾는다.

**복원한 것은 추정으로 표시한다.** 이력 파일이 있으면 확정이고, 벡터만
있으면 이웃 이미지를 후보로 잡은 추정이다. 확정으로 올리는 것은 사람이
한다 — 역추정한 이력을 확정처럼 쓰면 그 위에서 내린 판단이 전부 흔들린다.

읽기만 한다. 아무것도 고치지 않는다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indexer import scan_history  # noqa: E402
from indexer.scan import summarise  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="뱅크 구성 이력 복원")
    parser.add_argument("root", help="훑을 폴더")
    parser.add_argument("--diff", nargs=2, metavar=("이전", "이후"),
                        help="두 버전의 구성 차이")
    parser.add_argument("--images", action="store_true",
                        help="복원한 이미지 목록도 찍는다")
    args = parser.parse_args()

    result = scan_history(args.root)
    print(summarise(result))

    if args.images:
        for record in result.records:
            print(f"\n[{'확정' if not record.is_estimated else '추정'}] "
                  f"{record.bank_version} — {record.directory}")
            for line in record.evidence:
                print(f"    · {line}")
            for path in record.images[:20]:
                print(f"      {path}")
            if len(record.images) > 20:
                print(f"      … 외 {len(record.images) - 20}장")

    if args.diff:
        before, after = args.diff
        diff = result.diff(before, after)
        if diff is None:
            print(f"\n{before} 또는 {after} 를 찾지 못했다.")
            return 1
        print(f"\n{diff.describe()}")
        for path in diff.removed:
            print(f"  − {path}")
        for path in diff.added:
            print(f"  + {path}")
        if diff.is_estimated:
            print("\n**이 비교는 추정이다.** 한쪽 이력이 복원된 것이라 "
                  "담당자 확인이 필요하다.")

    if not result.records:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
