"""운영에 쓸 뱅크 판을 바꾼다 — 그리고 되돌린다. **사람이 실행한다.**

    python scripts/switch_bank.py --item pcb1-01
    python scripts/switch_bank.py --item pcb1-01 --to pcb1-01-v2_20260816-0910_c4f81c

인자 없이 부르면 저장된 판 목록과 지금 쓰는 판을 보여준다. `--to` 를 주면
바꾼다. 되돌리기도 같은 명령이다 — **이전 폴더 이름을 다시 주면 된다.**

── 왜 스크립트인가 ─────────────────────────────────────────────────────

`CURRENT` 가 가리키는 것이 실제 판정에 쓰이므로, 그것을 바꾸는 것은 배포다.
에이전트가 자동으로 하면 품질 검사 설비에 사람 확인 없이 새 모델이 들어간다.
`agents/release.py` 가 승인 요청 문서까지만 만드는 것과 같은 경계이고,
`tests/test_bank_store.py` 가 `agents/` 와 `scheduler/` 에서 `write_current`
를 부르지 않는지 검사한다.

**그래서 이 파일이 그 함수를 부르는 유일한 자리다.** 승인 요청 문서에 여기
찍히는 명령을 그대로 적어, 승인한 사람이 복사해 실행하면 되게 한다.

── 원복이 파일 이동이 아닌 이유 ────────────────────────────────────────

가리키는 이름 한 줄만 바꾼다. 파일을 옮기면 도중에 죽었을 때 어느 것이
운영본인지 알 수 없게 되고, 되돌릴 원본도 함께 사라진다.

읽는 것 말고는 `CURRENT` 한 파일만 고친다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inspection.store import (  # noqa: E402
    DEFAULT_STORE_ROOT,
    current_bank,
    list_banks,
    write_current,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="운영 뱅크 판 교체·원복")
    parser.add_argument("--item", required=True,
                        help="라인·품목 (예: pcb1-01). 저장소 폴더 이름과 같다")
    parser.add_argument("--to", default=None,
                        help="바꿀 판의 폴더 이름. 비우면 목록만 보여준다")
    parser.add_argument("--root", default=None,
                        help=f"저장소 자리. 비우면 {DEFAULT_STORE_ROOT}")
    return parser.parse_args()


def show(item: str, root: Path) -> int:
    stored = list_banks(item, root=root)
    if not stored:
        print(f"{item} 에 저장된 판이 없습니다. 자리: {root / item}")
        print("뱅크는 커밋되지 않으므로 먼저 구성해야 합니다.")
        return 1

    in_use = current_bank(item, root=root)
    print(f"{item} — 판 {len(stored)}개\n")
    print(f"  {'':2} {'판':<34} {'만든 때':<17} 설정")
    for bank in stored:
        mark = "→" if in_use is not None and bank.path == in_use.path else " "
        when = bank.built_at.strftime("%Y-%m-%d %H:%M")
        print(f"  {mark}  {bank.path.name:<34} {when:<17} {bank.config_id}")

    print()
    if in_use is None:
        print("지금 쓰는 판이 지정돼 있지 않습니다 (→ 없음).")
    else:
        print("→ 가 지금 판정에 쓰는 판입니다.")
    print("\n바꾸려면:")
    print(f"  python scripts/switch_bank.py --item {item} --to <판 이름>")
    return 0


def main() -> int:
    args = parse_args()
    root = Path(args.root) if args.root else DEFAULT_STORE_ROOT

    if args.to is None:
        return show(args.item, root)

    before = current_bank(args.item, root=root)
    try:
        write_current(args.item, args.to, root=root)
    except FileNotFoundError as exc:
        print(exc)
        return 1

    after = current_bank(args.item, root=root)
    print(f"{args.item} 운영 판을 바꿨습니다.")
    print(f"  전  {before.path.name if before else '없음'}")
    print(f"  후  {after.path.name if after else '없음'}")
    print("\n되돌리려면 이전 이름을 그대로 주면 됩니다:")
    if before is not None:
        print(f"  python scripts/switch_bank.py --item {args.item} --to {before.path.name}")
    print("\n**파일은 움직이지 않았습니다.** 가리키는 이름만 바뀌었습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
