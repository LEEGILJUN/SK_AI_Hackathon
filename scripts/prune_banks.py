"""오래된 뱅크 판을 지운다 — 사람이 실행한다.

    .venv/bin/python scripts/prune_banks.py                 무엇이 지워질지 보기만
    .venv/bin/python scripts/prune_banks.py --item pcb1-01  한 품목만
    .venv/bin/python scripts/prune_banks.py --apply         실제로 지운다

**기본은 시늉이다.** 지우는 것은 되돌릴 수 없으므로 `--apply` 를 붙여야
실제로 지운다. `scripts/switch_bank.py` 가 사람 손을 거치게 만든 것과 같은
이유다.

── 왜 필요한가 ─────────────────────────────────────────────────────────

재구성을 한 번 돌릴 때마다 판이 하나씩 생긴다. 측정하느라 열아홉 번
돌리면 열아홉 개가 쌓이고, 판 하나가 100MB 쯤이라 금세 기가 단위가 된다.
4090 에서 실제로 그랬다.

── 무엇을 남기는가 ─────────────────────────────────────────────────────

최신 `--keep` 개와 **`CURRENT` 가 가리키는 판**을 남긴다. 둘은 다를 수
있다 — 되돌린 뒤라면 운영본이 최신이 아니다. 그때 최신만 남기면 지금
판정에 쓰는 뱅크를 지우게 되고, 시연이 "배포된 뱅크가 없다"로 2단계에서
죽는다.

`--keep 3` 이면 지금 판과 그 앞 두 판이 남아 두 단계까지 되돌아갈 수 있다.
**되돌릴 수 있는 범위를 남기는 것이 이 값의 뜻이다.**
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inspection.store import DEFAULT_STORE_ROOT, prune_banks  # noqa: E402


def items(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description="오래된 뱅크 판 정리")
    parser.add_argument("--item", help="품목 열쇠(pcb1-01). 비우면 전부")
    parser.add_argument("--keep", type=int, default=3,
                        help="남길 최신 판 수. 운영본은 이와 별개로 남는다")
    parser.add_argument("--apply", action="store_true",
                        help="실제로 지운다. 없으면 무엇이 지워질지 보여주기만 한다")
    parser.add_argument("--root", default=str(DEFAULT_STORE_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    targets = [args.item] if args.item else items(root)
    if not targets:
        print(f"뱅크 저장소가 비어 있다: {root}")
        return 0

    total_removed = 0
    total_bytes = 0
    for item_key in targets:
        result = prune_banks(item_key, keep=args.keep, root=root,
                             dry_run=not args.apply)
        print(f"\n{result.reason}")
        for name in result.kept:
            print(f"  남김  {name}")
        for name in result.removed:
            print(f"  {'지움' if args.apply else '지울 것'}  {name}")
        total_removed += len(result.removed)
        total_bytes += result.freed_bytes

    print(f"\n합계 {total_removed}개 · {total_bytes / (1024*1024):.0f}MB")
    if not args.apply and total_removed:
        print("\n**아직 아무것도 안 지웠다.** 위 목록이 맞으면 `--apply` 를 붙여 다시 실행할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
