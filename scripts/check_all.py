"""검사를 한 번에 — 커밋 전에 이것만 돌리면 된다.

    .venv/bin/python scripts/check_all.py

── 왜 묶었나 ────────────────────────────────────────────────────────────

검사기가 넷으로 흩어져 있어서 **하나씩 빼먹었다.** 2026-08-15 하루에만
세 번이다.

  · 라인↔품목 매핑을 바꾸고 그것을 가리키는 문자열을 안 훑었다
  · 시험 둘이 깨진 채로 커밋했다 (출력에 떠 있었는데 안 읽었다)
  · 공장 생성기를 다섯 곳 고치고 `check_factory.py` 를 한 번도 안 돌렸다

셋 다 아래 넷 중 하나가 잡았을 것이다. **묶어 두면 빼먹을 수가 없다.**

── 무엇을 보나 ─────────────────────────────────────────────────────────

  1. pytest              코드가 도는가
  2. check_docs          문서가 코드와 어긋나지 않는가
  3. check_factory       공장 데이터가 계약에 맞는가
  4. check_scenarios     시나리오 형식이 맞는가

3·4 는 데이터 파일이 없으면 건너뛴다 — 아직 안 만들었을 수 있고, 그때
빨간불을 내면 이 명령 자체를 안 쓰게 된다.

**여기서 초록이어도 "맞다"는 뜻은 아니다.** 기계가 볼 수 있는 것만 본다.
설명이 코드가 실제 하는 일과 같은지, 측정값이 최신인지는 사람이 판단한다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

#: (이름, 명령, 이 파일이 없으면 건너뜀)
CHECKS: list[tuple[str, list[str], Path | None]] = [
    ("테스트", [PYTHON, "-m", "pytest", "tests/", "-q"], None),
    ("문서", [PYTHON, "scripts/check_docs.py"], None),
    ("공장 데이터", [PYTHON, "scripts/check_factory.py"], REPO_ROOT / "data" / "manifest.csv"),
    ("시나리오", [PYTHON, "scripts/check_scenarios.py"], REPO_ROOT / "data" / "scenarios.yaml"),
]


def main() -> int:
    results: list[tuple[str, str]] = []
    for name, command, required in CHECKS:
        if required is not None and not required.exists():
            results.append((name, f"건너뜀 — {required.relative_to(REPO_ROOT)} 없음"))
            continue

        print(f"\n{'━' * 62}\n  {name}\n{'━' * 62}")
        completed = subprocess.run(command, cwd=REPO_ROOT)
        results.append((name, "통과" if completed.returncode == 0 else "실패"))

    print(f"\n{'━' * 62}")
    failed = [name for name, verdict in results if verdict == "실패"]
    for name, verdict in results:
        mark = "✗" if verdict == "실패" else "✓"
        print(f"  {mark} {name:12s} {verdict}")

    if failed:
        print(f"\n{len(failed)}개가 실패했습니다: {', '.join(failed)}")
        print("**커밋하기 전에 고치세요.** 출력을 위로 올려 읽어 보십시오.")
        return 1

    print("\n전부 통과. 다만 기계가 볼 수 있는 것만 본 것입니다 —")
    print("설명이 코드와 같은지, 측정값이 최신인지는 사람이 판단합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
