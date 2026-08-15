"""다른 장비에 보낼 지시가 **실제로 실행 가능한지** 미리 검사한다.

    .venv/bin/python scripts/check_handoff.py 보낼글.md
    .venv/bin/python scripts/check_handoff.py -          # 표준입력

── 왜 있는가 ───────────────────────────────────────────────────────────

4090 에 `scripts/run_demo.py` 를 **세 번** 지시했는데 저장소에 만들어진 적이
없었다. 그쪽은 매번 대신할 스크립트를 조립했고, 그래서 화면에 새로 붙은
표시가 그쪽 출력에 안 나왔다. 알아채는 데 이틀이 걸렸다.

왕복 한 번이 5~13분이다. **없는 명령을 보내는 것은 그 시간을 통째로 버리는
것이고, 받는 쪽은 자기가 뭘 잘못했나 먼저 의심한다.**

── 무엇을 보는가 ───────────────────────────────────────────────────────

글에서 파이썬 스크립트 실행을 찾아 **파일이 실제로 있는지** 본다. 있으면
`--help` 로 인자까지 받아지는지 본다. 없는 것을 찾으면 실패한다.

`git pull` 로 받으라고 적은 커밋 해시가 **원격에 실제로 올라가 있는지**도
본다. 아직 push 안 한 해시를 적어 보내면 그쪽은 "안 받아진다"로 막힌다.

── 무엇을 못 보는가 ────────────────────────────────────────────────────

**내용이 맞는지는 못 본다.** 명령이 존재하고 인자를 받는다는 것뿐이다.
숫자가 최신인지, 주장이 코드와 맞는지는 여전히 사람이 판단한다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: `python scripts/xxx.py --flag` 형태를 찾는다. 앞에 붙은 실행 경로는 무엇이든 좋다.
_RUN = re.compile(r"(?:^|\s)(?:[\w./\\%-]*python(?:\.exe)?)\s+(-m\s+)?([\w./\\-]+\.py|[\w.]+)([^\n`]*)")

#: 7자리 이상 16진수 — 커밋 해시로 본다.
_HASH = re.compile(r"\b([0-9a-f]{7,40})\b")


def _commands(text: str) -> list[tuple[str, str, str]]:
    found = []
    for match in _RUN.finditer(text):
        module, target, rest = match.group(1), match.group(2), match.group(3)
        found.append((module or "", target, rest.strip()))
    return found


def _check_script(target: str, arguments: str) -> tuple[bool, str]:
    path = REPO_ROOT / target
    if not path.exists():
        return False, f"파일이 없다 — {target}"

    flags = [a for a in arguments.split() if a.startswith("--")]
    if not flags:
        return True, "있음"

    try:
        helped = subprocess.run(
            [sys.executable, str(path), "--help"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, cwd=REPO_ROOT,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return True, f"있음 (인자 확인 못 함: {exc})"

    unknown = [f for f in flags if f not in helped.stdout]
    if unknown:
        return False, f"받지 않는 인자 — {' '.join(unknown)}"
    return True, f"있음 · 인자 {len(flags)}개 확인"


def _check_hashes(text: str) -> list[tuple[bool, str]]:
    results = []
    for raw in dict.fromkeys(_HASH.findall(text)):
        try:
            remote = subprocess.run(
                ["git", "branch", "-r", "--contains", raw],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, cwd=REPO_ROOT,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if remote.returncode != 0:
            continue                      # 커밋이 아니다 — 그냥 16진수 문자열
        if remote.stdout.strip():
            results.append((True, f"{raw} — 원격에 있음"))
        else:
            results.append((False, f"{raw} — **아직 push 안 됨**. 받는 쪽이 못 받는다"))
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    source = sys.argv[1]
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")

    print("── 보낼 명령 ─────────────────────────────────────────────")
    failed = 0
    seen: set[str] = set()
    for module, target, arguments in _commands(text):
        if module:
            continue                      # `python -m pytest` 같은 것은 파일이 아니다
        key = f"{target} {arguments}"
        if key in seen:
            continue
        seen.add(key)
        ok, note = _check_script(target, arguments)
        print(f"  {'✓' if ok else '✗'} {target:<38} {note}")
        failed += int(not ok)

    hashes = _check_hashes(text)
    if hashes:
        print("\n── 커밋 해시 ─────────────────────────────────────────────")
        for ok, note in hashes:
            print(f"  {'✓' if ok else '✗'} {note}")
            failed += int(not ok)

    print()
    if failed:
        print(f"{failed}개가 실행 불가능합니다. **보내기 전에 고치세요.**")
        print("받는 쪽은 5~13분을 버리고, 자기가 뭘 잘못했나 먼저 의심합니다.")
        return 1

    print("명령은 전부 실행 가능합니다.")
    print("**내용이 맞는지는 못 봤습니다** — 숫자가 최신인지, 주장이 코드와")
    print("맞는지는 사람이 판단해야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
