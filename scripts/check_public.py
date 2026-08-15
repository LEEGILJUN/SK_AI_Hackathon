"""저장소를 공개해도 되는지 훑는다 — 예선 가점 항목이 깃허브 공개다.

**한 번 훑고 끝날 일이 아니다.** 문서와 주석이 계속 늘고, 사내 정보는
대개 무심코 들어간다. 실제로 한 건 걸린 적이 있다 — 팀원이 보낸
`pyproject.toml` 에 사내 저장소 주소가 있었다.

    .venv/bin/python scripts/check_public.py

**추적 중인 파일만 본다.** `.gitignore` 로 빠진 것은 공개 대상이 아니다.
다만 **커밋 이력도 함께 본다** — 지금 지웠어도 과거 커밋에 남아 있으면
공개하는 순간 보인다.

── 기계가 못 보는 것 ───────────────────────────────────────────────────

**실명은 여기서 판단하지 않는다.** 팀원 이름이 공개 저장소에 들어가는 것은
본인들이 정할 일이고, 해커톤 출품작이라 드러내는 편이 맞을 수도 있다.
몇 곳에 있는지만 세어 준다.

문장의 뜻도 못 본다. "현행 ~와 동일하게 맞춤" 같은 표현이 사내 시스템을
가리키는지는 사람이 읽어야 안다.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

OK, FAIL, WARN = "  ✓", "  ✗", "  ·"

#: 있으면 공개를 막는 것. 이름과 정규식.
BLOCKERS: list[tuple[str, str]] = [
    ("사설 IP 주소", r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
    ("사내 패키지 저장소", r"(?i)nexus|artifactory|jfrog|pypi-group-intern"),
    ("자격증명 값", r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{16,}"),
    ("이메일 주소", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("사번", r"사번\s*[:=]|\bEMP\d{5,}\b"),
    ("개인 절대 경로", r"/Users/[a-z]|C:\\\\Users\\\\|/home/[a-z]"),
]

#: 걸려도 되는 것. 검사기 자신과 예시·규칙 문서다.
ALLOWED = (
    "scripts/check_public.py",
    "noreply@anthropic.com",
    "example.com",
)

#: 사람이 읽어야 하는 것. 막지 않고 알린다.
NAMES = ("이길준", "장영진", "이동현")


def _run(*args: str) -> str:
    done = subprocess.run(args, cwd=REPO_ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    return done.stdout


def check_tracked() -> list[str]:
    """추적 중인 파일에 막을 것이 있는가."""
    problems: list[str] = []
    files = [f for f in _run("git", "ls-files").splitlines()
             if f.endswith((".py", ".md", ".yaml", ".yml", ".toml", ".txt",
                            ".json", ".jsonl", ".cfg", ".ini"))]

    for name in files:
        if any(a in name for a in ALLOWED):
            continue
        path = REPO_ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for label, pattern in BLOCKERS:
            for m in re.finditer(pattern, text):
                hit = m.group(0)
                if any(a in hit for a in ALLOWED):
                    continue
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{name}:{line}  [{label}] {hit[:60]}")
    return problems


def check_history() -> list[str]:
    """지금 지웠어도 과거 커밋에 남아 있으면 공개하는 순간 보인다.

    **`-S` 가 아니라 `-G` 를 쓴다.** `-S` 는 문자열을 그대로 받으므로 찾을
    값을 이 파일에 적어야 하고, 그러면 **검사기 자신이 이력에 그 값을 남긴다.**
    실제로 그렇게 만들었다가 자기 커밋을 잡았다.

    같은 이유로 이 파일을 검색 대상에서 뺀다. 패턴 자체는 남아 있어야 하므로
    파일을 빼는 것 말고는 방법이 없다.
    """
    problems: list[str] = []
    for label, pattern in BLOCKERS:
        if label in ("이메일 주소", "개인 절대 경로"):
            continue  # 이력에서 보기에는 잡음이 너무 많다
        found = _run(
            "git", "log", "--all", "--oneline", "-G", pattern,
            "--", ".", f":(exclude){Path(__file__).relative_to(REPO_ROOT).as_posix()}",
        ).strip()
        if found:
            first = found.splitlines()[0]
            problems.append(f"커밋 이력  [{label}]  {first}")
    return problems


def main() -> int:
    print("공개 준비 검사\n")

    tracked = check_tracked()
    if tracked:
        print(f"{FAIL} 추적 파일 — 공개를 막는 것 {len(tracked)}건")
        for p in tracked[:20]:
            print(f"      {p}")
    else:
        print(f"{OK} 추적 파일 — 사내 정보·자격증명 없음")

    history = check_history()
    if history:
        print(f"{FAIL} 커밋 이력 — {len(history)}건")
        for p in history:
            print(f"      {p}")
        print("      **지금 지워도 이력에 남습니다.** 이력을 고쳐야 합니다")
    else:
        print(f"{OK} 커밋 이력 — 과거 커밋에도 없음")

    counts = {n: len(_run("git", "grep", "-c", n).splitlines()) for n in NAMES}
    total = sum(counts.values())
    if total:
        print(f"{WARN} 실명이 {total}개 파일에 있습니다 — "
              + " · ".join(f"{n} {c}" for n, c in counts.items()))
        print("      **막지 않습니다.** 공개 저장소에 본인 이름을 넣을지는")
        print("      본인들이 정할 일이고, 해커톤 출품작이라 드러내는 편이")
        print("      맞을 수도 있습니다")

    print()
    if tracked or history:
        print("공개하기 전에 위 항목을 고치세요.")
        return 1
    print("공개를 막는 것은 없습니다.")
    print("**문장의 뜻은 기계가 못 봅니다** — 사내 시스템을 가리키는 표현이")
    print("없는지는 사람이 읽어야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
