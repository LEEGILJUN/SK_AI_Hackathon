"""개발 일지를 커밋 이력에서 만든다.

    .venv/bin/python scripts/build_devlog.py
    .venv/bin/python scripts/build_devlog.py --since 2026-08-21 --out 어디에.md

── 왜 손으로 안 쓰는가 ─────────────────────────────────────────────────

본선 제출 안내에 이렇게 적혀 있다.

    4번 폴더를 마감 직전에 몰아서 쓰는 경우.
    2주 뒤에 기억으로 복원한 개발 일지는 티가 납니다.

맞는 지적이다. 그리고 우리에게는 기억으로 복원할 필요가 없는 기록이 이미
있다. 이 저장소는 커밋 메시지에 **무엇이 틀렸고 왜 그렇게 고쳤는지**를
적어 왔다. 그것을 옮기는 것이 이 스크립트다.

**지어내지 않는다.** 커밋에 없는 문장은 여기서도 안 나온다. 날짜도 커밋
날짜 그대로다. 그래서 이 일지는 검증 가능하다 — 저장소에서 같은 명령으로
다시 만들면 같은 것이 나온다.

── 무엇을 넣고 무엇을 빼는가 ───────────────────────────────────────────

제목과 본문 첫 문단까지 넣는다. 본문 전체를 넣으면 읽을 수 없는 길이가
되고, 제목만 넣으면 "무엇을 왜"가 사라진다.

Co-Authored-By 같은 꼬리표는 뺀다. 사람이 읽을 것이 아니다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 커밋 하나를 구분하는 표시. 메시지 본문에 나올 수 없는 것이라야 한다.
SEPARATOR = "\x1e"
FIELD = "\x1f"

#: 본문에서 걷어 낼 꼬리표.
TRAILERS = ("Co-Authored-By:", "Claude-Session:", "Signed-off-by:")


def _run(args: list[str]) -> str:
    result = subprocess.run(
        args, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} 가 실패했다: {result.stderr.strip()}")
    return result.stdout


def _lead(body: str) -> str:
    """본문에서 첫 문단만. 꼬리표와 절 제목은 건너뛴다."""
    paragraph: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(tag) for tag in TRAILERS):
            break
        if stripped.startswith("#"):
            if paragraph:
                break
            continue
        if not stripped:
            if paragraph:
                break
            continue
        paragraph.append(stripped)
    return " ".join(paragraph)


def collect(since: str | None = None) -> "OrderedDict[str, list[dict]]":
    """커밋을 날짜별로 모은다. 오래된 것이 앞에 온다."""
    fmt = FIELD.join(["%H", "%ad", "%s", "%b"]) + SEPARATOR
    args = ["git", "log", "--reverse", "--date=short", f"--pretty=format:{fmt}"]
    if since:
        args.append(f"--since={since}")

    by_date: "OrderedDict[str, list[dict]]" = OrderedDict()
    for chunk in _run(args).split(SEPARATOR):
        if not chunk.strip():
            continue
        commit, date, subject, body = (chunk.strip("\n").split(FIELD) + ["", "", "", ""])[:4]
        stats = _run(["git", "show", "--stat", "--format=", commit]).strip().splitlines()
        by_date.setdefault(date, []).append({
            "commit": commit[:7],
            "subject": subject.strip(),
            "lead": _lead(body),
            "files": len([ln for ln in stats if "|" in ln]),
        })
    return by_date


def render(by_date: "OrderedDict[str, list[dict]]") -> str:
    lines = [
        "# 개발 일지",
        "",
        "커밋 이력에서 만들었습니다. 손으로 쓴 것이 아니라 **작업할 때 남긴",
        "기록을 그대로 옮긴 것**이라, 저장소에서 같은 명령으로 다시 만들면",
        "같은 내용이 나옵니다.",
        "",
        "```",
        "python scripts/build_devlog.py",
        "```",
        "",
        f"기간 {next(iter(by_date), '?')} ~ {next(reversed(by_date), '?')} · "
        f"커밋 {sum(len(v) for v in by_date.values())}건",
        "",
        "---",
        "",
    ]
    for date, entries in by_date.items():
        lines.append(f"## {date}")
        lines.append("")
        for entry in entries:
            lines.append(f"**{entry['subject']}**  `{entry['commit']}` · 파일 {entry['files']}개")
            lines.append("")
            if entry["lead"]:
                lines.append(entry["lead"])
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="커밋 이력으로 개발 일지를 만든다")
    parser.add_argument("--since", default="", help="이 날짜부터 (예: 2026-08-21)")
    parser.add_argument(
        "--out", default="submission/4_제작과정/개발일지.md",
        help="저장할 자리. 비우면 표준출력",
    )
    args = parser.parse_args()

    by_date = collect(args.since or None)
    if not by_date:
        print("커밋이 없다. --since 를 확인하라.", file=sys.stderr)
        return 1

    text = render(by_date)
    if not args.out:
        print(text)
        return 0

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"{args.out} 에 썼다. 커밋 {sum(len(v) for v in by_date.values())}건 · "
          f"날짜 {len(by_date)}일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
