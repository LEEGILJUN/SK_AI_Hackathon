"""제출 폴더를 저장소에서 조립한다.

    .venv/bin/python scripts/build_submission.py
    .venv/bin/python scripts/build_submission.py --check    # 빠진 것만 본다

── 왜 복사본을 커밋하지 않는가 ─────────────────────────────────────────

제출 폴더에는 기획서·설명서·실측 데이터가 들어가야 하는데, 그것들은 이미
`docs/` 에 있다. 복사해서 커밋하면 **두 벌이 되고 한쪽만 고쳐진다.** 이
저장소가 계속 경계해 온 문제다.

그래서 `submission/` 에는 README 만 두고, 붙는 파일은 이 스크립트가 그때
그때 가져온다. 조립 결과는 `.gitignore` 가 잡는다.

**README 는 사람이 쓴다.** 이 스크립트가 만들지 않는다. 심사위원이 읽는
글이라 기계가 지어낼 자리가 아니다.

── 폴더 이름을 바꾸지 않는다 ───────────────────────────────────────────

제출 안내에 이렇게 적혀 있다.

    이름을 바꾸거나 지우거나 새로 만들지 마세요. 평가는 이 구조를 그대로
    읽어가는 방식이라, 이름이 다르면 그 항목은 비어 있는 것으로 처리됩니다.

그래서 여기 적힌 폴더 이름을 손대면 안 된다.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = REPO_ROOT / "submission"

#: 심사가 폴더마다 확인하는 질문. 화면에 함께 찍어 무엇을 담는 자리인지 알린다.
QUESTIONS = {
    "1_기획서": "어떤 문제를 풀려고 했나",
    "2_결과물": "기획서에서 정의한 핵심 기능이 실제로 작동하는가",
    "3_검증결과": "작동한다는 걸 어떻게 확인했나",
    "4_제작과정": "계획대로 진행했고, 막힌 지점을 어떻게 풀었나",
}

#: (제출 폴더, 저장소 원본, 제출본 이름). 이름을 바꿔 넣는 것은 심사위원이
#: 파일 목록만 보고도 무엇인지 알게 하기 위해서다.
COPIES: list[tuple[str, str, str]] = [
    ("1_기획서", "docs/기획서.md", "기획서.md"),
    ("1_기획서", "docs/전체아키텍처.png", "전체아키텍처.png"),
    ("1_기획서", "docs/서비스개념도.png", "서비스개념도.png"),
    ("3_검증결과", "docs/설명서.md", "설명서.md"),
    ("3_검증결과", "docs/data_품목별임계값.json", "데이터_품목별임계값.json"),
    ("3_검증결과", "docs/data_점수분포.json", "데이터_점수분포.json"),
    ("3_검증결과", "docs/분포_12품목.png", "분포_12품목.png"),
    ("3_검증결과", "docs/분리도_설명.png", "분리도_설명.png"),
    ("3_검증결과", "docs/실험_pcbAUROC.md", "실험_pcb카테고리AUROC.md"),
    ("3_검증결과", "docs/실험_임계값.md", "실험_임계값.md"),
    ("3_검증결과", "docs/실험_VLM판독.md", "실험_판별1번_VLM판독.md"),
    ("3_검증결과", "docs/실험_판별5번.md", "실험_판별5번.md"),
    ("3_검증결과", "docs/실험_역추적크롭.md", "실험_역추적크롭.md"),
    ("3_검증결과", "docs/전처리와_해상도.md", "전처리와_해상도.md"),
    ("4_제작과정", "docs/운영_표준절차.md", "운영_표준절차.md"),
]


def build_devlog() -> tuple[bool, str]:
    """개발 일지를 커밋 이력에서 다시 만든다."""
    venv = REPO_ROOT / ".venv" / "bin" / "python"
    python = str(venv) if venv.exists() else sys.executable
    result = subprocess.run(
        [python, "scripts/build_devlog.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    return result.returncode == 0, (result.stdout or result.stderr).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="제출 폴더를 조립한다")
    parser.add_argument("--check", action="store_true", help="복사하지 않고 빠진 것만 본다")
    args = parser.parse_args()

    print("제출 폴더 조립\n")

    missing_source: list[str] = []
    copied = 0
    for folder, source, name in COPIES:
        origin = REPO_ROOT / source
        target = SUBMISSION / folder / name
        if not origin.exists():
            missing_source.append(source)
            continue
        if args.check:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        copied += 1

    if not args.check:
        ok, note = build_devlog()
        print(f"  개발 일지   {note}" if ok else f"  개발 일지   실패: {note}")
        print(f"  붙임 파일   {copied}개 복사")
        print()

    # ── 폴더마다 무엇이 있는가 ──────────────────────────────────────
    problems: list[str] = []
    for folder, question in QUESTIONS.items():
        path = SUBMISSION / folder
        readme = path / "README.md"
        files = sorted(p.name for p in path.glob("*") if p.is_file()) if path.is_dir() else []
        print(f"  {folder}  ({question})")
        if not readme.exists():
            problems.append(f"{folder}/README.md 가 없습니다. 심사가 이것부터 읽습니다.")
            print("      README 없음")
        for name in files:
            print(f"      {name}")
        print()

    if missing_source:
        print("저장소에 없는 원본")
        for source in missing_source:
            print(f"  {source}")
        print()

    # ── 아직 안 채워진 자리 ─────────────────────────────────────────
    #
    # **비어 있다는 것을 여기서 말해야 한다.** 마감 직전에 폴더를 열어 보고
    # 알면 늦다. 앱 URL 은 규정상 없으면 구현 완성도를 확인할 수 없다.
    blanks: list[str] = []
    result_readme = SUBMISSION / "2_결과물" / "README.md"
    if result_readme.exists():
        text = result_readme.read_text(encoding="utf-8")
        if "(배포 후 채웁니다)" in text:
            blanks.append(
                "2_결과물/README.md 의 앱 URL과 접속 계정이 비어 있습니다. "
                "규정: 열리지 않으면 구현 완성도를 확인할 수 없습니다."
            )
    root_readme = SUBMISSION / "README.md"
    if root_readme.exists() and "팀번호" in root_readme.read_text(encoding="utf-8"):
        blanks.append("팀번호와 팀명이 아직 안 정해졌습니다. 드라이브 폴더를 받으면 채웁니다.")

    for note in blanks:
        print(f"  아직: {note}")
    for note in problems:
        print(f"  문제: {note}")

    if not blanks and not problems:
        print("  빠진 것이 없습니다.")

    print("\n이 검사는 자리가 채워졌는지만 봅니다. **내용이 맞는지는 사람이 읽어야 합니다.**")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
