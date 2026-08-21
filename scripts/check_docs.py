"""문서가 코드와 어긋나지 않는지 검사한다.

문서는 조용히 낡는다. 코드가 바뀌어도 문서는 그대로 있고, 아무도 눈치채지
못한 채 팀원이 그 문서를 보고 엉뚱한 일을 한다. 실제로 이 저장소에서도
폴더 이름을 바꾼 뒤 문서 세 곳이 낡은 채로 남아 있었고, 테스트 건수가
89 로 적힌 곳이 있었다(실제 146).

사람이 주기적으로 훑는 것보다 기계가 매번 보는 편이 낫다.

실행:
    .venv/bin/python scripts/check_docs.py

검사하는 것
    1. 문서가 가리키는 경로가 실제로 있는가 (아직 만들 파일은 제외)
    2. 문서에 박힌 테스트 건수가 실제와 맞는가
    3. 이름이 바뀐 것을 옛 이름으로 부르고 있지 않은가
    4. 새로 생긴 문서가 문서지도에 등록됐는가
"""

from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 아직 만들어지지 않았지만 문서가 가리켜도 되는 것들.
#: 팀원이 앞으로 만들 파일이므로 "없다"가 오류가 아니다.
#: 만들어지면 여기서 빼도 되고 그냥 둬도 된다.
PLANNED = {
    "data/build_factory.py",
    "data/manifest.csv",
    "data/mes.csv",
    "data/factory_summary.txt",   # build_factory 산출물. 재현되므로 커밋 안 함
    "data/issue_history.jsonl",
    "data/factory/",
    "lookup/factory.py",
    "indexer/scan.py",
    "scripts/compute_quality_baseline.py",   # data/quality_baseline.yaml 이 가리킴
    "docs/화질지표정의.md",      # 깃사용법에서 파일 만들기 예시로 든 이름
}

#: 이름이 바뀐 것. 문서에 옛 이름이 남아 있으면 잡는다.
RENAMED = {
    r"\binspect/": "inspection/ (inspect 는 파이썬 표준 라이브러리 이름이라 못 씀)",
}

#: 문서지도에 등록돼야 하는 문서를 찾을 위치.
#:
#: `submission/` 도 본다. **심사위원이 읽는 문서라 낡은 숫자가 제일 비싸다** —
#: 저장소 문서는 팀원이 읽지만 저것은 채점하는 사람이 읽는다.
DOC_GLOBS = ("*.md", "docs/*.md", "examples/*.md", "submission/**/*.md")

DOC_MAP = REPO_ROOT / "docs" / "문서지도.md"

def nfc(text: str) -> str:
    """한글 문자열을 NFC 로 맞춘다.

    macOS 파일시스템은 한글 파일명을 NFD(자모 분리)로 돌려주는데, 문서에
    적힌 이름은 대개 NFC(완성형)다. 정규화하지 않으면 같은 이름인데도
    문자열 비교가 실패한다. 실제로 이 검사기가 그 버그로 오탐을 냈다.
    """
    return unicodedata.normalize("NFC", text)


PATH_PATTERN = re.compile(
    r"`((?:data|agents|inspection|lookup|app|scripts|tests|indexer|examples|docs)/[\w가-힣./_-]+)`"
)
COUNT_PATTERN = re.compile(r"(\d+)\s*건(?:\s*통과|\s*이\s*통과)?|전체\s*(\d+)건|pytest[^\n]*?(\d+)건")

problems: list[str] = []
warnings: list[str] = []

#: `--fix` — 테스트 건수만 실제 값으로 고쳐 쓴다.
#:
#: 이 숫자를 손으로 맞추는 일을 하루에 여덟 번 했다. 기계가 이미 양쪽 값을
#: 다 아는데 사람이 옮겨 적을 이유가 없다. **건수만** 고친다 — 경로도
#: 측정값도 안 건드린다. 그쪽은 틀렸을 때 사람이 봐야 할 것들이다.
FIX = "--fix" in sys.argv
repaired: list[str] = []


def documents() -> list[Path]:
    found: list[Path] = []
    for pattern in DOC_GLOBS:
        found.extend(sorted(REPO_ROOT.glob(pattern)))
    return found


def actual_test_count() -> int | None:
    """실제 테스트 건수를 센다. pytest 수집만 하고 실행하지 않는다.

    수집만 하는데도 2분이 걸린다. 테스트 모듈이 torch 를 부르고 CUDA 를
    초기화하기 때문이며, 콜드 스타트면 더 걸린다. 180초로 잡았더니 GPU
    장비에서 매번 제한시간에 걸려 "세지 못했습니다" 만 떴다. 넉넉히 준다 —
    이 검사는 자주 돌리는 것이 아니라 오래 걸려도 문제가 없다.
    """
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    python = str(venv_python) if venv_python.exists() else sys.executable
    try:
        result = subprocess.run(
            [python, "-m", "pytest", "tests/", "--collect-only", "-q"],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
        )
    except Exception:
        return None

    # 수집이 깨지면 pytest 는 일부만 세고도 "N tests collected" 를 출력한다.
    # 그 숫자를 그대로 믿으면 멀쩡한 문서를 틀린 값으로 고치게 된다.
    # 실제로 torch 가 안 깔린 환경에서 "146건" 을 "10건" 으로 바꾸라고 했다.
    # 환경이 깨진 것이지 문서가 틀린 것이 아니므로 세지 않은 것으로 본다.
    if result.returncode != 0:
        return None

    match = re.search(r"(\d+)\s+tests? collected", result.stdout)
    return int(match.group(1)) if match else None


def check_paths() -> None:
    for doc in documents():
        text = doc.read_text(encoding="utf-8")
        for match in PATH_PATTERN.finditer(text):
            target = match.group(1)
            if "*" in target or nfc(target) in {nfc(x) for x in PLANNED}:
                continue
            if not (REPO_ROOT / target).exists():
                rel = doc.relative_to(REPO_ROOT)
                problems.append(
                    f"  {rel}: `{target}` 을 가리키는데 그런 파일이 없습니다.\n"
                    f"      경로가 바뀌었거나, 아직 만들 파일이면 "
                    f"scripts/check_docs.py 의 PLANNED 에 추가하세요."
                )


def check_renamed() -> None:
    for doc in documents():
        text = doc.read_text(encoding="utf-8")
        for pattern, guidance in RENAMED.items():
            for match in re.finditer(pattern, text):
                line_no = text[: match.start()].count("\n") + 1
                line = text.split("\n")[line_no - 1]
                # "옛 이름" 자체를 설명하는 줄은 잡지 않는다
                if "옛" in line or "바뀐" in line or "쓸 수 없" in line:
                    continue
                rel = doc.relative_to(REPO_ROOT)
                problems.append(f"  {rel}:{line_no}: 옛 이름이 남아 있습니다 → {guidance}")


def check_test_counts(actual: int | None) -> None:
    if actual is None:
        warnings.append(
            "  테스트 건수를 세지 못했습니다. 수집이 깨졌을 수 있습니다.\n"
            "      `pytest tests/ --collect-only -q` 를 직접 돌려 보세요.\n"
            "      import 오류가 나면 환경 문제입니다. 문서의 숫자를 고치지 마세요."
        )
        return

    # "20건"이 시나리오 개수인지 테스트 개수인지는 문맥으로만 갈린다.
    # 숫자 근처에 테스트 관련 낱말이 있을 때만 본다.
    near = re.compile(r"(?:테스트|pytest|tests/)[^\n]{0,30}?(\d{2,4})\s*건"
                      r"|(\d{2,4})\s*건[^\n]{0,20}?(?:통과|테스트)")
    # 다른 것을 세는 문장은 제외한다. **다만 "테스트 NNN건" 처럼 낱말이
    # 바로 앞에 붙은 형태는 제외하지 않는다** — 같은 줄에 "이슈"나 "작업"이
    # 있다는 이유로 건너뛰어 `CLAUDE.md` 의 건수가 조용히 낡아 있었다.
    other = re.compile(r"시나리오|이미지|장\b|작업|케이스|커밋|이슈|로트")

    for doc in documents():
        # **낸 문서와 실측 기록은 그때의 숫자가 맞다.** 제출본을 지금 값으로
        # 고치면 무엇을 제출했는지 알 수 없게 되고, 실험 기록을 고치면 언제
        # 잰 것인지가 사라진다. 둘 다 낡는 것이 아니라 시점을 적은 것이다.
        # 개발 일지는 커밋 메시지를 옮긴 것이다. 그때 89건이었다는 문장은
        # 지금 480건이라고 해서 틀린 것이 아니다. **그 시점의 기록이다.**
        if (doc.name == "기획서.md" or doc.name.startswith("실험_")
                or doc.name == "개발일지.md"):
            continue
        text = doc.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.split("\n"), 1):
            for match in near.finditer(line):
                number = match.group(1) or match.group(2)
                # group(1) 은 "테스트/pytest/tests/" 가 앞에 붙은 형태다.
                # 그 경우는 무엇을 세는지 의심할 여지가 없다.
                if match.group(1) is None and other.search(line):
                    continue
                value = int(number)
                # **한 줄에 수집 건수와 통과 건수를 같이 적는 것이 맞다.**
                # "435건 가운데 430건을 통과하고 5건을 건너뛰었습니다" 에서
                # 430 은 통과 수이지 전체가 아니다. 같은 줄에 전체 건수가
                # 이미 있으면 나머지 숫자는 그 내역이므로 세지 않는다.
                # 이걸 안 두면 정확한 문장을 비틀어야 검사를 통과한다.
                if value != actual and re.search(rf"{actual}\s*건", line):
                    continue
                if value < 20 or value > 5000:
                    continue
                if value != actual:
                    rel = doc.relative_to(REPO_ROOT)
                    # 측정 기록에는 그때의 건수가 적혀 있다. 자동으로
                    # 현재 값으로 덮으면 **언제 잰 것인지가 사라진다.**
                    if FIX and not doc.name.startswith("실험_"):
                        # 숫자만 바꾼다. 그 줄의 다른 낱말은 건드리지 않는다.
                        fixed = line[:match.start(0)] + match.group(0).replace(
                            number, str(actual), 1
                        ) + line[match.end(0):]
                        lines = text.split("\n")
                        lines[line_no - 1] = fixed
                        text = "\n".join(lines)
                        doc.write_text(text, encoding="utf-8")
                        repaired.append(f"  {rel}:{line_no}: {value} → {actual}")
                        continue
                    problems.append(
                        f"  {rel}:{line_no}: 테스트 {value}건이라고 적혀 있는데 실제는 {actual}건입니다.\n"
                        f"      → {line.strip()[:70]}"
                    )


def check_doc_map() -> None:
    if not DOC_MAP.exists():
        warnings.append(f"  {DOC_MAP.relative_to(REPO_ROOT)} 가 없습니다. 문서 목록을 관리할 곳이 필요합니다.")
        return

    listed = nfc(DOC_MAP.read_text(encoding="utf-8"))
    for doc in documents():
        rel = nfc(doc.relative_to(REPO_ROOT).as_posix())
        if rel == nfc("docs/문서지도.md"):
            continue
        # 제출 폴더의 붙임 파일은 `scripts/build_submission.py` 가 docs/ 에서
        # 복사한 것이다. **원본이 이미 문서지도에 있다.** 복사본까지 등록하면
        # 목록이 두 벌이 되고, 이름을 바꿔 넣은 것이라 원본과 짝도 안 맞는다.
        if rel.startswith("submission/") and doc.name != "README.md":
            continue
        if rel not in listed and nfc(doc.name) not in listed:
            warnings.append(
                f"  {rel} 이 문서지도에 없습니다. 새로 만든 문서라면 등록해 주세요."
            )


def main() -> int:
    print("문서 검사\n")
    docs = documents()
    print(f"  대상 문서 {len(docs)}개")

    actual = actual_test_count()
    if actual is not None:
        print(f"  실제 테스트 {actual}건")
    print()

    check_paths()
    check_renamed()
    check_test_counts(actual)
    check_doc_map()

    if problems:
        print(f"고쳐야 할 것 {len(problems)}건\n")
        for line in problems:
            print(line)
        print()
    if warnings:
        print(f"확인해 보실 것 {len(warnings)}건\n")
        for line in warnings:
            print(line)
        print()
    if repaired:
        print(f"고친 것 {len(repaired)}건 (--fix)\n")
        for line in repaired:
            print(line)
        print()
    if not problems and not warnings:
        print("문서와 코드가 어긋난 곳이 없습니다.")

    print("이 검사는 기계로 확인 가능한 것만 봅니다. 내용이 맞는지는 사람이 판단해야 합니다.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
