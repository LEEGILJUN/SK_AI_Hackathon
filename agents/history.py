"""승인된 건을 이슈 이력에 되쓴다 — 루프를 닫는 자리.

**지금까지 이슈 이력은 읽기만 했다.** `find_similar_issues` 가 24건짜리
고정 파일을 뒤져 중복을 끊는데, 이번에 처리한 건은 어디에도 안 쌓였다.
그러면 같은 문제가 다음 달에 또 올라와도 "처음 보는 건"이 된다.

기획서가 Pain Point 다섯째로 적은 것이 정확히 이것이다.

    재학습 때마다 폴더가 남아 이미지 자체는 보존되지만, 그것이 어떤
    이슈에 대응해 무엇을 왜 추가한 것인지는 어디에도 기록되지 않는다.
    담당자가 바뀌면 모델이 무엇을 학습했고 어떤 판단으로 그렇게 했는지가
    사실상 사라진다.

**문서를 만드는 것과 이력이 쌓이는 것은 다르다.** 승인 요청 문서는
`release/` 폴더에 파일로 남지만, 그것만으로는 다음 이슈가 올라왔을 때
아무도 찾아보지 않는다. 조회되는 자리에 들어가야 자산이 된다.

── 언제 쌓는가 ─────────────────────────────────────────────────────────

**승인된 것만 `resolved=true` 로 쌓는다.** 후보를 만든 것은 해결이
아니다. 승인 전에 해결로 적으면 다음 이슈가 "이미 처리된 건"으로 잘못
끊긴다. 비승인은 쌓되 `resolved=false` 다 — 시도했고 안 됐다는 것도
다음 사람에게는 정보다.

── 무엇을 안 하는가 ────────────────────────────────────────────────────

**이력이 원인을 정하지 않는다.** 과거가 비슷하다고 이번 원인을 그것으로
정하면 진단이 유사도 맞히기가 된다. 원인은 언제나 `decide()` 가 판별
7항목으로 낸다. 이력의 역할은 중복 차단 하나다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

#: 이력 파일. `lookup/factory.py` 의 `find_similar_issues` 가 읽는 그 파일이다.
DEFAULT_HISTORY = Path(__file__).resolve().parent.parent / "data" / "issue_history.jsonl"


def next_issue_id(path: str | Path = DEFAULT_HISTORY) -> str:
    """다음 이슈 번호. 파일에 있는 가장 큰 번호 다음이다.

    **시각으로 만들지 않는다.** 같은 날 두 건이면 번호가 겹치고, 순서도
    안 보인다.
    """
    biggest = 0
    for record in read_history(path):
        raw = str(record.get("issue_id", ""))
        if raw.startswith("ISS-"):
            try:
                biggest = max(biggest, int(raw[4:]))
            except ValueError:
                continue
    return f"ISS-{biggest + 1:04d}"


def read_history(path: str | Path = DEFAULT_HISTORY) -> list[dict[str, Any]]:
    """이력을 읽는다. `_comment` 줄은 건너뛴다. 없으면 빈 목록."""
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "_comment" in record:
            continue
        out.append(record)
    return out


def append_resolved_issue(
    *,
    line: str,
    object_name: str,
    defect_type: str,
    cause: str,
    action: str,
    summary: str,
    resolved: bool,
    bank_version: str | None = None,
    document_no: str | None = None,
    approved_by: str | None = None,
    occurred_at: date | None = None,
    path: str | Path = DEFAULT_HISTORY,
) -> dict[str, Any]:
    """처리된 건을 이력에 한 줄 덧붙인다. **고치거나 지우지 않는다.**

    돌려주는 것은 쌓은 기록이다. 화면이 "이력에 남았습니다"를 보여줄 때
    쓴다. 무엇이 남았는지 보이지 않으면 남은 줄 모른다.
    """
    p = Path(path)
    record: dict[str, Any] = {
        "issue_id": next_issue_id(p),
        "line": line,
        "object_name": object_name,
        "defect_type": defect_type,
        "cause": cause,
        "action": action,
        "resolved": bool(resolved),
        "occurred_at": (occurred_at or date.today()).isoformat(),
        "summary": summary,
    }
    # 되짚을 때 필요한 것만 더 담는다. 승인 문서와 뱅크 판을 알면 그때
    # 무엇을 근거로 그렇게 했는지 찾아갈 수 있다.
    if bank_version:
        record["bank_version"] = bank_version
    if document_no:
        record["document_no"] = document_no
    if approved_by:
        record["approved_by"] = approved_by

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record
