"""인테이크 에이전트 — 자연어 이슈를 구조화한다 (작업 11).

판단하는 것은 두 가지다.

  정보가 충분한가   부족하면 진단으로 넘기지 않고 무엇이 더 필요한지 되묻는다
  이미 해결된 사례인가  같은 건이 다른 라인에서 규명·조치됐으면 여기서 끊는다

두 번째가 특히 중요하다. 중복 이슈를 그대로 흘려보내면 이미 답이 나온 일에
진단과 재학습을 다시 돌리게 된다. 그래프 검색의 역할이 여기 하나다.

언어 모델은 **추출에만** 쓴다. 충분성 판단과 중복 차단은 규칙으로 한다.
모델이 없으면 추출이 비고, 그때는 웹 양식에서 받은 값을 그대로 쓴다.
현장에서 사람이 채우는 항목이므로 그것이 원래 형태이기도 하다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any

from lookup.base import LookupLayer, PastIssue

from .adapters.base import ChatMessage, ModelAdapter

#: 진단으로 넘기기 위해 반드시 있어야 하는 항목.
REQUIRED_FIELDS = ("line", "object_name")

_EXTRACT_PROMPT = """You are structuring a defect report from a manufacturing line.
Extract only what the text actually states. Do not invent values.
Use null for anything not mentioned.

Respond with a single JSON object, no code fences:
{"line": string|null, "object_name": string|null, "defect_type": string|null,
 "area_hint": string|null, "observed_from": "YYYY-MM-DD"|null, "lot": string|null}

Report text:
"""


@dataclass
class IssueReport:
    """구조화된 이슈. 진단 에이전트가 받는 입력이다."""

    raw_text: str
    line: str | None = None
    object_name: str | None = None
    defect_type: str | None = None
    lot: str | None = None
    observed_from: date | None = None
    attachments: list[str] = field(default_factory=list)
    extracted_by: str = ""  # 무엇이 추출했는가. 빈 값이면 사람이 채운 것

    def missing_fields(self) -> list[str]:
        return [f for f in REQUIRED_FIELDS if not getattr(self, f)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "observed_from": self.observed_from.isoformat() if self.observed_from else None
        }


@dataclass
class IntakeResult:
    """인테이크 판정.

    verdict
        proceed         진단으로 넘긴다
        need_more_info  정보가 부족하다. missing 과 question 을 사람에게 돌려준다
        duplicate       이미 해결된 사례다. 진단하지 않는다
    """

    verdict: str
    report: IssueReport
    missing: list[str] = field(default_factory=list)
    question: str = ""
    duplicate_of: str | None = None
    similar: list[PastIssue] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "report": self.report.to_dict(),
            "missing": self.missing,
            "question": self.question,
            "duplicate_of": self.duplicate_of,
            "similar": [s.to_dict() for s in self.similar],
            "note": self.note,
        }


FIELD_LABEL_KO = {
    "line": "라인",
    "object_name": "대상 품목",
    "defect_type": "결함 유형",
}


def extract(text: str, adapter: ModelAdapter) -> IssueReport:
    """자연어에서 항목을 뽑는다.

    모델이 없거나 응답이 깨지면 빈 보고서를 돌려준다. 추측해서 채우면
    엉뚱한 라인의 뱅크를 건드리게 되므로, 비워 두고 되묻는 편이 안전하다.
    """
    report = IssueReport(raw_text=text)

    try:
        response = adapter.chat([ChatMessage.user(_EXTRACT_PROMPT + text)], json_object=True)
    except Exception:
        return report

    if response.is_stub:
        return report

    data = response.json()
    report.line = data.get("line") or None
    report.object_name = data.get("object_name") or None
    report.defect_type = data.get("defect_type") or None
    report.lot = data.get("lot") or None
    report.extracted_by = response.model or adapter.describe()

    raw_date = data.get("observed_from")
    if raw_date:
        try:
            report.observed_from = date.fromisoformat(str(raw_date))
        except ValueError:
            pass

    return report


def receive(
    text: str,
    adapter: ModelAdapter,
    lookup: LookupLayer | None = None,
    known: dict[str, Any] | None = None,
    attachments: list[str] | None = None,
    duplicate_similarity: float = 0.85,
) -> IntakeResult:
    """이슈를 접수하고 다음 단계로 넘길지 판단한다.

    known
        웹 양식에서 이미 받은 값. 추출 결과보다 우선한다. 사람이 고른 값이
        모델이 뽑은 값보다 정확하기 때문이다.
    """
    report = extract(text, adapter)

    for key, value in (known or {}).items():
        if value and hasattr(report, key):
            setattr(report, key, value)
    report.attachments = list(attachments or [])

    missing = report.missing_fields()
    if missing:
        labels = ", ".join(FIELD_LABEL_KO.get(f, f) for f in missing)
        return IntakeResult(
            verdict="need_more_info",
            report=report,
            missing=missing,
            question=f"{labels} 정보가 필요합니다. 알려주시면 진단을 시작하겠습니다.",
            note="추측으로 채우지 않는다. 잘못된 라인의 뱅크를 건드릴 수 있다.",
        )

    if not report.attachments:
        return IntakeResult(
            verdict="need_more_info",
            report=report,
            missing=["attachments"],
            question="해당 이미지를 첨부해 주세요. 이미지 없이는 원인을 규명할 수 없습니다.",
            note="판별 1·4·5번이 전부 이미지에 걸려 있다.",
        )

    similar: list[PastIssue] = []
    if lookup is not None:
        similar = lookup.find_similar_issues(
            line=report.line or "", object_name=report.object_name or "",
            defect_type=report.defect_type,
        )

    for issue in similar:
        if issue.resolved and issue.similarity >= duplicate_similarity:
            return IntakeResult(
                verdict="duplicate",
                report=report,
                duplicate_of=issue.issue_id,
                similar=similar,
                note=(
                    f"{issue.line} 에서 동일 증상이 이미 규명·조치됐다 "
                    f"(유사도 {issue.similarity:.2f}). 중복 작업을 막기 위해 진단하지 않는다."
                ),
            )

    return IntakeResult(
        verdict="proceed",
        report=report,
        similar=similar,
        note="정보가 충분하고 중복 이력이 없다. 진단으로 넘긴다.",
    )
