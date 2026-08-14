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
import re
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
 "area_hint": string|null, "observed_from": "YYYY-MM-DD"|null, "lot": string|null,
 "product_id": string|null}

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
    #: 제품명. MES 조회의 열쇠다 — 이슈는 보통 이미지가 아니라 이것으로 온다.
    product_id: str | None = None
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
    # **읽는 것을 빠뜨리고 있었다.** 모델이 뽑아 줘도 버려졌고, 4090 실측에서
    # "제품명을 못 뽑는다"로 보고됐다. `product_id` 는 MES 조회의 열쇠라
    # 이게 비면 접수가 되묻고 거기서 멈춘다.
    report.product_id = data.get("product_id") or None
    report.extracted_by = response.model or adapter.describe()

    raw_date = data.get("observed_from")
    if raw_date:
        try:
            report.observed_from = date.fromisoformat(str(raw_date))
        except ValueError:
            pass

    # 모델이 놓쳤으면 원문에서 직접 줍는다. **지어내지 않는다** — 아래
    # 정규식은 제품 코드 모양(영문 대문자 + 숫자 + 하이픈)에만 맞고, 없으면
    # 그대로 비워 둔다.
    if not report.product_id:
        report.product_id = find_product_id(text)

    return report


#: 제품 코드 모양. `PCB1-LOT-AAJ-img_0087` · `CAP-2026-0714-0031` 같은 것.
#:
#: 대문자로 시작해 하이픈으로 이어지고 숫자를 포함한다. 한국어 문장 안에
#: 섞여 있어도 잡힌다. **모델이 놓칠 때의 보조일 뿐 주 경로가 아니다** —
#: 자연어 추출이 주 입력이라는 설계를 정규식으로 대체하지 않는다.
_PRODUCT_ID = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9_]+){1,4})\b")

#: 이 낱말 뒤에 오는 것을 제품명으로 본다. 없으면 아무거나 줍지 않는다.
_PRODUCT_CUE = re.compile(r"(?:제품|품번|제품명|product)\s*[:：]?\s*$", re.IGNORECASE)

#: 이 낱말이 뒤따르면 제품이 아니라 로트다. `A-217 로트가 …` 같은 문장.
#:
#: `\b` 를 쓰지 않는다 — 한글은 조사가 붙어("로트**가**") 낱말 경계가
#: 성립하지 않고, 그래서 한 번 이 규칙이 통째로 안 먹었다.
_LOT_CUE = re.compile(r"^\s*(?:로트|랏|lot)", re.IGNORECASE)


def find_product_id(text: str) -> str | None:
    """원문에서 제품 코드처럼 생긴 것을 줍는다. 없으면 None.

    **모델이 뽑은 값이 있으면 부르지 않는다.** 자연어 추출이 주 경로이고
    이것은 그물이다.

    문맥을 본다. "제품 X" 처럼 앞에 단서가 있는 것을 먼저 고르고, 뒤에
    "로트"가 붙은 것은 제외한다 — `A-217 로트가 계속 빠집니다` 의 A-217 을
    제품명으로 넣으면 **MES 조회가 조용히 빈손이 된다.** 로트는 로트 칸에
    들어가야 조인이 맞는다.

    숫자가 하나도 없는 것도 제품 코드로 보지 않는다("PCB-기판" 같은 말).
    """
    if not text:
        return None

    cued: list[str] = []
    plain: list[str] = []
    for match in _PRODUCT_ID.finditer(text):
        candidate = match.group(1)
        if not any(ch.isdigit() for ch in candidate):
            continue
        if _LOT_CUE.match(text[match.end():]):
            continue
        (cued if _PRODUCT_CUE.search(text[:match.start()]) else plain).append(candidate)

    return (cued or plain or [None])[0]


def receive(
    text: str,
    adapter: ModelAdapter,
    lookup: LookupLayer | None = None,
    known: dict[str, Any] | None = None,
    attachments: list[str] | None = None,
    duplicate_similarity: float = 0.95,
    duplicate_requires_same_line: bool = True,
) -> IntakeResult:
    """이슈를 접수하고 다음 단계로 넘길지 판단한다.

    known
        웹 양식에서 이미 받은 값. 추출 결과보다 우선한다. 사람이 고른 값이
        모델이 뽑은 값보다 정확하기 때문이다.
    duplicate_similarity
        이 값을 넘고 해결된 사례가 있으면 중복으로 보고 끊는다.
    duplicate_requires_same_line
        **다른 라인의 같은 증상은 중복이 아니다.** 라인마다 뱅크가 따로이므로
        1라인 뱅크가 오염됐다고 2라인 뱅크도 오염됐다는 뜻이 아니다. 유사도만
        보고 끊으면 실제로 존재하는 문제를 "이미 해결된 건"으로 덮는다.
        관련 사례로는 여전히 similar 에 담겨 다음 단계로 넘어간다.
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

    # 이미지가 있어야 진단할 수 있다. 다만 **첨부만이 길은 아니다** — 제품명이나
    # 로트를 알면 MES 조회로 이미지를 찾아낼 수 있고, 현장에서는 그쪽이 더 흔하다
    # ("A-217 로트가 계속 빠집니다"). 둘 다 없을 때만 되묻는다.
    if not report.attachments and not (report.product_id or report.lot):
        return IntakeResult(
            verdict="need_more_info",
            report=report,
            missing=["attachments"],
            question=(
                "해당 이미지를 첨부하시거나 제품명·로트를 알려 주세요. "
                "제품명이 있으면 MES 에서 이미지를 찾을 수 있습니다."
            ),
            note="판별 1·4·5번이 전부 이미지에 걸려 있다.",
        )

    similar: list[PastIssue] = []
    if lookup is not None:
        similar = lookup.find_similar_issues(
            line=report.line or "", object_name=report.object_name or "",
            defect_type=report.defect_type,
        )

    for issue in similar:
        if duplicate_requires_same_line and issue.line != report.line:
            continue
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
