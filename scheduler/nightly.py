"""야간에 쌓인 것을 아침에 처리한다.

── 무엇을 하는가 ───────────────────────────────────────────────────────

    1. pending 구간에서 아직 검사하지 않은 이미지를 집는다
    2. 그 품목의 뱅크로 추론해 **미검**을 가려낸다
    3. 미검 건마다 이슈 원문을 만들어 기존 파이프라인에 넣는다
    4. 진단 결과와 승인 요청을 모아 보고서를 낸다

**사람이 접수한 것과 같은 경로로 돈다.** 스케줄러 전용 진단 경로를 따로
만들면 두 벌이 되고, 한쪽만 고쳐져 시연과 실제가 갈린다.

── 배포하지 않는다 ─────────────────────────────────────────────────────

**승인 요청 문서에서 멈춘다.** 자동으로 가는 것은 진단 · 큐레이션 계획 ·
재구성 **후보** 생성 · 게이트 · 섀도까지다. 실제 장비 반영은 사람이 한다.

품질 검사 설비의 특성상 의도적으로 배제한 경계이고, **이 경계 자체가 제안의
설득 근거다.** `tests/test_scheduler.py` 가 배포 함수가 생기지 않았는지
검사한다.

── 무엇을 안 하는가 ────────────────────────────────────────────────────

미검이 없으면 이슈를 만들지 않는다. **"밤새 아무 일도 없었다"가 정상**이고,
그때 억지로 이슈를 만들면 사람이 확인할 것만 늘어난다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from lookup.base import ImageRecord

#: 이 구간의 이미지를 처리한다. manifest 의 `split` 값이다.
PENDING_SPLIT = "pending"

#: 한 번에 처리할 미검 건수 상한.
#:
#: 밤새 쌓인 것이 많아도 **사람이 확인할 수 있는 양**을 넘기면 안 된다.
#: 섀도의 논거가 "불일치 건만 보면 되니 공수가 준다"인데, 수백 건을 들이밀면
#: 그 논거가 무너진다. 넘치면 보고서에 몇 건을 남겼는지 적는다.
MAX_ISSUES_PER_RUN = 5


@dataclass
class PendingIssue:
    """스케줄러가 만든 이슈 하나와 그 처리 결과."""

    product_id: str
    line: str
    object_name: str
    lot: str | None
    score: float
    issue_text: str
    #: 파이프라인 실행 결과(RunOutcome). 진단·승인 요청이 여기 들어 있다.
    outcome: Any = None

    @property
    def cause(self) -> str | None:
        return getattr(getattr(self.outcome, "diagnosis", None), "cause", None)

    @property
    def needs_rebuild(self) -> bool:
        diagnosis = getattr(self.outcome, "diagnosis", None)
        return bool(getattr(diagnosis, "requires_bank_rebuild", False))

    @property
    def has_approval_request(self) -> bool:
        return bool(getattr(self.outcome, "approval_markdown", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "line": self.line,
            "object_name": self.object_name,
            "lot": self.lot,
            "score": self.score,
            "cause": self.cause,
            "needs_rebuild": self.needs_rebuild,
            "approval_request": self.has_approval_request,
        }


@dataclass
class NightlyReport:
    """한 번 돈 결과. 아침에 사람이 읽는 것이다."""

    ran_at: datetime
    scanned: int = 0
    missed: int = 0
    issues: list[PendingIssue] = field(default_factory=list)
    #: 상한에 걸려 이번에 처리하지 않은 건수. **조용히 자르지 않는다.**
    deferred: int = 0
    note: str = ""

    @property
    def rebuild_requested(self) -> int:
        return sum(1 for issue in self.issues if issue.needs_rebuild)

    @property
    def approvals(self) -> int:
        return sum(1 for issue in self.issues if issue.has_approval_request)

    def describe(self) -> str:
        """사람이 읽는 한 문단."""
        if not self.scanned:
            return (
                f"{self.ran_at:%Y-%m-%d %H:%M} — 처리할 생산분이 없습니다. "
                f"아직 검사하지 않은 이미지가 쌓이지 않았습니다."
            )
        if not self.missed:
            return (
                f"{self.ran_at:%Y-%m-%d %H:%M} — {self.scanned}장을 검사했고 "
                f"미검이 없습니다. **조치할 것이 없습니다.**"
            )

        lines = [
            f"{self.ran_at:%Y-%m-%d %H:%M} — {self.scanned}장 중 미검 "
            f"{self.missed}건을 찾아 {len(self.issues)}건을 진단했습니다."
        ]
        if self.deferred:
            lines.append(
                f"  {self.deferred}건은 이번에 처리하지 않았습니다 "
                f"(한 번에 {MAX_ISSUES_PER_RUN}건까지)."
            )
        for issue in self.issues:
            mark = "재구성 필요" if issue.needs_rebuild else "재구성 아님"
            lines.append(
                f"  · {issue.product_id} — {issue.cause or '진단 보류'} ({mark})"
            )
        if self.rebuild_requested:
            lines.append(
                f"  승인 요청 {self.approvals}건을 만들어 두었습니다. "
                f"**배포는 사람이 합니다.**"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran_at": self.ran_at.isoformat(),
            "scanned": self.scanned,
            "missed": self.missed,
            "handled": len(self.issues),
            "deferred": self.deferred,
            "rebuild_requested": self.rebuild_requested,
            "approvals": self.approvals,
            "issues": [i.to_dict() for i in self.issues],
            "note": self.describe(),
        }


def scan_pending(lookup, line: str, object_name: str) -> list[ImageRecord]:
    """아직 검사하지 않은 생산분을 집는다.

    조회 계층이 구간을 모르면(`split` 이 없으면) 빈 목록이다. **모르는 것을
    "전부 처리 대상"으로 삼으면 이미 판정된 이미지를 다시 돌린다.**
    """
    records = lookup.find_images(line=line, object_name=object_name, limit=10_000)
    return [r for r in records if r.split == PENDING_SPLIT]


def build_issue_text(record: ImageRecord, score: float, threshold: float) -> str:
    """스케줄러가 만든 이슈 원문.

    **사람이 쓴 것처럼 자연어로 쓴다.** 인테이크가 원문에서 항목을 뽑는 것이
    설계이고, 스케줄러만 구조화된 입력을 넣으면 그 경로가 시험되지 않는다.
    제품명을 본문에 넣는 것도 같은 이유다.
    """
    line_no = record.line.split("_")[-1].lstrip("0") or record.line
    return (
        f"{line_no}라인 야간 생산분 자동 점검에서 미검 의심 건이 나왔습니다. "
        f"제품 {record.product_id} 이며 이상 점수 {score:.3f} 로 "
        f"임계값 {threshold:.2f} 아래입니다. 확인 바랍니다."
    )


def run_nightly(
    factory,
    lookup,
    adapters,
    *,
    now: datetime | None = None,
    threshold: float = 2.20,
    max_issues: int = MAX_ISSUES_PER_RUN,
) -> NightlyReport:
    """쌓인 생산분을 처리하고 보고서를 낸다.

    **배포하지 않는다.** 파이프라인이 승인 요청 문서까지 만들고 멈춘다.
    """
    from app.pipeline import DEMO_ITEMS, run_pipeline
    from inspection import score_images

    ran_at = now or datetime.now()
    report = NightlyReport(ran_at=ran_at)

    candidates: list[tuple[ImageRecord, float]] = []
    for line, object_name, _category in DEMO_ITEMS:
        item = factory.item_for(line, object_name)
        if item is None:
            continue
        records = scan_pending(lookup, line, object_name)
        if not records:
            continue
        report.scanned += len(records)

        paths = [factory.resolve(r.path) for r in records]
        results = score_images(paths, item.bank, factory.embedder, root=factory.root)
        by_path = {r.image: r for r in results}

        for record in records:
            inferred = by_path.get(record.path)
            if inferred is None:
                continue
            # 미검 — 사람이 확인한 결과는 불량인데 설비 점수가 임계값 아래다.
            if record.ground_truth == "defect" and inferred.score < threshold:
                candidates.append((record, inferred.score))

    report.missed = len(candidates)
    if not candidates:
        return report

    # 점수가 낮을수록 "확실히 놓친 것"이라 먼저 본다.
    candidates.sort(key=lambda pair: pair[1])
    handled, deferred = candidates[:max_issues], candidates[max_issues:]
    report.deferred = len(deferred)

    for record, score in handled:
        issue = PendingIssue(
            product_id=record.product_id,
            line=record.line,
            object_name=record.object_name,
            lot=record.lot,
            score=score,
            issue_text=build_issue_text(record, score, threshold),
        )
        # **사람이 접수한 것과 같은 경로다.** 스케줄러 전용 진단을 따로 만들면
        # 두 벌이 되고 한쪽만 고쳐진다.
        issue.outcome = run_pipeline(
            factory,
            issue.issue_text,
            adapters=adapters,
            threshold=threshold,
            context={
                "line": record.line,
                "object_name": record.object_name,
                "product_id": record.product_id,
                "lot": record.lot or "",
            },
        )
        report.issues.append(issue)

    return report
