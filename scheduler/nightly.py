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
from datetime import datetime
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
    #: 돌지 못하고 건너뛴 품목과 이유. 이것도 조용히 넘기지 않는다.
    skipped: list[str] = field(default_factory=list)
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
            head = (
                f"{self.ran_at:%Y-%m-%d %H:%M} — 처리할 생산분이 없습니다. "
                f"아직 검사하지 않은 이미지가 쌓이지 않았습니다."
            )
            return "\n".join([head] + [f"  건너뜀 — {n}" for n in self.skipped])
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
            if issue.cause is None:
                # **판정이 안 난 것을 "재구성 아님"으로 적지 않는다.** 결론이
                # 난 것처럼 읽혀서 사람이 넘겨 버린다.
                lines.append(f"  · {issue.product_id} — 판정 보류 (사람 확인 필요)")
            else:
                mark = "재구성 필요" if issue.needs_rebuild else "재구성 아님"
                lines.append(f"  · {issue.product_id} — {issue.cause} ({mark})")
        held = sum(1 for i in self.issues if i.cause is None)
        if held:
            lines.append(
                f"  {held}건은 판별 5번(최근접 패치가 결함인가)에 답이 없어 "
                f"판정을 보류했습니다. **추측으로 원인을 정하지 않습니다.**"
            )
        for note in self.skipped:
            lines.append(f"  건너뜀 — {note}")
        if self.approvals:
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
            "skipped": list(self.skipped),
            "issues": [i.to_dict() for i in self.issues],
            "note": self.describe(),
        }


def is_missed(record: ImageRecord, score: float, threshold: float) -> bool:
    """이 건이 미검인가.

    **미검의 정의는 `ImageRecord.is_missed` 하나뿐이다** — 사람이 확인한
    결과는 불량인데 설비는 양품으로 흘려보낸 것. 여기서 다시 정의하면 두 벌이
    되고 한쪽만 고쳐진다.

    **pending 은 아직 검사하지 않은 구간이므로 기록된 판정을 믿지 않는다.**
    거기에 `verdict` 가 들어 있으면 그것은 지난 판정이거나 데이터를 만들 때
    채워 넣은 값이고, 우리가 답할 것은 "지금 뱅크가 이것을 놓치는가"다. 지난
    판정을 믿으면 이미 잡히는 건까지 미검으로 올려 사람이 헛일을 한다.

    다른 구간이면 기록된 설비 판정을 그대로 믿는다. 이미 판정이 끝난 것을
    다시 추론해 뒤집는 것은 우리 일이 아니다.

    `ground_truth` 가 없으면(사람이 아직 안 본 건) 미검이라 말할 수 없다.
    **점수가 낮다는 것만으로 미검이라 부르면 진짜 양품을 이슈로 만든다.**
    """
    if record.ground_truth is None:
        return False
    if record.split != PENDING_SPLIT and record.verdict is not None:
        return record.is_missed
    return record.ground_truth == "defect" and score < threshold


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
    max_issues: int = MAX_ISSUES_PER_RUN,
) -> NightlyReport:
    """쌓인 생산분을 처리하고 보고서를 낸다.

    **배포하지 않는다.** 파이프라인이 승인 요청 문서까지 만들고 멈춘다.

    임계값을 인자로 받지 않는다. **판별 3번은 조회 계층이 답하는 값**이고
    (`get_threshold`), 여기서 기본값을 들고 있으면 파이프라인이 쓰는 값과
    달라진다. 품목마다 다른 것도 그 이유다.
    """
    from app.pipeline import DEMO_ITEMS, run_pipeline
    from inspection import score_images

    ran_at = now or datetime.now()
    report = NightlyReport(ran_at=ran_at)

    candidates: list[tuple[ImageRecord, float, float]] = []
    for line, object_name, _category in DEMO_ITEMS:
        item = factory.item_for(line, object_name)
        if item is None:
            continue
        records = scan_pending(lookup, line, object_name)
        if not records:
            continue
        report.scanned += len(records)

        # 판별 3번 — 임계값은 품목·뱅크판마다 다르고 조회 계층이 답한다.
        record = lookup.get_threshold(line, object_name, item.bank.version)
        if record is None:
            # 값을 못 찾으면 예외가 아니라 None 이 오는 것이 조회 계층의 규약이다.
            # 임계값 없이 미검을 가리면 그 기준을 우리가 지어내는 것이 된다.
            report.skipped.append(
                f"{line}/{object_name} — 운영 임계값이 없어 판정하지 않았습니다"
            )
            continue
        threshold = record.value

        paths = [factory.resolve(r.path) for r in records]
        results = score_images(paths, item.bank, factory.embedder, root=factory.root)
        by_path = {r.image: r for r in results}

        for record in records:
            inferred = by_path.get(record.path)
            if inferred is None:
                continue
            if is_missed(record, inferred.score, threshold):
                candidates.append((record, inferred.score, threshold))

    report.missed = len(candidates)
    if not candidates:
        return report

    # 점수가 낮을수록 "확실히 놓친 것"이라 먼저 본다.
    candidates.sort(key=lambda found: found[1])
    handled, deferred = candidates[:max_issues], candidates[max_issues:]
    report.deferred = len(deferred)

    for record, score, threshold in handled:
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
            # **판별 5번을 미리 정하지 않는다.** 최근접 패치가 결함인지 진짜
            # 정상품인지가 뱅크 오염과 정상 분포 중첩을 가르는 유일한 자리다.
            # 여기에 "defect" 를 넣으면 스케줄러가 만든 이슈는 전부 오염으로
            # 기운다. 모델이 없으면 판정은 보류되고, **그것이 옳은 동작이다.**
            patch_override=None,
            adapters=adapters,
            threshold=threshold,
            # 같은 조회 계층을 넘긴다. 안 넘기면 파이프라인이 자기 목을 새로
            # 만들어, 여기서 찾은 미검을 저기서 못 찾는다.
            lookup=lookup,
            context={
                "line": record.line,
                "object_name": record.object_name,
                "product_id": record.product_id,
                "lot": record.lot or "",
            },
        )
        report.issues.append(issue)

    return report
