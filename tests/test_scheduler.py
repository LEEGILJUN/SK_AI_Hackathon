"""예약 실행 스케줄러 — 작업 21.

**과제 이름이 "자율 운영"인데 사람이 버튼을 눌러야만 돌면 자율이 아니다.**
야간에 쌓인 생산분을 아침에 보고받는 그림이다.

여기서 지키는 것 셋.

  1. **배포하지 않는다.** 승인 요청 문서에서 멈춘다
  2. 사람이 접수한 것과 **같은 경로**로 돈다 — 전용 진단을 따로 만들지 않는다
  3. 조용히 자르지 않는다 — 상한에 걸린 건수를 보고서에 남긴다
"""

from __future__ import annotations

import ast
from datetime import datetime, time
from pathlib import Path

from scheduler import due, next_run, run_nightly, scan_pending
from scheduler.nightly import (
    MAX_ISSUES_PER_RUN,
    NightlyReport,
    PendingIssue,
    build_issue_text,
    is_missed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 시각 판정 ───────────────────────────────────────────────────────────


def test_it_does_not_run_before_the_appointed_time():
    at = time(7, 0)
    assert due(datetime(2026, 8, 15, 6, 59), at) is False
    assert due(datetime(2026, 8, 15, 7, 0), at) is True


def test_it_does_not_run_twice_in_a_day():
    """같은 날 이미 돌았으면 안 돈다. 두 번 돌면 같은 이슈가 두 번 접수된다."""
    at = time(7, 0)
    ran = datetime(2026, 8, 15, 7, 0)
    assert due(datetime(2026, 8, 15, 9, 0), at, last_run=ran) is False
    assert due(datetime(2026, 8, 16, 7, 0), at, last_run=ran) is True


def test_a_late_wake_up_still_handles_that_day():
    """7시에 돌기로 했는데 9시에 깨어나도 그날 몫을 처리한다.

    놓친 날을 건너뛰면 그날 쌓인 것이 영영 처리되지 않는다.
    """
    assert due(datetime(2026, 8, 15, 9, 30), time(7, 0)) is True


def test_the_next_run_is_shown():
    now = datetime(2026, 8, 15, 9, 0)
    assert next_run(now, time(7, 0)) == datetime(2026, 8, 16, 7, 0)
    assert next_run(datetime(2026, 8, 15, 6, 0), time(7, 0)) == datetime(2026, 8, 15, 7, 0)


# ── 배포하지 않는다 ─────────────────────────────────────────────────────


def test_the_scheduler_has_no_deploy_function():
    """배포 함수가 생기지 않았는지 검사한다.

    **품질 검사 설비의 특성상 의도적으로 배제한 경계다.** 자동으로 가는
    것은 진단·큐레이션·재구성 후보·게이트·섀도까지이고, 실제 장비 반영은
    사람이 한다. 이 경계 자체가 제안의 설득 근거다.

    `agents/release.py` 에 같은 성격의 시험이 이미 있다. 스케줄러는 자동
    실행이라 여기가 더 위험한 자리다.
    """
    banned = {"deploy", "apply", "install", "push_to_line", "activate"}
    for path in sorted((REPO_ROOT / "scheduler").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        clash = {n for n in names if any(n == b or n.startswith(b + "_") for b in banned)}
        assert not clash, f"{path.name} 에 배포로 읽히는 함수가 있다: {clash}"


def test_the_report_says_a_human_deploys():
    """보고서가 "배포는 사람이 한다"를 그대로 적는다.

    자동으로 여기까지 왔다는 것을 보여주면서 경계를 함께 말해야 한다.
    """
    outcome = type("O", (), {
        "diagnosis": type("D", (), {"cause": "bank_contamination",
                                    "requires_bank_rebuild": True})(),
        "approval_markdown": "# 승인 요청",
    })()
    report = NightlyReport(ran_at=datetime(2026, 8, 15, 7, 0), scanned=100, missed=1)
    report.issues.append(
        PendingIssue("PCB1-x", "line_01", "pcb1", "LOT-PEND-01-1", 1.2, "…", outcome)
    )
    text = report.describe()
    assert "배포는 사람이 합니다" in text
    assert report.approvals == 1


# ── 조용히 자르지 않는다 ────────────────────────────────────────────────


def test_a_deferred_backlog_is_reported():
    """상한에 걸려 안 한 것을 보고서에 남긴다.

    조용히 자르면 "다 처리했다"로 읽힌다.
    """
    report = NightlyReport(
        ran_at=datetime(2026, 8, 15, 7, 0), scanned=100, missed=12, deferred=7
    )
    report.issues = [PendingIssue("x", "line_01", "pcb1", None, 1.0, "…") for _ in range(5)]
    assert "7건은 이번에 처리하지 않았습니다" in report.describe()


def test_nothing_to_do_is_a_normal_outcome():
    """미검이 없으면 이슈를 만들지 않는다.

    "밤새 아무 일도 없었다"가 정상이다. 억지로 이슈를 만들면 사람이 확인할
    것만 는다.
    """
    quiet = NightlyReport(ran_at=datetime(2026, 8, 15, 7, 0), scanned=100, missed=0)
    assert quiet.issues == []
    assert "미검이 없습니다" in quiet.describe()

    empty = NightlyReport(ran_at=datetime(2026, 8, 15, 7, 0))
    assert "쌓이지 않았습니다" in empty.describe()


# ── 사람이 접수한 것과 같은 경로 ────────────────────────────────────────


def test_the_generated_issue_is_natural_language_with_the_product_id():
    """스케줄러가 만든 이슈도 자연어이고 제품명이 본문에 있다.

    인테이크가 원문에서 항목을 뽑는 것이 설계다. 스케줄러만 구조화된 입력을
    넣으면 그 경로가 시험되지 않고, 시연에서 "모델이 뽑았다"고 말할 수 없다.
    """
    from lookup.base import ImageRecord

    record = ImageRecord(
        product_id="PCB1-LOT-PEND-01-1-img_0042", path="p.png",
        line="line_01", object_name="pcb1", lot="LOT-PEND-01-1",
    )
    text = build_issue_text(record, score=1.35, threshold=2.20)

    assert "PCB1-LOT-PEND-01-1-img_0042" in text
    assert "1라인" in text
    assert "{" not in text, "구조화된 값이 아니라 자연어여야 한다"


def test_only_pending_images_are_picked_up():
    """이미 판정된 구간은 건드리지 않는다.

    조회 계층이 구간을 모르면 빈 목록이다 — **모르는 것을 "전부 처리
    대상"으로 삼으면 이미 판정된 이미지를 다시 돌린다.**
    """
    from lookup.base import ImageRecord

    class Stub:
        def find_images(self, **_kwargs):
            return [
                ImageRecord("a", "a.png", "line_01", "pcb1", split="operation"),
                ImageRecord("b", "b.png", "line_01", "pcb1", split="pending"),
                ImageRecord("c", "c.png", "line_01", "pcb1", split="holdout"),
                ImageRecord("d", "d.png", "line_01", "pcb1", split=None),
            ]

    picked = scan_pending(Stub(), "line_01", "pcb1")
    assert [r.product_id for r in picked] == ["b"]


def test_the_cap_is_small_enough_to_be_reviewable():
    """한 번에 처리할 건수 상한이 사람이 볼 수 있는 크기다.

    섀도의 논거가 "불일치 건만 보면 되니 공수가 준다"인데, 수백 건을
    들이밀면 그 논거가 무너진다.
    """
    assert 1 <= MAX_ISSUES_PER_RUN <= 10


# ── 실제 경로로 끝까지 돈다 ─────────────────────────────────────────────
#
# 아래 넷은 **손으로 만든 스텁이 아니라 저장소 기본 구성**(DemoFactory +
# MockLookup)으로 돈다. 스텁만 보면 "찾은 미검이 파이프라인에 전달되지 않아
# 전건 판정 보류"인 상태를 통과시킨다 — 실제로 그런 적이 있다.


def _run(factory, lookup):
    from agents.adapters import build_adapters

    return run_nightly(factory, lookup, build_adapters(), now=datetime(2026, 8, 15, 7, 0))


def test_it_actually_reaches_diagnosis_end_to_end(demo_factory, demo_lookup):
    """미검을 찾고 그것이 파이프라인까지 전달되는가.

    **여기가 무너지면 스케줄러는 이슈만 만들고 아무것도 진단하지 못한다.**
    파이프라인이 자기 조회 계층을 새로 만들면 여기서 찾은 미검을 저기서 못
    찾아 3단계에서 멈춘다.
    """
    report = _run(demo_factory, demo_lookup)
    assert report.scanned > 0, "pending 구간이 비어 있으면 스케줄러가 할 일이 없다"
    assert report.missed > 0 and report.issues

    outcome = report.issues[0].outcome
    assert outcome.missed_records, "파이프라인이 같은 미검을 못 찾았다 — 조회 계층이 둘이다"
    assert [s.status for s in outcome.stages if s.key == "diagnose"] != ["skipped"]


def test_the_pipeline_shares_the_schedulers_lookup(demo_factory):
    """한 실행 안에 조회 계층은 하나다.

    둘이면 서로 다른 이미지·임계값을 보고, 조회 기록도 갈라져 화면의 방식별
    집계가 실제와 어긋난다. **주입한 것을 무시하고 자기 목을 만들면 스케줄러가
    찾은 미검을 파이프라인이 못 찾는다** — 실제로 그런 적이 있다.
    """
    from agents.adapters import build_adapters
    from lookup import MockLookup

    factory = demo_factory

    class Spy(MockLookup):
        seen: list = []
        def find_images(self, **kwargs):
            type(self).seen.append(kwargs)
            return super().find_images(**kwargs)

    lookup = Spy(catalog=factory.catalog, banks=factory.bank_versions(),
                 quality_provider=factory.quality_baseline)
    report = run_nightly(factory, lookup, build_adapters(),
                         now=datetime(2026, 8, 15, 7, 0))

    # 스케줄러가 부른 것과 파이프라인이 부른 것이 **같은 객체**에 쌓인다.
    assert len(Spy.seen) > len(report.issues), (
        "파이프라인이 주입한 조회 계층을 쓰지 않고 자기 목을 만들었다"
    )
    assert report.issues[0].outcome.retrievals, "조회 기록이 주입한 계층에서 나와야 한다"



def test_it_does_not_pre_decide_check_five(demo_factory, demo_lookup):
    """판별 5번을 스케줄러가 미리 정하지 않는다.

    **최근접 패치가 결함인가 진짜 정상품인가가 뱅크 오염과 정상 분포 중첩을
    가르는 유일한 자리다.** `run_pipeline` 의 기본값이 "defect" 이므로 그냥
    부르면 자동 접수 건이 전부 오염으로 기운다. 모델이 없으면 보류가 옳다.
    """
    report = _run(demo_factory, demo_lookup)
    assert all(i.outcome.patch_override is None for i in report.issues)


def test_the_threshold_comes_from_the_lookup_layer():
    """임계값을 코드에 박지 않는다 — 판별 3번은 조회 계층이 답한다.

    품목·뱅크판마다 다르고, 파이프라인은 이미 `get_threshold` 를 쓴다.
    스케줄러가 자기 기본값을 들고 있으면 같은 실행 안에서 두 값이 돈다.
    """
    import inspect as _inspect

    from scheduler import nightly

    assert "threshold" not in _inspect.signature(nightly.run_nightly).parameters
    source = Path(nightly.__file__).read_text(encoding="utf-8")
    assert "get_threshold" in source
    assert "2.20" not in source, "임계값이 코드에 박혀 있다"


def test_a_missing_threshold_is_reported_not_guessed():
    """임계값이 없으면 판정하지 않고 보고서에 적는다.

    없는 기준을 지어내 미검을 가리면 그 판정은 우리가 만든 것이다.
    """
    report = NightlyReport(ran_at=datetime(2026, 8, 15, 7, 0))
    report.skipped.append("line_01/pcb1 — 운영 임계값이 없어 판정하지 않았습니다")
    assert "건너뜀" in report.describe()


def test_a_withheld_verdict_is_not_written_as_a_decision():
    """판정 보류를 "재구성 아님"으로 적지 않는다.

    결론이 난 것처럼 읽혀서 사람이 그냥 넘긴다.
    """
    report = NightlyReport(ran_at=datetime(2026, 8, 15, 7, 0), scanned=10, missed=1)
    report.issues.append(PendingIssue("x", "line_01", "pcb1", None, 1.0, "…"))
    text = report.describe()
    assert "판정 보류" in text and "재구성 아님" not in text


def test_an_unverified_image_is_never_called_a_miss():
    """사람이 아직 확인하지 않은 건은 미검이라 부르지 않는다.

    **점수가 낮다는 것만으로 미검이라 하면 진짜 양품이 이슈가 된다.**
    미검의 정의는 `ImageRecord.is_missed` 하나뿐이다.
    """
    from lookup.base import ImageRecord

    unseen = ImageRecord("a", "a.png", "line_01", "pcb1", ground_truth=None)
    assert is_missed(unseen, score=0.01, threshold=2.20) is False

    recorded = ImageRecord("b", "b.png", "line_01", "pcb1",
                           ground_truth="defect", verdict="pass")
    assert is_missed(recorded, score=99.0, threshold=2.20) is True   # 기록된 판정을 믿는다
    assert recorded.is_missed is True

    fresh = ImageRecord("c", "c.png", "line_01", "pcb1", ground_truth="defect")
    assert is_missed(fresh, score=0.01, threshold=2.20) is True      # 판정 기록이 없으면 점수로
    assert is_missed(fresh, score=9.99, threshold=2.20) is False
