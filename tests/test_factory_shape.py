"""실제 규모로 돌려 규모 때문에 갈리는 결함을 Mac 에서 잡는다.

**두 번 겪었다.** Mac 의 합성 공장은 로트가 14장이고 4090 의 실데이터는
100장인데, 그 차이 하나로 결함이 Mac 을 통과하고 4090 에서 터졌다.

    로트가 상한 50 에 잘려 결함이 먼저 사라짐    (e737a04)
    스케줄러와 파이프라인이 같은 미검을 두고 갈림  (070e665)

4090 왕복은 비싸다 — 시험 한 바퀴가 13분이고 실모델 실행은 더 든다. 여기
있는 시험들은 **이미지 없이 레코드만으로** 같은 조건을 만든다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scheduler.nightly import scan_pending
from tests.factory_shape import (
    DEFECT_PER_LOT,
    LINES,
    LOT_SIZE,
    NORMAL_PER_LOT,
    SPLIT_LOTS,
    counts_by_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── 모양이 실제와 같은가 ────────────────────────────────────────────────


def test_the_shape_matches_the_real_generator():
    """`data/build_factory.py` 의 값과 어긋나면 여기서 걸린다.

    생성기가 로트 크기를 바꿨는데 이 모형이 그대로면, 시험은 통과하는데
    4090 에서만 또 터진다. **모형이 낡는 것이 제일 위험하다.**
    """
    source = (REPO_ROOT / "data/build_factory.py").read_text(encoding="utf-8")

    assert f"NORMAL_IMAGES_PER_LOT = {NORMAL_PER_LOT}" in source
    assert f"ANOMALY_IMAGES_PER_LOT = {DEFECT_PER_LOT}" in source
    for line, obj in LINES.items():
        assert f'"{line}": "{obj}"' in source or f"'{line}': '{obj}'" in source


def test_the_catalog_is_the_size_the_document_says(shaped_catalog):
    """배분표(`docs/전처리와_해상도.md`)와 같은가."""
    assert len(shaped_catalog) == 4_000
    assert len({(r.line, r.lot) for r in shaped_catalog}) == 40
    assert counts_by_split(shaped_catalog) == {
        "bank": 1_200, "operation": 1_600, "holdout": 800, "pending": 400,
    }


def test_the_bank_window_has_no_defects(shaped_catalog):
    """뱅크 구간은 초기 수집이라 결함이 없다. 뱅크 오염은 시나리오가 넣는다."""
    bank = [r for r in shaped_catalog if r.split == "bank"]
    assert bank and not [r for r in bank if r.ground_truth == "defect"]


def test_pending_has_no_recorded_verdict(shaped_catalog):
    """pending 은 **아직 검사하지 않은** 구간이다.

    설비 판정이 들어 있으면 스케줄러가 그것을 믿고 미검을 가려서, 지금 뱅크가
    실제로 놓치는지와 무관한 답을 낸다.
    """
    pending = [r for r in shaped_catalog if r.split == "pending"]
    assert pending and all(r.verdict is None for r in pending)


# ── 4090 이 잡았던 결함을 여기서 잡는가 ─────────────────────────────────


def test_a_full_lot_comes_back_whole(shaped_lookup):
    """로트 하나를 조회하면 100장이 다 오고 결함 10장이 들어 있는가.

    **4090 에서 터진 자리다.** 기본 상한 50 에 잘리면 정상이 앞에 있어서
    결함이 먼저 사라지고, 파이프라인은 "미검 없음"이라고 답한다.
    """
    lot = next(r.lot for r in shaped_lookup.catalog if r.split == "operation")
    got = shaped_lookup.find_images(lot=lot, limit=10_000)

    assert len(got) == LOT_SIZE, f"로트 {LOT_SIZE}장인데 {len(got)}장만 왔다"
    assert sum(1 for r in got if r.ground_truth == "defect") == DEFECT_PER_LOT


def test_the_default_limit_would_have_dropped_every_defect(shaped_lookup):
    """상한을 안 넘기면 결함이 전부 사라지는 것을 못박아 둔다.

    이 시험이 실패하면 **결함이 잘리지 않게 됐다는 뜻이 아니라** 카탈로그
    순서가 바뀌었다는 뜻이다. 그때는 위 시험이 규모 결함을 못 잡는다.
    """
    lot = next(r.lot for r in shaped_lookup.catalog if r.split == "operation")
    truncated = shaped_lookup.find_images(lot=lot)          # limit 기본값

    assert len(truncated) < LOT_SIZE
    assert not [r for r in truncated if r.ground_truth == "defect"], (
        "정상이 앞에 오지 않는다 — 잘림 결함을 재현하지 못한다"
    )


def test_the_pipeline_keeps_the_defects_at_real_lot_size(shaped_lookup, demo_factory):
    """파이프라인이 실제 로트 크기에서도 결함을 잃지 않는가.

    조회 계층만 보는 위 시험과 달리 **`lookup_mes` 를 실제로 부른다.**
    """
    from agents.adapters import build_adapters
    from app.pipeline import _DemoSession

    record = next(r for r in shaped_lookup.catalog
                  if r.split == "operation" and r.ground_truth == "defect"
                  and r.object_name == "pcb1")
    session = _DemoSession(
        demo_factory, "x", {}, None, build_adapters(), 2.20, shaped_lookup
    )
    session.intake_issue(line=record.line, object_name=record.object_name,
                         defect_type="스크래치", product_id=record.product_id)
    result = session.lookup_mes()

    assert result["images"] == LOT_SIZE
    assert result["defects"] == DEFECT_PER_LOT, (
        f"결함 {DEFECT_PER_LOT}장이 와야 하는데 {result['defects']}장"
    )


def test_the_scheduler_picks_up_exactly_one_days_production(shaped_lookup):
    """스케줄러가 pending 만, 라인당 하루치만 집는가.

    다른 구간까지 집으면 **이미 판정이 끝난 것을 다시 돌린다.**
    """
    for line, obj in LINES.items():
        picked = scan_pending(shaped_lookup, line, obj)
        assert len(picked) == SPLIT_LOTS["pending"] * LOT_SIZE
        assert all(r.split == "pending" for r in picked)
        assert len({r.lot for r in picked}) == SPLIT_LOTS["pending"]


@pytest.mark.parametrize("split", ["bank", "operation", "holdout"])
def test_the_scheduler_never_touches_a_judged_split(shaped_lookup, split):
    """pending 이 아닌 구간은 하나도 집지 않는다."""
    picked = scan_pending(shaped_lookup, "line_01", "pcb1")
    assert not [r for r in picked if r.split == split]


def test_a_stale_verdict_on_pending_is_ignored(shaped_catalog):
    """pending 행에 설비 판정이 적혀 있어도 믿지 않는다.

    **manifest 를 만드는 쪽이 그 칸을 채울 수 있다.** 우리 모형은 비워 두지만
    데이터 담당의 생성기가 채우면 스케줄러가 그것을 믿고 미검을 가려서, **지금
    뱅크가 실제로 놓치는지와 무관한 답**을 낸다. 이미 잡히는 건까지 미검으로
    올려 사람이 헛일을 한다.
    """
    from dataclasses import replace

    from scheduler.nightly import is_missed

    pending = next(r for r in shaped_catalog
                   if r.split == "pending" and r.ground_truth == "defect")
    stale = replace(pending, verdict="pass")      # 지난 판정이 적혀 있다

    # 지금 뱅크는 이것을 잡는다(점수가 임계값 위). 미검이 아니다.
    assert is_missed(stale, score=9.99, threshold=2.20) is False, (
        "pending 의 지난 판정을 믿고 있다 — 이미 잡히는 건을 미검으로 올린다"
    )
    # 지금 뱅크가 놓치면 그때는 미검이다.
    assert is_missed(stale, score=0.01, threshold=2.20) is True

    judged = replace(pending, split="operation", verdict="pass")
    assert is_missed(judged, score=9.99, threshold=2.20) is True, (
        "판정이 끝난 구간은 기록을 그대로 믿는다"
    )
