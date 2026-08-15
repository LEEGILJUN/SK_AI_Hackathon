"""실제 공장과 **같은 모양**의 카탈로그를 이미지 없이 만든다.

── 왜 필요한가 ─────────────────────────────────────────────────────────

Mac 의 합성 공장은 로트가 14장이고 4090 의 실데이터는 100장이다. 이 차이
하나로 결함 두 건이 Mac 을 통과하고 4090 에서 터졌다.

    로트가 상한 50 에 잘려 결함이 먼저 사라짐   (e737a04)
    스케줄러와 파이프라인이 같은 미검을 두고 갈림 (070e665 · e737a04)

**4090 왕복은 비싸다** — 시험 한 바퀴가 13분이고 실모델 실행은 더 든다.
규모 때문에 갈리는 결함은 Mac 이 먼저 잡아야 한다.

── 무엇을 만드는가 ─────────────────────────────────────────────────────

`data/manifest.csv` 와 같은 배분의 `ImageRecord` 목록이다. 라인당 10일 ·
로트당 100장(정상 90 · 결함 10)이고 뱅크 구간만 정상뿐이다.

    bank       3로트   300장   결함 0%    초기 뱅크 구성 재료
    operation  4로트   400장   결함 10%   운영 — 미검 발생
    holdout    2로트   200장   결함 10%   게이트가 성능을 잼
    pending    1로트   100장   결함 10%   아직 검사 안 됨

라인 4개 · 로트 40개 · **4,000장**이다.

── 무엇을 안 만드는가 ──────────────────────────────────────────────────

**이미지 파일을 만들지 않는다.** 여기서 잡으려는 것은 조회·상한·순서·집계
처럼 레코드만으로 드러나는 결함이고, 그것 때문에 4,000장을 디스크에 쓰면
시험이 느려져서 아무도 안 돌린다. 추론이 필요한 시험은 `demo_factory` 를
쓴다.

`path` 는 실제로 없는 경로다. **이 카탈로그로 추론을 돌리면 안 된다.**
"""

from __future__ import annotations

from datetime import date, timedelta

from lookup.base import ImageRecord

#: 라인과 품목. `data/build_factory.py` 의 VALID_LINES 와 같아야 한다.
LINES = {
    "line_01": "pcb1",
    "line_02": "pcb2",
    "line_03": "pcb3",
    "line_04": "pcb4",
}

#: 구간마다 로트 몇 개인가. 라인당 10일이다.
SPLIT_LOTS = {"bank": 3, "operation": 4, "holdout": 2, "pending": 1}

#: 로트 하나의 구성. `data/build_factory.py` 의 값과 같아야 한다.
NORMAL_PER_LOT = 90
DEFECT_PER_LOT = 10
LOT_SIZE = NORMAL_PER_LOT + DEFECT_PER_LOT

#: 뱅크 구간은 초기 수집이라 결함이 섞이지 않는다. 뱅크 오염은 시나리오가 넣는다.
NO_DEFECTS_IN = {"bank"}

START = date(2026, 6, 1)


def build_catalog() -> list[ImageRecord]:
    """실제 manifest 와 같은 배분의 레코드 목록.

    **정상을 먼저 넣고 결함을 뒤에 넣는다.** 실제 생성기와 같은 순서이고,
    조회가 조용히 자를 때 **결함이 먼저 사라지는** 것이 이 순서 때문이다.
    순서를 바꾸면 그 결함을 못 잡는다.
    """
    catalog: list[ImageRecord] = []
    for line, obj in LINES.items():
        day = 0
        for split, lot_count in SPLIT_LOTS.items():
            for _ in range(lot_count):
                captured = START + timedelta(days=day)
                lot = f"LOT-{captured:%Y%m%d}-{line[-2:]}"
                defects = 0 if split in NO_DEFECTS_IN else DEFECT_PER_LOT
                normals = LOT_SIZE - defects
                for i in range(normals):
                    catalog.append(_record(line, obj, lot, split, captured, i, "pass"))
                for i in range(defects):
                    catalog.append(_record(line, obj, lot, split, captured, i, "defect"))
                day += 1
    return catalog


def _record(line, obj, lot, split, captured, index, truth) -> ImageRecord:
    kind = "N" if truth == "pass" else "D"
    return ImageRecord(
        product_id=f"{obj.upper()}-{line[-2:]}-{lot[-11:]}-{kind}{index:03d}",
        # 실제로 없는 경로다. 이 카탈로그로 추론을 돌리면 안 된다.
        path=f"data/factory/{line}/{obj}/{split}/{kind.lower()}{index:03d}.png",
        line=line,
        object_name=obj,
        lot=lot,
        captured_at=captured,
        split=split,
        # pending 은 아직 검사하지 않은 구간이라 설비 판정이 없다.
        verdict=None if split == "pending" else "pass",
        ground_truth=truth,
        equipment=f"CAM-{line[-2:]}-{(index % 2) + 1}",
    )


def counts_by_split(catalog) -> dict[str, int]:
    out: dict[str, int] = {}
    for record in catalog:
        out[record.split] = out.get(record.split, 0) + 1
    return out
