"""가상 공장 데이터 검사 — 이동현용.

`data/build_factory.py` 로 만든 결과가 진단 에이전트가 쓸 수 있는 형태인지
확인합니다. **무엇이 잘못됐는지 한국어로 알려주므로 코드를 읽지 않아도 됩니다.**

실행:
    .venv/bin/python scripts/check_factory.py

검사 순서 (앞에서 막히면 뒤는 건너뜁니다)
    1. manifest.csv 가 있고 필요한 열이 다 있는가
    2. manifest 에 적힌 이미지가 실제로 존재하는가
    3. mes.csv 가 manifest 와 lot_id 로 붙는가
    4. 뱅크 구간과 운영 구간이 나뉘어 있는가
    5. 진단이 물어볼 조건(일자·로트)이 조회 가능한가

한 번에 다 맞추려 하지 마세요. 1번부터 하나씩 초록으로 만들면 됩니다.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MANIFEST = REPO_ROOT / "data" / "manifest.csv"
MES = REPO_ROOT / "data" / "mes.csv"
FACTORY = REPO_ROOT / "data" / "factory"

MANIFEST_COLUMNS = [
    "image_path", "line", "object", "date", "lot_id",
    "equipment_id", "split", "label", "mask_path", "visa_source",
]
MES_COLUMNS = [
    "lot_id", "line", "object", "date", "started_at",
    "equipment_id", "inspected_count", "defect_count", "operator_shift",
]
#: pending 은 "아직 검사 안 된 생산분"이다. 예약 스케줄러와 섀도 평가가
#: 이 구간을 쓴다 — 밤새 쌓인 것이 없으면 스케줄러가 할 일이 없다.
VALID_SPLIT = {"bank", "operation", "holdout", "pending"}
VALID_LABEL = {"normal", "defect"}

OK = "  ✓"
NG = "  ✗"


def fail(message: str, hint: str = "") -> None:
    print(f"{NG} {message}")
    if hint:
        for line in hint.splitlines():
            print(f"      {line}")


def main() -> int:
    try:
        import pandas as pd
    except ImportError:
        print("pandas 가 없습니다. pip install -r requirements.txt 를 먼저 실행하세요.")
        return 1

    print("가상 공장 데이터 검사\n")

    # ── 1. manifest.csv ─────────────────────────────────────────────
    print("1. manifest.csv")
    if not MANIFEST.exists():
        fail(
            f"{MANIFEST.relative_to(REPO_ROOT)} 가 없습니다.",
            "data/build_factory.py 가 이 파일을 만들어야 합니다.\n"
            "형식은 examples/가상공장_구조.md 를 보세요.",
        )
        return 1

    manifest = pd.read_csv(MANIFEST)
    missing = [c for c in MANIFEST_COLUMNS if c not in manifest.columns]
    if missing:
        fail(
            f"열이 빠졌습니다: {', '.join(missing)}",
            f"필요한 열: {', '.join(MANIFEST_COLUMNS)}",
        )
        return 1
    print(f"{OK} 열 {len(MANIFEST_COLUMNS)}개 확인, 행 {len(manifest):,}개")

    bad_split = set(manifest["split"].dropna().unique()) - VALID_SPLIT
    if bad_split:
        fail(f"split 열에 모르는 값이 있습니다: {sorted(bad_split)}",
             f"허용: {sorted(VALID_SPLIT)}")
    bad_label = set(manifest["label"].dropna().unique()) - VALID_LABEL
    if bad_label:
        fail(f"label 열에 모르는 값이 있습니다: {sorted(bad_label)}",
             f"허용: {sorted(VALID_LABEL)}")

    # ── 2. 이미지가 실제로 있는가 ───────────────────────────────────
    print("\n2. 이미지 존재 확인")
    summary = REPO_ROOT / "data" / "factory_summary.txt"
    if summary.exists() and "건너뜀 (--no-images)" in summary.read_text(encoding="utf-8"):
        print(f"{OK} 건너뜁니다 — --no-images 로 만든 데이터입니다")
        print("      조회 계층 개발에는 CSV 만 있으면 되지만, **시연에는 이미지가 필요합니다.**")
        print("      시연용은 `python data/build_factory.py` 로 다시 만드세요 (4090 에서).")
    else:
        sample = manifest["image_path"].head(200)
        absent = [p for p in sample if not (FACTORY / str(p)).exists()]
        if absent:
            fail(
                f"manifest 에 적힌 이미지 중 {len(absent)}개를 찾지 못했습니다 (200개 표본).",
                f"예: {absent[0]}\n"
                f"image_path 는 data/factory/ 기준 상대 경로여야 합니다.\n"
                f"절대 경로나 VisA 원본 경로를 넣으면 안 됩니다.",
            )
        else:
            print(f"{OK} 표본 {len(sample)}개 모두 존재")

    # ── 3. mes.csv 와 조인 ──────────────────────────────────────────
    print("\n3. mes.csv 와 조인")
    if not MES.exists():
        fail(f"{MES.relative_to(REPO_ROOT)} 가 없습니다.")
        return 1

    mes = pd.read_csv(MES)
    missing = [c for c in MES_COLUMNS if c not in mes.columns]
    if missing:
        fail(f"열이 빠졌습니다: {', '.join(missing)}",
             f"필요한 열: {', '.join(MES_COLUMNS)}")
        return 1
    print(f"{OK} 열 {len(MES_COLUMNS)}개 확인, 로트 {len(mes):,}개")

    manifest_lots = set(manifest["lot_id"].dropna().unique())
    mes_lots = set(mes["lot_id"].dropna().unique())

    orphan = manifest_lots - mes_lots
    if orphan:
        fail(
            f"manifest 에는 있는데 mes 에 없는 로트가 {len(orphan)}개 있습니다.",
            f"예: {sorted(orphan)[:3]}\n"
            f"진단이 '이 로트의 생산 기록'을 조회할 때 빕니다.",
        )
    empty = mes_lots - manifest_lots
    if empty:
        fail(
            f"mes 에는 있는데 이미지가 하나도 없는 로트가 {len(empty)}개 있습니다.",
            f"예: {sorted(empty)[:3]}",
        )
    if not orphan and not empty:
        print(f"{OK} lot_id 로 양쪽이 정확히 붙습니다 ({len(manifest_lots)}개 로트)")

    # ── 4. 구간 분리 ────────────────────────────────────────────────
    print("\n4. 뱅크 구간과 운영 구간")
    bank_rows = manifest[manifest["split"] == "bank"]
    op_rows = manifest[manifest["split"] == "operation"]

    if bank_rows.empty:
        fail("split=bank 인 행이 없습니다.", "1~4일분을 뱅크 구성용으로 표시해야 합니다.")
    if op_rows.empty:
        fail("split=operation 인 행이 없습니다.", "5~7일분을 운영 데이터로 표시해야 합니다.")

    if not bank_rows.empty and not op_rows.empty:
        # **라인별로 비교합니다.** 화질 기준 분포는 라인·품목마다 따로 뽑으므로
        # 구간도 라인 안에서만 갈리면 됩니다. 전역으로 비교하면 2라인의 초기
        # 수집일이 1라인의 운영일과 같은 날이라는 이유로 걸립니다 — 서로
        # 아무 상관이 없는데도요. 실제로 그 오탐이 났습니다 (2026-08-15).
        overlaps: list[str] = []
        for line in sorted(set(manifest["line"].dropna().unique())):
            bank_dates = set(bank_rows[bank_rows["line"] == line]["date"].unique())
            op_dates = set(op_rows[op_rows["line"] == line]["date"].unique())
            shared = bank_dates & op_dates
            if shared:
                overlaps.append(f"{line}: {sorted(shared)[:3]}")
        if overlaps:
            fail(
                f"뱅크 구간과 운영 구간의 일자가 겹칩니다 — {' / '.join(overlaps[:3])}",
                "화질 기준 분포를 뱅크 구간에서만 뽑아야 하는데, 겹치면\n"
                "열화가 주입된 운영 데이터가 섞여 기준이 뱅크 오염됩니다.\n"
                "그러면 설비·광학 원인을 영영 잡지 못합니다.",
            )
        else:
            n_bank = len(set(bank_rows["date"].unique()))
            n_op = len(set(op_rows["date"].unique()))
            print(f"{OK} 라인마다 뱅크 구간과 운영 구간이 갈립니다 (뱅크 {n_bank}일 / 운영 {n_op}일)")

        if not (bank_rows["label"] == "normal").all():
            n = int((bank_rows["label"] != "normal").sum())
            fail(
                f"뱅크 구간에 결함으로 표시된 이미지가 {n}개 있습니다.",
                "뱅크는 정상 이미지로만 만듭니다. 오염 시나리오는 injection 으로\n"
                "일부러 섞는 것이며, manifest 의 label 은 사실대로 적어야 합니다.",
            )
        else:
            print(f"{OK} 뱅크 구간은 전부 정상 이미지")

    # ── 5. 진단이 물어볼 조건 ───────────────────────────────────────
    print("\n5. 진단이 조회할 조건")
    for column, what in (("date", "일자"), ("lot_id", "로트"), ("equipment_id", "설비")):
        values = manifest[column].dropna().unique()
        if len(values) <= 1:
            fail(
                f"{what}({column}) 값이 {len(values)}종류뿐입니다.",
                "커버리지 부족 시나리오를 만들려면 조건이 여러 개여야 합니다.\n"
                "특정 조건을 뱅크에서 빼는 것이 그 시나리오의 재현 방법입니다.",
            )
        else:
            print(f"{OK} {what} {len(values)}종류")

    lines = manifest["line"].dropna().unique()
    if len(lines) < 3:
        print(f"  · 라인이 {len(lines)}개입니다. 3~4개를 권장합니다 (VisA 객체 3~4종).")
    else:
        print(f"{OK} 라인 {len(lines)}개: {', '.join(sorted(lines))}")

    print("\n" + "─" * 60)
    print("여기까지 통과하면 조회 계층(lookup/) 구현으로 넘어가시면 됩니다.")
    print("그다음 확인: .venv/bin/python -m pytest tests/test_lookup_contract.py -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
