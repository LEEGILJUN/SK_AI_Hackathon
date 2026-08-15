"""판별 7번 면적이 실데이터에서 어떻게 나오는지 잰다 — 장영진에게 넘길 근거.

    python scripts/measure_mask.py                 # 기본 품목
    python scripts/measure_mask.py --object pcb2
    python scripts/measure_mask.py --images 20

**값을 고치지 않는다.** `data/criteria.yaml` 은 장영진 소유이고 면적 기준
150px² 와 `binarize_threshold: 0.5` 에는 아직 근거 TODO 가 붙어 있다. 이
스크립트는 그 근거를 만들기 위한 측정만 한다.

── 왜 재는가 ───────────────────────────────────────────────────────────

4090 실측에서 결함 이미지 면적이 **52,995px²** 로 나왔다. 448×448 = 200,704
이므로 화면의 26% 다. pcb 결함이 그렇게 클 리 없다.

의심되는 자리는 정규화 기준이다. `inspection/mask.py` 는 격자값을 **운영
임계값**으로 나눈다. 그 근거였던 "정상 최대 0.05 · 결함 1.0 근처" 는 합성
데이터 실측이고, **합성 측정이 VisA 로 옮겨가지 않은 전례가 여러 번 있다.**

미검 이미지는 정의상 최고 점수가 임계값 아래다. 컷오프가 임계값의 절반이면
텍스처가 있는 표면에서 상당수 패치가 그것을 넘을 수 있다.

── 무엇을 내는가 ───────────────────────────────────────────────────────

정상과 결함 각각에 대해 컷오프를 바꿔 가며 면적을 낸다. **정상과 결함이
갈리는 컷오프**가 있는지, 있다면 그때 면적이 얼마인지가 답이다.

정상 면적이 결함 면적과 겹치면 그 컷오프로는 못 가른다 — 기준값을 아무리
바꿔도 소용없고, 정규화 방식 자체를 바꿔야 한다는 뜻이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from inspection.mask import defect_area  # noqa: E402

#: 재볼 컷오프. criteria.yaml 의 현재 값 0.5 를 가운데 두고 넓게 훑는다.
CUTOFFS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.1)


def main() -> int:
    parser = argparse.ArgumentParser(description="판별 7번 면적 측정")
    parser.add_argument("--object", default="pcb1")
    parser.add_argument("--line", default="line_01")
    parser.add_argument("--images", type=int, default=10, help="정상·결함 각 몇 장")
    args = parser.parse_args()

    from app.pipeline import DemoFactory

    factory = DemoFactory()
    item = factory.item_for(args.line, args.object)
    if item is None:
        print(f"{args.line}/{args.object} 에 뱅크가 없습니다.")
        return 1

    records = [r for r in factory.catalog
               if r.line == args.line and r.object_name == args.object]
    normals = [r for r in records if r.ground_truth == "pass"][: args.images]
    defects = [r for r in records if r.ground_truth == "defect"][: args.images]
    if not normals or not defects:
        print("정상 또는 결함 이미지가 없습니다.")
        return 1

    crop = factory.embedder.config.crop
    threshold = 2.20
    print(f"품목 {args.object} · 입력 {crop}px ({crop * crop:,}px²) · 임계값 {threshold}")
    print(f"정상 {len(normals)}장 · 결함 {len(defects)}장\n")

    from inspection import score_images

    scored = {}
    for label, group in (("정상", normals), ("결함", defects)):
        paths = [factory.resolve(r.path) for r in group]
        scored[label] = score_images(paths, item.bank, factory.embedder,
                                     root=factory.root)

    print("이상 점수")
    for label, results in scored.items():
        values = [r.score for r in results]
        print(f"  {label}  최소 {min(values):.3f} · 중앙 {np.median(values):.3f} "
              f"· 최대 {max(values):.3f}")

    print("\n컷오프별 가장 큰 덩어리 면적 (px²) — 중앙값")
    print(f"  {'컷오프':>7} {'정상':>12} {'결함':>12}   {'갈리는가':>8}")
    for cutoff in CUTOFFS:
        row = {}
        for label, results in scored.items():
            areas = [
                defect_area(r.patch_distances, crop=crop, threshold=threshold,
                            binarize_threshold=cutoff)
                for r in results
            ]
            row[label] = areas
        n_med, d_med = np.median(row["정상"]), np.median(row["결함"])
        separates = "예" if max(row["정상"]) < min(row["결함"]) else "아니오"
        print(f"  {cutoff:>7.2f} {n_med:>12,.0f} {d_med:>12,.0f}   {separates:>8}")

    print(
        "\n'갈리는가' 는 **정상 최대 < 결함 최소** 일 때만 예입니다.\n"
        "전부 아니오면 컷오프로는 못 가릅니다 — 기준값을 바꿀 것이 아니라\n"
        "정규화 방식을 바꿔야 한다는 뜻입니다.\n"
        "\n이 표를 장영진에게 그대로 넘기면 됩니다. criteria.yaml 은 고치지 마세요."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
