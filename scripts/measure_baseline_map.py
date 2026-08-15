"""역추적이 결함 자리를 짚는가 — 자리별 기준선을 넣으면 나아지는가.

    python scripts/measure_baseline_map.py --category pcb1
    python scripts/measure_baseline_map.py --category pcb1 --top 5

**시각 언어 모델을 부르지 않는다.** 정답 마스크와 좌표만 대조하므로 빠르고,
모델 판독 성능과 히트맵 위치 정확도를 섞지 않는다. 4090 측정에서 마스크로
정확히 잘라 주면 9/10 이 나왔으므로 **모델은 문제가 아니고 위치가 문제**다.
그 위치만 따로 잰다.

── 무엇을 비교하는가 ───────────────────────────────────────────────────

    raw       지금 동작. 거리가 가장 큰 자리
    residual  자리별 중앙값을 뺀 뒤 가장 큰 자리
    robust    뺀 뒤 자리별 변동 폭으로 나눈 뒤 가장 큰 자리

pcb1 에서 좌표가 행 43~45 에 몰렸다. 기판의 특정 구조물이 늘 거리가 크다는
뜻이고, 그렇다면 그 자리의 기준을 빼면 진짜 결함이 드러나야 한다.

`--top N` 을 주면 상위 N 자리 중 하나라도 결함 위면 맞은 것으로 센다. 크롭을
여러 장 모델에 주는 방식이 통할지 미리 보는 값이다.

── 무엇을 안 하는가 ────────────────────────────────────────────────────

**추론 경로를 바꾸지 않는다.** 이 스크립트는 재기만 한다. 어느 방식을 쓸지는
이 표를 보고 사람이 정한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from inspection import PatchEmbedder, build_bank, score_image  # noqa: E402
from inspection.baseline_map import MODES, build_baseline, hottest_cells  # noqa: E402
from inspection.crop import patch_box  # noqa: E402
from inspection.types import PatchRef  # noqa: E402
from app.pipeline import visa_category_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="역추적 위치 정확도")
    parser.add_argument("--category", default="pcb1")
    parser.add_argument("--bank-normals", type=int, default=150, help="뱅크 구성 장수")
    parser.add_argument("--baseline-normals", type=int, default=50,
                        help="자리별 기준선을 만들 정상 장수. 뱅크와 겹치지 않게 뒤에서 집는다")
    parser.add_argument("--defects", type=int, default=10)
    parser.add_argument("--top", type=int, default=1, help="상위 몇 자리까지 맞으면 인정")
    parser.add_argument("--margin", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = visa_category_dir(args.category, REPO_ROOT)
    normal_dir, defect_dir, mask_dir = (base / "Images" / "Normal",
                                        base / "Images" / "Anomaly",
                                        base / "Masks" / "Anomaly")
    if not normal_dir.exists():
        print(f"카테고리를 찾지 못했습니다: {base}")
        return 1

    normals = sorted(normal_dir.glob("*.JPG"))
    defects = sorted(defect_dir.glob("*.JPG"))[: args.defects]
    bank_normals = normals[: args.bank_normals]
    # **뱅크에 쓴 것과 겹치지 않게 뒤에서 집는다.** 뱅크 구성 이미지는 자기
    # 자신과의 거리가 0 에 가까워 기준선이 실제보다 낮아진다.
    baseline_normals = normals[args.bank_normals:][: args.baseline_normals]
    if not baseline_normals:
        print("기준선을 만들 정상 이미지가 모자랍니다. --bank-normals 를 줄이세요.")
        return 1

    embedder = PatchEmbedder()
    print(f"뱅크 구성 {len(bank_normals)}장 · 기준선 {len(baseline_normals)}장 "
          f"· 결함 {len(defects)}장")
    bank = build_bank(bank_normals, embedder, root=REPO_ROOT)

    baseline = build_baseline([
        score_image(p, bank, embedder, top_k=1, root=REPO_ROOT).patch_distances
        for p in baseline_normals
    ])
    median = np.asarray(baseline["median"])
    print(f"격자 {baseline['grid'][0]}x{baseline['grid'][1]} · "
          f"자리별 기준 중앙값 최소 {median.min():.3f} · 최대 {median.max():.3f}")
    hot_row, hot_col = np.unravel_index(int(np.argmax(median)), median.shape)
    print(f"정상에서도 늘 큰 자리: ({hot_row},{hot_col})\n")

    header = f"{'이미지':<11}" + "".join(f"{m:>26}" for m in MODES)
    print(header)
    print("─" * len(header))

    hits = {mode: 0 for mode in MODES}
    counted = 0
    for path in defects:
        mask_path = mask_dir / (path.stem + ".png")
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        image = Image.open(path)
        result = score_image(path, bank, embedder, top_k=1, root=REPO_ROOT)
        grid = (result.grid_h, result.grid_w)

        cells = []
        for mode in MODES:
            picked = hottest_cells(result.patch_distances, args.top,
                                   baseline=baseline, mode=mode)
            hit = False
            for row, col in picked:
                ref = PatchRef(source_image=str(path), row=row, col=col,
                               patch_index=row * grid[1] + col)
                left, top, right, bottom = patch_box(ref, grid, image.size,
                                                     embedder.config, margin=args.margin)
                cy, cx = (top + bottom) // 2, (left + right) // 2
                h, w = mask.shape
                if 0 <= cy < h and 0 <= cx < w and mask[cy, cx]:
                    hit = True
                    break
            hits[mode] += int(hit)
            cells.append(f"{str(picked[0]):>14} {'O' if hit else 'X':>3}")
        counted += 1
        print(f"{path.name:<11}" + "".join(f"{c:>26}" for c in cells))

    print(f"\n결함 {counted}장 · 상위 {args.top}자리 기준 중심 적중")
    for mode in MODES:
        print(f"  {mode:<10} {hits[mode]}/{counted}")
    print(
        "\n**raw 가 지금 동작입니다.** residual 이나 robust 가 뚜렷하게 높으면\n"
        "자리별 기준선을 넣을 근거가 됩니다. 비슷하면 넣지 마세요 — 단계가\n"
        "하나 늘고 정상 이미지를 더 돌려야 합니다.\n"
        "\n셋 다 낮으면 위치 문제가 아니라 이 품목에서 PatchCore 거리 자체가\n"
        "결함을 짚지 못하는 것이고, 그때는 판별 1번을 단독 차단 조건에서 빼는\n"
        "쪽이 답입니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
