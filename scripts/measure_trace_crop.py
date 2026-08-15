"""역추적 좌표로 잘라도 시각 언어 모델이 결함을 보는가 — 판별 1번의 실전 가능 여부.

앞선 실험에서 전체 이미지를 그대로 주면 결함을 놓쳤고, VisA 정답 마스크로
결함 위치를 잘라 확대해 주면 틀리지 않았다. 그런데 **실제 운영에는 마스크가
없다.** 그래서 이 측정이 필요하다.

마스크 대신 PatchCore 의 이상 점수가 가장 높은 패치로 위치를 짚는다.
미검출은 "점수가 임계값 아래"라는 뜻이지 "히트맵이 엉뚱한 곳을 가리킨다"는
뜻이 아니므로 될 법하지만, 그것은 추정이고 여기서 재는 것이 사실이다.

세 가지를 잰다.

    1. 역추적 좌표가 실제 결함과 겹치는가   ← 이게 먼저다. 마스크와 대조
    2. 그 크롭으로 시각 언어 모델이 맞히는가
    3. 정답 마스크로 자른 것과 얼마나 차이 나는가

읽기만 한다. 저장소 파일을 고치지 않는다.

    python scripts/measure_trace_crop.py --max-normal 150 --defects 10

── 실측 (2026-08-14, 4090, capsules, 정상 150장, qwen3vl-8b) ─────────────

  역추적 중심이 결함 위       8/10
  역추적 크롭이 결함을 담음  10/10
  역추적 크롭으로 defect 판정 9/10
  마스크 크롭으로 defect 판정 8/10   ← 상한선

**역추적이 마스크보다 나았다.** 마스크 박스는 결함 크기에 따라 달라져 작은
결함이면 맥락이 부족해지는데, 역추적 크롭은 격자 한 칸 기준이라 항상 같은
크기다. 고정 크기가 가변 크기보다 낫다.

── margin 기본값이 64 인 이유 ────────────────────────────────────────────

같은 모델·좌표·이미지에서 여유 픽셀만 바꿔 재봤다.

  여유 24px → 크롭  63x64  →  역추적 0/10 · 마스크 0/10   (스무 건 전부 unknown)
  여유 64px → 크롭 143x144 →  역추적 9/10 · 마스크 8/10

24px 에서 전부 unknown 이 나온 것은 모델이 지어내지 않고 "무엇을 보는지
모르겠다"고 답한 것이다. 격자 한 칸이 원본에서 15x16px 뿐이라 그것만 떼어
주면 제품의 어느 부위인지 알 수 없다.

**판독을 좌우한 것은 모델이 아니라 크롭 크기였다.** 해상도 256/224 에서
결함이 격자 한 칸에 묻혔던 것, 1500px 전체 화면에서 묻혔던 것과 같은 구조다.
셋 다 "모델에게 무엇을 얼마나 보여주는가"의 문제이지 모델을 바꿔 푸는
문제가 아니었다.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.adapters import build_adapters  # noqa: E402
from agents.vision import judge_defect_visible  # noqa: E402
from inspection import PatchEmbedder, build_bank, score_image  # noqa: E402
from inspection.crop import patch_box  # noqa: E402

ENLARGE_TO = 512  # 잘라낸 조각의 짧은 변을 이만큼 키운다. 격자 한 칸은 수십 px 뿐이다

#: 패치 자리 사방 여유 픽셀. 24 로 두면 전부 unknown 이 나온다 (위 docstring 참조).
DEFAULT_MARGIN = 64


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--category", default="capsules")
    p.add_argument("--visa-root", default="VisA_20220922")
    p.add_argument("--max-normal", type=int, default=150, help="뱅크에 넣을 정상 이미지 수")
    p.add_argument("--defects", type=int, default=10, help="검사할 결함 이미지 수")
    p.add_argument(
        "--margin", type=int, default=DEFAULT_MARGIN,
        help=f"패치 자리 사방 여유 픽셀 (기본 {DEFAULT_MARGIN}). "
             f"24 로 낮추면 맥락이 사라져 전부 unknown 이 된다",
    )
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    return p.parse_args()


def enlarge(img: Image.Image) -> Image.Image:
    """짧은 변이 ENLARGE_TO 보다 작으면 키운다. 확대는 정보를 늘리지 않지만
    모델이 볼 수 있는 크기로 만들어 준다."""
    short = min(img.size)
    if short >= ENLARGE_TO:
        return img
    scale = ENLARGE_TO / short
    return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)


def mask_array(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path).convert("L")) > 0


def mask_box(mask: np.ndarray, margin: int, size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    w, h = size
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + margin), min(h, int(ys.max()) + margin))


def overlap_stats(box: tuple[int, int, int, int], mask: np.ndarray) -> tuple[bool, float]:
    """크롭 영역이 결함을 얼마나 담았는가.

    center_hit
        패치 자리의 중심이 결함 픽셀 위에 있는가. 가장 엄격한 기준이며
        **히트맵 정확도의 실제 값은 이쪽이다.**
    captured
        전체 결함 픽셀 중 이 영역 안에 들어온 비율. 판독에는 이쪽이 실질적이다.
        다만 결함이 30~45px 인데 박스가 143px 이라 조금 빗나가도 담기므로,
        이 값이 높다고 히트맵이 정확하다고 말하면 안 된다.
    """
    left, top, right, bottom = box
    h, w = mask.shape
    inside = mask[max(0, top):min(h, bottom), max(0, left):min(w, right)]
    total = int(mask.sum())
    captured = float(inside.sum()) / total if total else 0.0

    cy, cx = (top + bottom) // 2, (left + right) // 2
    center_hit = bool(0 <= cy < h and 0 <= cx < w and mask[cy, cx])
    return center_hit, captured


def main() -> int:
    args = parse_args()
    base = REPO_ROOT / args.visa_root / args.category / "Data"
    normal_dir, defect_dir, mask_dir = (base / "Images" / "Normal",
                                        base / "Images" / "Anomaly",
                                        base / "Masks" / "Anomaly")
    if not normal_dir.exists():
        print(f"VisA 를 못 찾았다: {normal_dir}")
        return 1

    normals = sorted(normal_dir.glob("*.JPG"))[: args.max_normal]
    defects = sorted(defect_dir.glob("*.JPG"))[: args.defects]

    embedder = PatchEmbedder()
    print(f"장치 {embedder.device} · 정상 {len(normals)}장으로 뱅크 구성 중...")
    bank = build_bank(normals, embedder, coreset_ratio=args.coreset_ratio,
                      bank_version="trace_crop_probe")
    print(f"뱅크 패치 {bank.embeddings.shape[0]:,}개\n")

    _, vlm = build_adapters()
    tmp = Path(tempfile.mkdtemp(prefix="trace_crop_"))
    print(f"시각 언어 모델 {vlm.describe()} · 여유 {args.margin}px · 확대 {ENLARGE_TO}px\n")

    head = (f"{'이미지':<11} {'점수':>7} {'좌표':>9} {'중심':>5} {'담김':>7} {'크롭':>10} "
            f"{'역추적크롭':<10} {'마스크크롭':<10}")
    print(head)
    print("-" * len(head))
    grid_printed = False

    center_hits = trace_ok = mask_ok = captured_ok = 0
    counted = 0

    for p in defects:
        mp = mask_dir / (p.stem + ".png")
        if not mp.exists():
            continue
        mask = mask_array(mp)
        img = Image.open(p)

        result = score_image(p, bank, embedder, top_k=1, root=REPO_ROOT)
        ref = result.matches[0].query
        grid = (result.grid_h, result.grid_w)
        box = patch_box(ref, grid, img.size, embedder.config, margin=args.margin)

        if not grid_printed:
            cell = patch_box(ref, grid, img.size, embedder.config, margin=0)
            print(f"# 격자 {grid[0]}x{grid[1]} · 원본 {img.size[0]}x{img.size[1]} · "
                  f"격자 한 칸 {cell[2]-cell[0]}x{cell[3]-cell[1]}px\n")
            grid_printed = True

        center_hit, captured = overlap_stats(box, mask)
        center_hits += int(center_hit)
        captured_ok += int(captured > 0)

        # 역추적 좌표로 자른 것
        tp = tmp / f"t_{p.stem}.png"
        enlarge(img.crop(box)).save(tp)
        tj = judge_defect_visible(vlm, tp, reported_defect="표면 결함")

        # 정답 마스크로 자른 것 — 비교용
        mbox = mask_box(mask, args.margin, img.size)
        mp_out = tmp / f"m_{p.stem}.png"
        enlarge(img.crop(mbox)).save(mp_out)
        mj = judge_defect_visible(vlm, mp_out, reported_defect="표면 결함")

        # **판별 1번 어휘다.** 전에 여기서 5번 어휘("defect")와 견주어
        # 판독이 항상 0/10 으로 나올 뻔했다.
        trace_ok += int(tj.verdict == "visible")
        mask_ok += int(mj.verdict == "visible")
        counted += 1

        size = f"{box[2]-box[0]}x{box[3]-box[1]}"
        print(f"{p.name:<11} {result.score:>7.3f} {f'({ref.row},{ref.col})':>9} "
              f"{'O' if center_hit else 'X':>5} {captured:>6.0%} {size:>10}  "
              f"{tj.verdict:<10} {mj.verdict:<10}")

    if not counted:
        print("마스크가 있는 결함 이미지를 찾지 못했다.")
        return 1

    print(f"\n결함 {counted}장")
    print(f"  역추적 중심이 결함 위      {center_hits}/{counted}   ← 히트맵 정확도의 실제 값")
    print(f"  역추적 크롭이 결함을 담음  {captured_ok}/{counted}")
    print(f"  역추적 크롭으로 defect 판정 {trace_ok}/{counted}")
    print(f"  마스크 크롭으로 defect 판정 {mask_ok}/{counted}   ← 상한선")
    print("\n역추적이 마스크에 가까울수록 판별 1번을 실전에 쓸 수 있다.")
    print("담김 비율은 결함이 작고 박스가 커서 높게 나온다. 히트맵 정확도로 인용하지 말 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
