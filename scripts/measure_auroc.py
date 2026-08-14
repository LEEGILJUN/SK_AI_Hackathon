"""카테고리별 AUROC — 조건을 고정해서 나란히 놓을 수 있게 잰다.

**AUROC 는 홀드아웃 구성이 다르면 직접 비교할 수 없다**(CLAUDE.md). 그래서
이 스크립트는 분할을 인자로 못박고 결과에 함께 찍는다. 조건 없는 AUROC 는
나중에 쓸 수 없다.

demo_trace.py 로는 이 표를 못 만든다. 그쪽은 holdout_size = len(normal)//4 로
나누고 결함을 12장만 쓴다 — 정상 150장을 주면 뱅크 113 · 홀드아웃 37/12 가 된다.
여기서는 뱅크와 홀드아웃을 겹치지 않게 따로 떼어 낸다.

── 분할 규칙 ─────────────────────────────────────────────────────────────

    normals = sorted(Normal/*.JPG)
    bank_normal     = normals[0 : B]           기본 B=150
    holdout_normal  = normals[B : B+N]         기본 N=25   (뱅크와 겹치지 않음)
    holdout_defect  = sorted(Anomaly/*.JPG)[0 : D]   기본 D=25

파일명 정렬 순서를 그대로 쓴다. 무작위로 뽑지 않는 이유는 같은 명령이 같은
수치를 내야 하기 때문이다. 오염은 넣지 않는다(깨끗한 뱅크).

── 교차 측정 ─────────────────────────────────────────────────────────────

    --cross pcb1:pcb2

pcb1 정상으로 뱅크를 세우고 pcb2 의 정상 이미지를 재서 점수가 어디에 찍히는지
본다. 같은 보드의 앞뒤면(pcb1·pcb2)은 배경과 조명이 같으므로, 점수가 뜨면
그 이유가 "배경이 달라서"가 아니라 "정상 패치가 달라서"로 좁혀진다.
뱅크를 라인마다 따로 두어야 하는지가 여기서 갈린다.

── 실측 (2026-08-14, 4090, 512/448, 정상 150장, 홀드아웃 25/25) ──────────

    pcb1      1.000   겹침 없음
    pcb2      0.998   겹침 있음
    pcb3      1.000   겹침 없음
    pcb4      1.000   겹침 없음
    capsules  0.989   겹침 있음   ← 같은 조건. 기판이 캡슐보다 낫다

512/448 이 기판에서 모자랄 것이라는 우려는 재현되지 않았다. 768 은 조건이
성립하지 않아 재지 않았다.

교차: pcb1 뱅크로 재면 pcb2 정상이 3.421~3.643 으로, pcb1 결함 중앙값
3.172 보다도 높다. AUROC 1.000, 겹침 없음. 뱅크는 라인마다 따로여야 한다.

읽기만 한다. 저장소 파일을 고치지 않는다.

    python scripts/measure_auroc.py --categories pcb1 pcb2 pcb3 pcb4
    python scripts/measure_auroc.py --categories pcb1 --resize 768 --crop 672
    python scripts/measure_auroc.py --cross pcb1:pcb2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inspection import PatchEmbedder, build_bank, score_images  # noqa: E402
from inspection.features import FeatureConfig  # noqa: E402
from inspection.sweep import sweep_thresholds  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="카테고리별 AUROC")
    p.add_argument("--visa-root", default="VisA_20220922")
    p.add_argument("--categories", nargs="*", default=[])
    p.add_argument("--cross", default=None, help="뱅크:질의 (예: pcb1:pcb2)")
    p.add_argument("--bank-normal", type=int, default=150)
    p.add_argument("--holdout-normal", type=int, default=25)
    p.add_argument("--holdout-defect", type=int, default=25)
    p.add_argument("--resize", type=int, default=None, help="비우면 FeatureConfig 기본(512)")
    p.add_argument("--crop", type=int, default=None, help="비우면 FeatureConfig 기본(448)")
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def make_config(args: argparse.Namespace) -> FeatureConfig:
    if args.resize is None and args.crop is None:
        return FeatureConfig()
    base = FeatureConfig()
    return FeatureConfig(
        backbone=base.backbone,
        layers=base.layers,
        weights=base.weights,
        resize=args.resize or base.resize,
        crop=args.crop or base.crop,
        neighborhood=base.neighborhood,
    )


def split(root: Path, category: str, args: argparse.Namespace):
    base = root / category / "Data" / "Images"
    normals = sorted((base / "Normal").glob("*.JPG"))
    defects = sorted((base / "Anomaly").glob("*.JPG"))
    b, n, d = args.bank_normal, args.holdout_normal, args.holdout_defect
    if len(normals) < b + n:
        raise SystemExit(f"{category}: 정상이 {len(normals)}장뿐이라 {b}+{n} 을 못 뗀다")
    return normals[:b], normals[b:b + n], defects[:d]


def build(category: str, bank_normal, embedder, args):
    return build_bank(
        bank_normal, embedder,
        coreset_ratio=args.coreset_ratio, seed=args.seed,
        bank_version=f"{category}-auroc",
    )


def main() -> int:
    args = parse_args()
    root = REPO_ROOT / args.visa_root
    config = make_config(args)
    embedder = PatchEmbedder(config)

    print("측정 조건")
    print(f"  해상도       {config.resize}/{config.crop}")
    print(f"  백본         {config.backbone} · layers {list(config.layers)}")
    print(f"  coreset      {args.coreset_ratio}  (seed {args.seed})")
    print(f"  뱅크 정상    {args.bank_normal}장  = normals[0:{args.bank_normal}]")
    print(f"  홀드아웃     정상 {args.holdout_normal}장 = normals[{args.bank_normal}:"
          f"{args.bank_normal + args.holdout_normal}]  (뱅크와 겹치지 않음)")
    print(f"               결함 {args.holdout_defect}장 = anomalies[0:{args.holdout_defect}]")
    print(f"  오염         0장 (깨끗한 뱅크)")
    print(f"  장치         {embedder.device}")
    print()

    if args.cross:
        bank_cat, query_cat = args.cross.split(":")
        bank_normal, own_holdout, _ = split(root, bank_cat, args)
        _, other_normal, _ = split(root, query_cat, args)
        bank = build(bank_cat, bank_normal, embedder, args)
        print(f"{bank_cat} 뱅크 {bank.embeddings.shape[0]:,}행 으로 두 무리를 잰다\n")

        own = sorted(r.score for r in score_images(own_holdout, bank, embedder, root=root))
        oth = sorted(r.score for r in score_images(other_normal, bank, embedder, root=root))
        med_own = own[len(own) // 2]
        med_oth = oth[len(oth) // 2]
        print(f"{'무리':<28} {'장수':>4} {'최소':>8} {'중앙':>8} {'최대':>8}")
        print("-" * 62)
        print(f"{bank_cat + ' 정상 (같은 카테고리)':<28} {len(own):>4} {own[0]:>8.3f} {med_own:>8.3f} {own[-1]:>8.3f}")
        print(f"{query_cat + ' 정상 (다른 카테고리)':<28} {len(oth):>4} {oth[0]:>8.3f} {med_oth:>8.3f} {oth[-1]:>8.3f}")
        curve = sweep_thresholds(own, oth)  # 다른 카테고리를 '불량' 자리에 놓고 분리도를 본다
        print(f"\n두 무리의 분리도 AUROC {curve.auroc():.3f}")
        print(f"겹침 없음 여부: {query_cat} 최소 {oth[0]:.3f} vs {bank_cat} 최대 {own[-1]:.3f}"
              f" → {'완전히 갈림' if oth[0] > own[-1] else '겹침 있음'}")
        print("\nAUROC 1.000 이고 겹침이 없으면, 같은 보드라도 뱅크를 따로 세워야 한다는 뜻이다.")
        return 0

    print(f"{'카테고리':<10} {'AUROC':>7} {'뱅크행':>9} {'정상 중앙':>10} {'결함 중앙':>10} {'겹침':>6}")
    print("-" * 60)
    for cat in args.categories:
        bank_normal, holdout_normal, holdout_defect = split(root, cat, args)
        bank = build(cat, bank_normal, embedder, args)
        ns = sorted(r.score for r in score_images(holdout_normal, bank, embedder, root=root))
        ds = sorted(r.score for r in score_images(holdout_defect, bank, embedder, root=root))
        curve = sweep_thresholds(ns, ds)
        overlap = "있음" if ds[0] <= ns[-1] else "없음"
        print(f"{cat:<10} {curve.auroc():>7.3f} {bank.embeddings.shape[0]:>9,} "
              f"{ns[len(ns)//2]:>10.3f} {ds[len(ds)//2]:>10.3f} {overlap:>6}")
    print("\n같은 조건에서만 나란히 놓을 수 있다. 위 '측정 조건' 을 결과와 함께 인용할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
