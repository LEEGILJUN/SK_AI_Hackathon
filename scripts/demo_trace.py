"""뱅크 오염 역추적 데모.

VisA 가 준비되기 전에도 추론과 역추적이 실제로 도는지 눈으로 확인하려고
만든 스크립트다. 합성 이미지로 아래 세 장면을 순서대로 보여준다.

  1. 깨끗한 뱅크에서는 결함이 높은 점수를 받는다
  2. 정상셋에 결함이 섞이면 같은 결함의 점수가 떨어진다 (미검출)
  3. 최근접 패치를 되짚으면 섞여 들어간 이미지가 지목된다

실행:
    .venv/bin/python scripts/demo_trace.py

가상 공장 데이터가 준비되면 --normal-dir / --defect-dir 로 실제 이미지를
가리켜 같은 흐름을 그대로 돌릴 수 있다.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from inspection import (  # noqa: E402
    FeatureConfig,
    PatchEmbedder,
    build_bank,
    describe,
    score_image,
)


def collect(directory: Path) -> list[Path]:
    images = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    if not images:
        raise SystemExit(f"이미지를 찾지 못했다: {directory}")
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="뱅크 오염 역추적 데모")
    parser.add_argument("--normal-dir", type=Path, help="정상 이미지 폴더 (없으면 합성 데이터 사용)")
    parser.add_argument("--defect-dir", type=Path, help="결함 이미지 폴더")
    parser.add_argument("--backbone", default=None, help="백본. 기본은 합성 resnet18 / 실데이터 wide_resnet50_2")
    parser.add_argument("--contaminants", type=int, default=2, help="뱅크에 섞을 결함 장수")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    synthetic = args.normal_dir is None
    if synthetic:
        sys.path.insert(0, str(REPO_ROOT))
        from tests.synthetic import write_set

        root = Path(tempfile.mkdtemp(prefix="demo_factory_"))
        normal = write_set(root / "normal", 12, "normal", seed_offset=0)
        defect = write_set(root / "defect", 4, "defect", seed_offset=500)
        config = FeatureConfig(backbone=args.backbone or "resnet18", resize=64, crop=64)
        print(f"합성 데이터 사용 (실데이터는 --normal-dir 로 지정)\n  위치: {root}")
    else:
        if args.defect_dir is None:
            raise SystemExit("--normal-dir 을 주면 --defect-dir 도 주어야 한다.")
        root = Path.cwd()
        normal = collect(args.normal_dir)
        defect = collect(args.defect_dir)
        config = FeatureConfig(backbone=args.backbone or "wide_resnet50_2")

    if len(defect) < args.contaminants + 1:
        raise SystemExit(f"결함 이미지가 부족하다. 최소 {args.contaminants + 1}장 필요.")

    embedder = PatchEmbedder(config)
    print(f"  장치: {describe(embedder.device)} | 백본: {config.backbone}")

    contaminants = defect[: args.contaminants]
    query = defect[args.contaminants]

    shared = dict(coreset_ratio=0.25, seed=args.seed, root=root)
    print("\n뱅크를 만드는 중...")
    clean = build_bank(normal, embedder, bank_version="clean-v1", **shared)
    dirty = build_bank(
        list(normal) + list(contaminants), embedder, bank_version="contaminated-v2", **shared
    )
    print(f"  깨끗한 뱅크   {len(clean):>6}행   정상 {len(normal)}장")
    print(f"  오염된 뱅크   {len(dirty):>6}행   정상 {len(normal)}장 + 결함 {len(contaminants)}장 혼입")

    clean_result = score_image(query, clean, embedder, root=root)
    dirty_result = score_image(query, dirty, embedder, root=root)

    print(f"\n질의 이미지: {clean_result.image}   (뱅크에 넣지 않은 별개의 결함)")
    print(f"  깨끗한 뱅크   점수 {clean_result.score:.4f}")
    print(f"  오염된 뱅크   점수 {dirty_result.score:.4f}", end="")
    if dirty_result.score < clean_result.score:
        drop = (1 - dirty_result.score / clean_result.score) * 100
        print(f"   ({drop:.1f}% 하락 — 놓치기 시작하는 구간)")
    else:
        print("   (하락하지 않음)")

    top = dirty_result.top_match
    if top is None:
        raise SystemExit("최근접 패치를 찾지 못했다.")

    print("\n최근접 패치 역추적")
    print(f"  질의  {top.query.source_image}  격자({top.query.row},{top.query.col})")
    print(f"    ↓ 거리 {top.distance:.4f}  (뱅크 행 {top.bank_row_index})")
    print(f"  뱅크  {top.bank.source_image}  격자({top.bank.row},{top.bank.col})")

    contaminant_names = set()
    for path in contaminants:
        try:
            contaminant_names.add(Path(path).relative_to(root).as_posix())
        except ValueError:
            contaminant_names.add(Path(path).as_posix())

    if top.bank.source_image in contaminant_names:
        print("\n  → 지목된 것이 섞여 들어간 결함 이미지다. 원인은 뱅크 오염.")
        print("     다음 단계: 이 패치가 결함인지 진짜 정상품인지 판독(판별 항목 5번).")
        print("     결함이면 오염 제거 후 재구성, 진짜 정상품이면 정상 분포 중첩이라 재구성은 답이 아니다.")
    else:
        print("\n  → 지목된 것이 원래 정상 이미지다. 오염이 아닐 수 있다.")


if __name__ == "__main__":
    main()
