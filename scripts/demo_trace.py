"""뱅크 오염 진단 데모 — 역추적과 임계값 스윕.

VisA 가 준비되기 전에도 추론·역추적·스윕이 실제로 도는지 눈으로 확인하려고
만든 스크립트다. 합성 이미지로 아래를 순서대로 보여준다.

  1. 깨끗한 뱅크로 임계값을 잡는다
  2. 정상셋에 결함이 섞이면 같은 임계값에서 전건 미검출이 된다
  3. 임계값 스윕 — 다시 잡으면 잡히는가?
  4. 최근접 패치 역추적 — 왜 점수가 떨어졌는가?

3번과 4번을 함께 보는 것이 요점이다. 스윕만 보면 "임계값을 다시 잡으면
된다"로 읽히지만, 그건 증상을 덮는 것이다. 점수가 떨어진 이유가 뱅크에
섞인 결함이라는 사실은 역추적에서만 나온다.

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
    assess_threshold_feasibility,
    build_bank,
    describe,
    format_curve,
    score_image,
    score_images,
    sweep_from_results,
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
        normal = write_set(root / "normal", 18, "normal", seed_offset=0)
        defect = write_set(root / "defect", 8, "defect", seed_offset=500)
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

    # 정상 일부는 뱅크에 넣지 않고 남긴다. 임계값 스윕에서 과검률을 재려면
    # 뱅크 구성에 쓰이지 않은 양품이 있어야 한다.
    holdout_size = max(1, len(normal) // 4)
    bank_normal = normal[:-holdout_size]
    holdout_normal = normal[-holdout_size:]

    shared = dict(coreset_ratio=0.25, seed=args.seed, root=root)
    print("\n뱅크를 만드는 중...")
    clean = build_bank(bank_normal, embedder, bank_version="clean-v1", **shared)
    dirty = build_bank(
        list(bank_normal) + list(contaminants), embedder, bank_version="contaminated-v2", **shared
    )
    print(f"  깨끗한 뱅크   {len(clean):>6}행   정상 {len(bank_normal)}장")
    print(
        f"  오염된 뱅크   {len(dirty):>6}행   정상 {len(bank_normal)}장 "
        f"+ 결함 {len(contaminants)}장 혼입"
    )

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

    # ── 임계값 스윕 ──────────────────────────────────────────────────
    # 뱅크에 쓰지 않은 양품과 불량이 있어야 검출률·과검률을 잴 수 있다.
    holdout_normal = [p for p in normal if p not in set(bank_normal)]
    holdout_defect = [p for p in defect if p not in set(contaminants)]
    if not holdout_normal or not holdout_defect:
        print("\n(홀드아웃이 부족해 임계값 스윕은 건너뛴다.)")
        return

    print("\n" + "=" * 68)
    print(f"임계값 스윕   홀드아웃 양품 {len(holdout_normal)}장 / 불량 {len(holdout_defect)}장")
    print("=" * 68)

    for label, bank in (("깨끗한 뱅크", clean), ("오염된 뱅크", dirty)):
        normals = score_images(holdout_normal, bank, embedder, root=root)
        defects = score_images(holdout_defect, bank, embedder, root=root)
        curve = sweep_from_results(normals, defects)
        verdict = assess_threshold_feasibility(curve, target_detection=1.0, max_acceptable_fpr=0.05)

        print(f"\n── {label} ──   AUROC {curve.auroc():.3f}")
        print(format_curve(curve, rows=5))
        print(f"  판정: {'임계값 조정으로 해결 가능' if verdict.achievable else '임계값 조정으로 해결 불가'}")
        print(f"  근거: {verdict.reason}")

        if label == "깨끗한 뱅크" and verdict.required_threshold is not None:
            operating = verdict.required_threshold
        elif verdict.required_threshold is not None:
            # 운영 임계값은 그대로인데 점수만 떨어진 상황을 보여준다.
            at_operating = curve.at_threshold(operating)
            print(
                f"\n  운영 임계값 {operating:.4f} 를 그대로 두면: "
                f"검출 {at_operating.detected}/{len(defects)}건, "
                f"미검출 {at_operating.missed}건 (검출률 {at_operating.detection_rate:.0%})"
            )

    print(
        "\n요점: 스윕만 보면 두 경우 모두 '임계값을 다시 잡으면 된다'로 읽힌다.\n"
        "      점수가 떨어진 이유가 뱅크에 섞인 결함이라는 사실은 역추적에서만 나온다.\n"
        "      임계값을 다시 잡으면 증상은 사라지지만 오염은 그대로 남는다."
    )


if __name__ == "__main__":
    main()
