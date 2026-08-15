"""엔드투엔드 데모 — 자연어 이슈에서 원인과 근거까지.

1주 차 완료 기준을 그대로 실행한다.

    "자연어 이슈를 입력하면 원인 분류 결과와 근거가 출력된다"

지금은 조회 계층이 목이고 시각 언어 모델이 스텁이다. 그래서 이 스크립트의
목적은 성능 확인이 아니라 **경로가 이어졌는지 확인**이다. 특히 근거가 모자랄
때 진단이 판정을 보류하는지를 본다.

--patch-judgment 로 판별 5번 값을 손으로 넣으면, 모델을 붙였을 때 어떤 판정이
나올지를 미리 볼 수 있다. 시각 언어 모델이 준비되면 이 인자를 빼면 된다.

실행:
    .venv/bin/python scripts/demo_diagnose.py
    .venv/bin/python scripts/demo_diagnose.py --patch-judgment defect
    .venv/bin/python scripts/demo_diagnose.py --patch-judgment normal --sweep-infeasible
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.adapters import build_adapters  # noqa: E402
from agents.diagnose import collect_evidence, decide, narrate  # noqa: E402
from agents.intake import receive  # noqa: E402
from agents.vision import VisionJudgment, judge_bank_patch, judge_defect_visible  # noqa: E402
from inspection import (  # noqa: E402
    FeatureConfig,
    PatchEmbedder,
    build_bank,
    crop_patch,
    crop_with_context,
    score_image,
)
from inspection.quality import assess_quality  # noqa: E402
from inspection.sweep import FeasibilityVerdict  # noqa: E402
from lookup import MockLookup, resolved_duplicate  # noqa: E402

ISSUE_TEXT = "2라인 캡슐 표면 찍힘이 며칠째 계속 빠집니다. 육안으로는 명확한데 검사에서 양품으로 나옵니다."

RULE = "─" * 72


def head(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="이슈 접수 → 진단 엔드투엔드")
    parser.add_argument(
        "--patch-judgment",
        choices=["defect", "genuine_normal", "unknown"],
        help="판별 5번을 손으로 지정한다. 시각 언어 모델이 없을 때 판정 갈림을 확인하는 용도",
    )
    parser.add_argument(
        "--sweep-infeasible", action="store_true", help="임계값으로 해결 불가한 상황을 가정"
    )
    parser.add_argument("--duplicate", action="store_true", help="이미 해결된 동일 사례가 있다고 가정")
    parser.add_argument("--threshold", type=float, default=2.20)
    args = parser.parse_args()

    llm, vlm = build_adapters()
    lookup = MockLookup(
        threshold=args.threshold,
        similar_issues=[resolved_duplicate()] if args.duplicate else [],
    )

    # ── 데이터 준비 (합성) ──────────────────────────────────────────
    from tests.synthetic import write_set

    root = Path(tempfile.mkdtemp(prefix="demo_diag_"))
    normal = write_set(root / "normal", 12, "normal", seed_offset=0)
    defect = write_set(root / "defect", 4, "defect", seed_offset=500)
    contaminants, query = defect[:2], defect[3]

    embedder = PatchEmbedder(FeatureConfig(backbone="resnet18", crop=64))
    bank = build_bank(
        list(normal) + list(contaminants),
        embedder,
        coreset_ratio=0.25,
        seed=42,
        bank_version="v3",
        root=root,
    )

    # ── 1. 인테이크 ─────────────────────────────────────────────────
    head("1. 인테이크 — 정보가 충분한가, 이미 해결된 사례인가")
    print(f"입력: {ISSUE_TEXT}")

    intake = receive(
        ISSUE_TEXT,
        llm,
        lookup=lookup,
        known={"line": "line_02", "object_name": "capsules", "defect_type": "dent"},
        attachments=[str(query)],
    )
    print(f"\n판정: {intake.verdict}")
    print(f"근거: {intake.note}")
    if intake.report.extracted_by:
        print(f"추출: {intake.report.extracted_by}")
    else:
        print("추출: 언어 모델 미연결 — 웹 양식 값을 사용")

    if intake.verdict != "proceed":
        if intake.question:
            print(f"\n되묻기: {intake.question}")
        if intake.duplicate_of:
            print(f"\n중복 이슈: {intake.duplicate_of} — 진단을 진행하지 않는다.")
        print("\n여기서 멈춘다.")
        return

    # ── 2. 판별 항목 수집 ───────────────────────────────────────────
    head("2. 판별 항목 수집")

    result = score_image(query, bank, embedder, root=root)
    threshold = lookup.get_threshold("line_02", "capsules", bank.version)
    baseline = lookup.get_quality_baseline("line_02", "capsules")
    profile = lookup.get_bank_profile(bank.version)
    criteria = lookup.get_criteria("line_02", "capsules", "dent")

    quality = assess_quality(
        [query], baseline.stats, tolerance_sigma=3.0, min_images=1
    )

    visible = judge_defect_visible(vlm, query, reported_defect="표면 찍힘")

    top = result.top_match
    patch_judgment = None
    if top is not None:
        if args.patch_judgment:
            patch_judgment = VisionJudgment(
                verdict=args.patch_judgment,
                confidence=0.95,
                reason=f"손으로 지정한 값 ({args.patch_judgment})",
                model="manual-override",
                is_stub=False,
            )
        else:
            patch_image = crop_patch(root / top.bank.source_image, top.bank, bank.grid)
            context = crop_with_context(root / top.bank.source_image, top.bank, bank.grid)
            patch_judgment = judge_bank_patch(vlm, patch_image, context)

    # 마스크 면적: 이상 맵에서 임계 초과 격자 수를 픽셀로 환산
    cell_area = (64 / result.grid_h) * (64 / result.grid_w)
    hot_cells = sum(1 for row in result.patch_distances for v in row if v >= threshold.value * 0.8)
    defect_area = hot_cells * cell_area

    sweep = None
    if args.sweep_infeasible:
        sweep = FeasibilityVerdict(
            achievable=False,
            target_detection=1.0,
            max_acceptable_fpr=0.05,
            required_threshold=0.62,
            resulting_fpr=0.71,
            resulting_detection=1.0,
            auroc=0.58,
            reason="검출률 100%를 달성하려면 과검률이 71.0%가 된다. 허용치를 크게 넘는다.",
        )

    evidence = collect_evidence(
        defect_visible=visible,
        quality=quality,
        inference=result,
        threshold=threshold,
        patch_judgment=patch_judgment,
        bank_profile=profile,
        condition_key="date",
        condition_value="2026-06-01",
        criteria=criteria,
        defect_area=defect_area,
    )

    for item in evidence:
        mark = "✓" if item.usable else "·"
        print(f"  {mark} {item.item_no}. {item.name:<26} {str(item.value):<12} {item.detail[:60]}")

    # ── 3. 진단 ─────────────────────────────────────────────────────
    head("3. 진단 — 원인 6종 중 무엇인가")

    diagnosis = decide(evidence, sweep=sweep, similar_issues=intake.similar)

    if diagnosis.cause:
        print(f"원인: {diagnosis.cause_label} ({diagnosis.cause})")
        print(f"뱅크 재구성 필요: {'예' if diagnosis.requires_bank_rebuild else '아니오'}")
        print(f"확신도: {diagnosis.confidence}   사람 확인: {'필요' if diagnosis.needs_human else '불필요'}")
        print(f"\n근거: {diagnosis.reasoning}")
        print(f"\n권고 조치: {', '.join(diagnosis.recommended_actions)}")
        if diagnosis.forbidden_actions:
            print(f"금지 조치: {', '.join(diagnosis.forbidden_actions)}")
    else:
        print("원인: 판정 보류")
        print(f"\n사유: {diagnosis.blocking_reason}")
        if diagnosis.candidate_causes:
            print(f"\n남은 후보: {', '.join(diagnosis.candidate_causes)}")
        if diagnosis.reasoning:
            print(f"확인 대상: {diagnosis.reasoning}")

    # ── 4. 서술 ─────────────────────────────────────────────────────
    head("4. 리포트 서술 (언어 모델)")
    print(narrate(diagnosis, llm))

    print(f"\n{RULE}")
    print("조회 계층은 목(lookup/mock.py), 데이터는 합성 이미지다. 성능 수치가 아니다.")


if __name__ == "__main__":
    main()
