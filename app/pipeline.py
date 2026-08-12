"""데모 파이프라인 — 이슈 접수부터 승인 요청까지 한 번에.

웹 화면과 CLI 데모가 같은 코드를 쓰도록 여기 모았다. 두 벌로 갈라지면
시연 직전에 한쪽만 고쳐 놓고 다른 쪽이 깨지는 일이 생긴다.

시연용 데이터는 합성 이미지가 기본이다. VisA 실데이터는 뱅크 구성에 수십 초가
걸려 화면이 멈춘 것처럼 보이므로, 웹에서는 빠른 쪽을 기본으로 두고 필요하면
바꿀 수 있게 했다.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents.adapters import ModelAdapter, build_adapters
from agents.curate import CurationPlan, plan_curation
from agents.diagnose import DiagnosisResult, collect_evidence, decide
from agents.gate import GateResult, ReproducibilityResult, check_reproducibility, evaluate_gate
from agents.intake import IntakeResult, receive
from agents.rebuild import DirectoryImageSource, RebuildResult, execute_rebuild
from agents.release import ReleasePackage, prepare_release
from agents.vision import VisionJudgment, judge_defect_visible
from inspection import (
    FeatureConfig,
    MemoryBank,
    PatchEmbedder,
    build_bank,
    score_image,
    score_images,
    shadow_compare,
    sweep_from_results,
)
from inspection.quality import assess_quality
from inspection.shadow import ShadowReport
from lookup import MockLookup

DEMO_CONFIG = FeatureConfig(backbone="resnet18", resize=64, crop=64)
DEFAULT_ISSUE = "2라인 캡슐 표면 찍힘이 며칠째 계속 빠집니다. 육안으로는 명확한데 검사에서 양품으로 나옵니다."


@dataclass
class Stage:
    """화면에 한 칸으로 표시되는 단계."""

    key: str
    title: str
    status: str            # done | blocked | skipped | pending
    headline: str = ""
    detail: str = ""
    rows: list[tuple[str, str]] = field(default_factory=list)
    note: str = ""


@dataclass
class RunOutcome:
    """파이프라인 한 번의 결과 전체."""

    issue_text: str
    stages: list[Stage] = field(default_factory=list)
    intake: IntakeResult | None = None
    diagnosis: DiagnosisResult | None = None
    plan: CurationPlan | None = None
    rebuild: RebuildResult | None = None
    gate: GateResult | None = None
    shadow: ShadowReport | None = None
    reproducibility: ReproducibilityResult | None = None
    package: ReleasePackage | None = None
    approval_markdown: str = ""
    patch_override: str | None = None

    @property
    def finished(self) -> bool:
        return self.package is not None


class DemoFactory:
    """합성 이미지로 만든 가상 공장. 한 번 만들어 두고 재사용한다."""

    def __init__(self, normal_count: int = 16, defect_count: int = 6, contaminants: int = 2):
        from tests.synthetic import write_set

        self.root = Path(tempfile.mkdtemp(prefix="shvo_demo_"))
        self.normal = write_set(self.root / "normal", normal_count, "normal", seed_offset=0)
        self.defect = write_set(self.root / "defect", defect_count, "defect", seed_offset=500)
        self.contaminants = self.defect[:contaminants]
        self.query = self.defect[contaminants]
        self.holdout_normal = self.normal[-4:]
        self.holdout_defect = self.defect[contaminants + 1 :]
        self.bank_normal = self.normal[:-4]

        self.embedder = PatchEmbedder(DEMO_CONFIG)
        self.bank: MemoryBank = build_bank(
            list(self.bank_normal) + list(self.contaminants),
            self.embedder,
            coreset_ratio=0.25,
            seed=42,
            bank_version="v3",
            root=self.root,
        )
        self.clean_reference: MemoryBank = build_bank(
            self.bank_normal,
            self.embedder,
            coreset_ratio=0.25,
            seed=42,
            bank_version="v3-clean",
            root=self.root,
        )

    def relative(self, path: Path) -> str:
        return Path(path).relative_to(self.root).as_posix()

    @property
    def contaminant_names(self) -> set[str]:
        return {self.relative(p) for p in self.contaminants}


def run_pipeline(
    factory: DemoFactory,
    issue_text: str = DEFAULT_ISSUE,
    patch_override: str | None = "defect",
    adapters: tuple[ModelAdapter, ModelAdapter] | None = None,
    threshold: float = 2.20,
) -> RunOutcome:
    """이슈 한 건을 접수부터 승인 요청까지 돌린다.

    patch_override
        판별 5번(뱅크 패치가 결함인가)을 손으로 지정한다. 시각 언어 모델이
        아직 붙지 않았으므로 시연에서는 이 값을 바꿔 가며 판정이 갈리는 것을
        보여준다. None 이면 모델에 물어보고, 모델이 없으면 판정이 보류된다.
    """
    llm, vlm = adapters or build_adapters()
    lookup = MockLookup(threshold=threshold)
    outcome = RunOutcome(issue_text=issue_text, patch_override=patch_override)

    # ── 1. 인테이크 ─────────────────────────────────────────────────
    intake = receive(
        issue_text,
        llm,
        lookup=lookup,
        known={"line": "line_02", "object_name": "capsules", "defect_type": "dent"},
        attachments=[str(factory.query)],
    )
    outcome.intake = intake
    outcome.stages.append(
        Stage(
            key="intake",
            title="1. 인테이크",
            status="done" if intake.verdict == "proceed" else "blocked",
            headline={"proceed": "진단으로 넘김", "need_more_info": "정보 부족 — 되물음",
                      "duplicate": "이미 해결된 사례 — 중단"}[intake.verdict],
            detail=intake.note,
            rows=[("라인", intake.report.line or "—"),
                  ("품목", intake.report.object_name or "—"),
                  ("결함 유형", intake.report.defect_type or "—"),
                  ("첨부", f"{len(intake.report.attachments)}장")],
            note=intake.question,
        )
    )
    if intake.verdict != "proceed":
        return outcome

    # ── 2. 판별 항목 ────────────────────────────────────────────────
    result = score_image(factory.query, factory.bank, factory.embedder, root=factory.root)
    baseline = lookup.get_quality_baseline("line_02", "capsules")
    quality = assess_quality([factory.query], baseline.stats, min_images=1)
    visible = judge_defect_visible(vlm, factory.query, reported_defect="표면 찍힘")

    patch_judgment: VisionJudgment | None = None
    if patch_override:
        patch_judgment = VisionJudgment(
            verdict=patch_override, confidence=0.95,
            reason=f"시연을 위해 지정한 값 ({patch_override})",
            model="manual", is_stub=False,
        )

    cell = (DEMO_CONFIG.crop / result.grid_h) * (DEMO_CONFIG.crop / result.grid_w)
    hot = sum(1 for row in result.patch_distances for v in row if v >= threshold * 0.8)

    evidence = collect_evidence(
        defect_visible=visible,
        quality=quality,
        inference=result,
        threshold=lookup.get_threshold("line_02", "capsules", factory.bank.version),
        patch_judgment=patch_judgment,
        bank_profile=lookup.get_bank_profile(factory.bank.version),
        conditions={"date": "2026-06-01"},
        criteria=lookup.get_criteria("line_02", "capsules", "dent"),
        defect_area=hot * cell,
    )
    outcome.stages.append(
        Stage(
            key="evidence",
            title="2. 판별 7항목",
            status="done",
            headline=f"{sum(1 for e in evidence if e.usable)}/7 확인",
            rows=[
                (f"{e.item_no}. {e.name}", f"{'○' if e.usable else '×'}  {e.value}")
                for e in evidence
            ],
            note="시각 언어 모델을 쓰는 것은 1번과 5번뿐입니다. 나머지는 조회와 계산입니다.",
        )
    )

    # ── 3. 진단 ─────────────────────────────────────────────────────
    diagnosis = decide(evidence, similar_issues=intake.similar)
    outcome.diagnosis = diagnosis
    outcome.stages.append(
        Stage(
            key="diagnose",
            title="3. 진단",
            status="done" if diagnosis.cause else "blocked",
            headline=diagnosis.cause_label,
            detail=diagnosis.reasoning or diagnosis.blocking_reason,
            rows=[
                ("뱅크 재구성 필요", {True: "예", False: "아니오", None: "—"}[diagnosis.requires_bank_rebuild]),
                ("확신도", diagnosis.confidence),
                ("사람 확인", "필요" if diagnosis.needs_human else "불필요"),
                ("권고 조치", ", ".join(diagnosis.recommended_actions) or "—"),
                ("금지 조치", ", ".join(diagnosis.forbidden_actions) or "—"),
            ],
        )
    )

    # ── 4. 큐레이션 ─────────────────────────────────────────────────
    missed = score_images(factory.holdout_defect, factory.bank, factory.embedder, root=factory.root)
    plan = plan_curation(diagnosis, bank=factory.bank, missed_results=missed)
    outcome.plan = plan
    outcome.stages.append(
        Stage(
            key="curate",
            title="4. 데이터 큐레이션",
            status="done" if plan.touches_bank else "skipped",
            headline=plan.summary(),
            detail=plan.reason,
            rows=[(c.image, f"근거 {c.evidence_count}갈래 · {c.reason}") for c in plan.remove[:6]]
            or [(a.condition_key + "=" + a.condition_value, a.reason) for a in plan.add[:6]],
            note="; ".join(plan.alternative_actions),
        )
    )
    if not plan.touches_bank:
        return outcome

    # ── 5. 재구성 ───────────────────────────────────────────────────
    rebuild = execute_rebuild(
        plan, factory.bank, DirectoryImageSource(factory.root), factory.embedder,
        triggered_by=issue_text,
    )
    outcome.rebuild = rebuild
    outcome.stages.append(
        Stage(
            key="rebuild",
            title="5. 뱅크 재구성",
            status="done" if rebuild.executed else "blocked",
            headline=(f"{rebuild.record.from_version} → {rebuild.record.to_version}"
                      if rebuild.record else "실행 안 함"),
            detail=rebuild.reason,
            rows=[("제거", ", ".join(rebuild.record.removed) or "—"),
                  ("유지", f"{rebuild.record.kept_count}장"),
                  ("새 뱅크", f"{len(rebuild.bank)}행")] if rebuild.record else [],
            note="새 뱅크는 후보일 뿐 실제 판정에 쓰이지 않습니다.",
        )
    )
    if not rebuild.executed or rebuild.bank is None:
        return outcome

    # ── 6. 평가 게이트 ──────────────────────────────────────────────
    normals = score_images(factory.holdout_normal, rebuild.bank, factory.embedder, root=factory.root)
    defects = score_images(factory.holdout_defect, rebuild.bank, factory.embedder, root=factory.root)
    curve = sweep_from_results(normals, defects)
    new_threshold = (curve.threshold_for_detection(1.0) or curve.points[0]).threshold

    before = sweep_from_results(
        score_images(factory.holdout_normal, factory.bank, factory.embedder, root=factory.root),
        score_images(factory.holdout_defect, factory.bank, factory.embedder, root=factory.root),
    )
    gate = evaluate_gate(
        [r.score for r in normals], [r.score for r in defects],
        threshold=new_threshold, baseline_curve=before,
        candidate_version=rebuild.bank.version,
    )
    outcome.gate = gate
    outcome.stages.append(
        Stage(
            key="gate",
            title="6. 평가 게이트",
            status="done" if gate.passed else "blocked",
            headline="통과" if gate.passed else "미통과",
            detail=gate.reason,
            rows=[(c.name, f"{c.value} (기준 {c.threshold}) {'통과' if c.passed else '미달'}")
                  for c in gate.checks],
        )
    )

    # ── 7. 섀도 비교 ────────────────────────────────────────────────
    shadow = shadow_compare(
        list(factory.holdout_normal) + list(factory.holdout_defect),
        factory.bank, rebuild.bank,
        current_threshold=threshold, candidate_threshold=new_threshold,
        embedder=factory.embedder, root=factory.root,
    )
    outcome.shadow = shadow
    outcome.stages.append(
        Stage(
            key="shadow",
            title="7. 섀도 비교",
            status="done",
            headline=f"{shadow.total}장 중 {shadow.review_count}장만 확인",
            detail=shadow.summary(),
            rows=[(d.image, f"{d.current_verdict} → {d.candidate_verdict} "
                            f"({d.current_score:.3f} → {d.candidate_score:.3f})")
                  for d in shadow.disagreements[:8]],
            note="신규 뱅크를 실제 판정에 쓰지 않고 병렬로만 돌린 결과입니다.",
        )
    )

    # ── 8. 재현성 ───────────────────────────────────────────────────
    reproducibility = check_reproducibility(
        lambda: decide(evidence).to_dict()["cause"], runs=10
    )
    outcome.reproducibility = reproducibility

    # ── 9. 릴리즈 ───────────────────────────────────────────────────
    package = prepare_release(
        factory.root / "release" / rebuild.bank.version,
        bank=rebuild.bank, record=rebuild.record, diagnosis=diagnosis, plan=plan,
        gate=gate, shadow=shadow, reproducibility=reproducibility, issue_text=issue_text,
    )
    outcome.package = package
    outcome.approval_markdown = package.approval_document.read_text(encoding="utf-8")
    outcome.stages.append(
        Stage(
            key="release",
            title="8. 승인 요청",
            status="done",
            headline="배포 대기 — 자동 반영 없음",
            detail=f"승인 요청 문서를 생성했습니다. 재현성 {reproducibility.runs}회 "
                   f"{'일치' if reproducibility.identical else '불일치'}.",
            rows=[("배포 승인", "아니오 (사람이 결정)")]
                 + [("승인 보류 사유", r) for r in package.blocking_reasons],
            note="릴리즈 에이전트는 배포 패키지와 승인 요청까지만 만듭니다.",
        )
    )
    return outcome
