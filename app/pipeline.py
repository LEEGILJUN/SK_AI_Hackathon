"""데모 파이프라인 — 이슈 접수부터 승인 요청까지, 도구 호출로.

웹 화면과 CLI 데모가 같은 코드를 쓰도록 여기 모았다. 두 벌로 갈라지면
시연 직전에 한쪽만 고쳐 놓고 다른 쪽이 깨지는 일이 생긴다.

── 왜 함수를 순서대로 부르지 않는가 ─────────────────────────────────────

예전 구현은 receive → decide → plan → rebuild 를 파이썬 코드가 순서대로
불렀다. 그러면 화면에 보이는 것은 "에이전트가 판단하는 것"이 아니라
"파이프라인이 도는 것"이다. 순서가 코드에 박혀 있으면 원인이 무엇이든 같은
길을 간다.

지금은 모든 단계가 `agents.tools.ToolRegistry` 의 도구다. 언어 모델은
**어떤 도구를 어떤 순서로 부를지**만 정하고, 판정 내용은 도구 안의 규칙이
낸다. 이 분리가 깨지면 진단이 인상 평가가 된다.

── 모델이 없을 때 ───────────────────────────────────────────────────────

언어 모델이 안 붙어 있으면 `run_agent` 는 도구를 하나도 부르지 않고 멈춘다
(스텁이 그럴듯한 답을 지어내지 않게 만들어 두었기 때문이다). 그때는 **같은
도구들을 고정 순서로 재생**한다. 실행 경로는 한 벌이고 "누가 순서를
정했는가"만 달라지며, 화면에 어느 쪽이었는지 표시한다. 이것을 숨기면
시연에서 모델이 판단한 것처럼 보인다.

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
from agents.tools import (
    CHECKS_SPEC,
    DIAGNOSE_SPEC,
    GATE_SPEC,
    INTAKE_SPEC,
    PLAN_SPEC,
    REBUILD_SPEC,
    RELEASE_SPEC,
    SHADOW_SPEC,
    AgentRun,
    Tool,
    ToolCall,
    ToolRegistry,
    run_agent,
)
from agents.vision import VisionJudgment, judge_bank_patch, judge_defect_visible
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
from inspection.crop import crop_patch, crop_with_context
from inspection.quality import assess_quality
from inspection.shadow import ShadowReport
from lookup import MockLookup

DEMO_CONFIG = FeatureConfig(backbone="resnet18", resize=64, crop=64)
DEFAULT_ISSUE = "2라인 캡슐 표면 찍힘이 며칠째 계속 빠집니다. 육안으로는 명확한데 검사에서 양품으로 나옵니다."

#: 웹 양식이 아무것도 안 주면 쓰는 값. 인테이크가 추측으로 채우지 않게 하려면
#: 사람이 고른 값이 있어야 한다. **코드에 박힌 정답이 아니라 양식의 기본값이다.**
DEFAULT_CONTEXT = {"line": "line_02", "object_name": "capsules", "defect_type": "dent"}

#: 모델이 없을 때 재생할 고정 순서. 언어 모델이 붙으면 이 순서를 모델이 정한다.
FALLBACK_SEQUENCE: list[tuple[str, dict[str, Any]]] = [
    ("intake_issue", {}),
    ("run_checks", {}),
    ("diagnose_issue", {}),
    ("plan_curation", {}),
    ("rebuild_bank", {"confirm": True}),
    ("evaluate_gate", {}),
    ("shadow_compare", {}),
    ("prepare_release", {}),
]

AGENT_PROMPT = """현장에서 미검출 이슈가 하나 접수됐다. 접수부터 승인 요청까지 진행하라.

이슈 원문:
{issue}

먼저 접수하고, 판별 항목을 모은 뒤 진단하라. 원인이 뱅크 재구성으로 풀리는
것이 아니면 재구성을 부르지 말고 거기서 멈춰라."""


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
    #: 도구 순서를 누가 정했는가. "model" 이면 언어 모델, "fallback" 이면 고정 순서.
    driver: str = "fallback"
    driver_note: str = ""
    agent_run: AgentRun | None = None

    @property
    def finished(self) -> bool:
        return self.package is not None

    @property
    def tool_trace(self) -> list[tuple[str, str]]:
        """무엇이 어떤 순서로 불렸는가. 화면에 그대로 띄운다."""
        if self.agent_run is None:
            return []
        return [
            (r.name, "성공" if r.ok else f"실패 — {r.error}")
            for r in self.agent_run.tool_results
        ]


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


class _DemoSession:
    """도구들이 함께 쓰는 상태.

    도구는 순수 함수가 아니다. 앞 단계의 결과를 뒤 단계가 읽어야 하고, 언어
    모델이 순서를 잘못 잡으면 "먼저 X 를 해야 한다"고 되돌려 줘야 한다.
    그 상태를 여기 모아 둔다.
    """

    def __init__(
        self,
        factory: DemoFactory,
        issue_text: str,
        context: dict[str, Any],
        patch_override: str | None,
        adapters: tuple[ModelAdapter, ModelAdapter],
        threshold: float,
    ):
        self.factory = factory
        self.issue_text = issue_text
        self.context = context
        self.patch_override = patch_override
        self.llm, self.vlm = adapters
        self.threshold = threshold
        self.lookup = MockLookup(threshold=threshold)
        self.outcome = RunOutcome(issue_text=issue_text, patch_override=patch_override)

        self.evidence: list[Any] | None = None
        self.inference = None
        self.new_threshold: float | None = None
        self.baseline_curve = None

    # ── 도구 1. 인테이크 ────────────────────────────────────────────────

    def intake_issue(self, line: str = "", object_name: str = "", defect_type: str = "") -> dict:
        """자연어 이슈를 구조화하고 진단으로 넘길지 판단한다."""
        f = self.factory
        known = dict(self.context)
        # 모델이 인자로 준 값이 있으면 그것을 쓴다. 없으면 양식에서 받은 값.
        for key, value in (("line", line), ("object_name", object_name), ("defect_type", defect_type)):
            if value:
                known[key] = value

        intake = receive(
            self.issue_text, self.llm, lookup=self.lookup,
            known=known, attachments=[str(f.query)],
        )
        self.outcome.intake = intake
        self.outcome.stages.append(
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
        return {
            "verdict": intake.verdict,
            "line": intake.report.line,
            "object_name": intake.report.object_name,
            "missing": intake.missing,
            "next": "run_checks" if intake.verdict == "proceed" else "중단. 사람에게 되물어야 한다.",
        }

    # ── 도구 2. 판별 7항목 ──────────────────────────────────────────────

    def run_checks(self) -> dict:
        """판별 항목 일곱 가지를 모은다. 진단의 입력이다."""
        intake = self.outcome.intake
        if intake is None:
            raise RuntimeError("먼저 intake_issue 를 불러야 한다.")
        if intake.verdict != "proceed":
            raise RuntimeError(f"인테이크가 진행하지 않기로 했다: {intake.verdict}")

        f = self.factory
        line = intake.report.line or self.context["line"]
        obj = intake.report.object_name or self.context["object_name"]

        result = score_image(f.query, f.bank, f.embedder, root=f.root)
        self.inference = result
        baseline = self.lookup.get_quality_baseline(line, obj)
        quality = assess_quality([f.query], baseline.stats, min_images=1)
        visible = judge_defect_visible(self.vlm, f.query, reported_defect=intake.report.defect_type or "표면 결함")

        patch_judgment = self._judge_nearest_patch(result)

        cell = (DEMO_CONFIG.crop / result.grid_h) * (DEMO_CONFIG.crop / result.grid_w)
        hot = sum(1 for row in result.patch_distances for v in row if v >= self.threshold * 0.8)

        self.evidence = collect_evidence(
            defect_visible=visible,
            quality=quality,
            inference=result,
            threshold=self.lookup.get_threshold(line, obj, f.bank.version),
            patch_judgment=patch_judgment,
            bank_profile=self.lookup.get_bank_profile(f.bank.version),
            conditions={"date": "2026-06-01"},
            criteria=self.lookup.get_criteria(line, obj, intake.report.defect_type or "dent"),
            defect_area=hot * cell,
        )
        self.outcome.stages.append(
            Stage(
                key="evidence",
                title="2. 판별 7항목",
                status="done",
                headline=f"{sum(1 for e in self.evidence if e.usable)}/7 확인",
                rows=[
                    (f"{e.item_no}. {e.name}", f"{'○' if e.usable else '×'}  {e.value}")
                    for e in self.evidence
                ],
                note="시각 언어 모델을 쓰는 것은 1번과 5번뿐입니다. 나머지는 조회와 계산입니다.",
            )
        )
        return {
            "usable": sum(1 for e in self.evidence if e.usable),
            "total": len(self.evidence),
            "items": [{"no": e.item_no, "name": e.name, "usable": e.usable, "value": str(e.value)}
                      for e in self.evidence],
            "next": "diagnose_issue",
        }

    def _judge_nearest_patch(self, result) -> VisionJudgment | None:
        """판별 5번 — 최근접 뱅크 패치가 결함인가 진짜 정상품인가.

        시연에서는 값을 손으로 지정해 조치가 정반대로 갈리는 것을 보여줄 수
        있다. 지정하지 않으면 **역추적이 가리킨 자리를 실제로 잘라 모델에게
        묻는다.** 모델이 없으면 unknown 이 돌아오고 진단은 그 항목을 근거에서
        빼며, 그것이 옳은 동작이다 — 지어낸 답이 근거로 올라가면 안 된다.
        """
        if self.patch_override:
            return VisionJudgment(
                verdict=self.patch_override, confidence=0.95,
                reason=f"시연을 위해 지정한 값 ({self.patch_override})",
                model="manual", is_stub=False,
            )

        top = result.top_match
        if top is None:
            return None

        f = self.factory
        ref = top.bank
        grid = (result.grid_h, result.grid_w)
        source = f.root / ref.source_image
        if not source.exists():
            return None

        # 판별 1번 실험에서 정해진 값. 여유가 작으면 무엇을 보는지 모른다고 답한다.
        patch = crop_patch(source, ref, grid, f.embedder.config, margin=64, enlarge_to=512)
        context = crop_with_context(source, ref, grid, f.embedder.config, context_cells=2)
        return judge_bank_patch(self.vlm, patch, context_image=context)

    # ── 도구 3. 진단 ────────────────────────────────────────────────────

    def diagnose_issue(self, line: str = "", object_name: str = "") -> dict:
        """판별 항목으로 원인 6종 중 하나를 정한다. 판정은 규칙이 낸다."""
        if self.evidence is None:
            raise RuntimeError("먼저 run_checks 를 불러야 한다.")

        diagnosis = decide(self.evidence, similar_issues=self.outcome.intake.similar)
        self.outcome.diagnosis = diagnosis
        self.outcome.stages.append(
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
        return {
            "cause": diagnosis.cause,
            "cause_label": diagnosis.cause_label,
            "requires_bank_rebuild": diagnosis.requires_bank_rebuild,
            "reasoning": diagnosis.reasoning or diagnosis.blocking_reason,
            "forbidden_actions": diagnosis.forbidden_actions,
            "next": "plan_curation",
        }

    # ── 도구 4. 큐레이션 ────────────────────────────────────────────────

    def plan_curation(self) -> dict:
        """뱅크에서 무엇을 빼고 무엇을 채울지 정한다."""
        diagnosis = self.outcome.diagnosis
        if diagnosis is None:
            raise RuntimeError("먼저 diagnose_issue 를 불러야 한다.")

        f = self.factory
        missed = score_images(f.holdout_defect, f.bank, f.embedder, root=f.root)
        plan = plan_curation(diagnosis, bank=f.bank, missed_results=missed)
        self.outcome.plan = plan
        self.outcome.stages.append(
            Stage(
                key="curate",
                title="4. 데이터 큐레이션",
                status="done" if plan.touches_bank else "skipped",
                headline=plan.summary(),
                detail=plan.reason,
                rows=[(c.image, f"근거 {c.evidence_count}갈래 · 패치 {c.patch_count}개 · {c.reason}")
                      for c in plan.remove[:6]]
                or [(a.condition_key + "=" + a.condition_value, a.reason) for a in plan.add[:6]],
                note="; ".join(plan.alternative_actions),
            )
        )
        return {
            "touches_bank": plan.touches_bank,
            "summary": plan.summary(),
            "reason": plan.reason,
            "remove": [c.image for c in plan.remove],
            "coverage_cost": plan.coverage_cost.to_dict() if plan.coverage_cost else None,
            "next": "rebuild_bank" if plan.touches_bank else "여기서 멈춘다. 재구성은 답이 아니다.",
        }

    # ── 도구 5. 재구성 ──────────────────────────────────────────────────

    def rebuild_bank(self, confirm: bool = False) -> dict:
        """계획대로 새 뱅크를 만든다. 배포하지 않는다."""
        plan = self.outcome.plan
        if plan is None:
            raise RuntimeError("먼저 plan_curation 을 불러야 한다.")
        if not confirm:
            return {"executed": False, "reason": "confirm=true 로 다시 불러야 실행한다."}

        f = self.factory
        rebuild = execute_rebuild(
            plan, f.bank, DirectoryImageSource(f.root), f.embedder,
            triggered_by=self.issue_text,
        )
        self.outcome.rebuild = rebuild
        self.outcome.stages.append(
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
        return {
            "executed": rebuild.executed,
            "reason": rebuild.reason,
            "next": "evaluate_gate" if rebuild.executed else "중단",
        }

    # ── 도구 6. 평가 게이트 ─────────────────────────────────────────────

    def evaluate_gate(self) -> dict:
        """새 뱅크가 배포 후보가 될 만한지 본다."""
        rebuild = self.outcome.rebuild
        if rebuild is None or not rebuild.executed or rebuild.bank is None:
            raise RuntimeError("먼저 rebuild_bank 를 실행해야 한다.")

        f = self.factory
        normals = score_images(f.holdout_normal, rebuild.bank, f.embedder, root=f.root)
        defects = score_images(f.holdout_defect, rebuild.bank, f.embedder, root=f.root)
        curve = sweep_from_results(normals, defects)
        self.new_threshold = (curve.threshold_for_detection(1.0) or curve.points[0]).threshold

        self.baseline_curve = sweep_from_results(
            score_images(f.holdout_normal, f.bank, f.embedder, root=f.root),
            score_images(f.holdout_defect, f.bank, f.embedder, root=f.root),
        )
        gate = evaluate_gate(
            [r.score for r in normals], [r.score for r in defects],
            threshold=self.new_threshold, baseline_curve=self.baseline_curve,
            candidate_version=rebuild.bank.version,
        )
        self.outcome.gate = gate
        self.outcome.stages.append(
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
        return {"passed": gate.passed, "reason": gate.reason, "next": "shadow_compare"}

    # ── 도구 7. 섀도 비교 ───────────────────────────────────────────────

    def shadow_compare(self) -> dict:
        """신규 뱅크를 판정에 쓰지 않고 병렬로만 돌려 갈리는 건만 뽑는다."""
        rebuild = self.outcome.rebuild
        if rebuild is None or rebuild.bank is None or self.new_threshold is None:
            raise RuntimeError("먼저 evaluate_gate 를 불러야 한다.")

        f = self.factory
        shadow = shadow_compare(
            list(f.holdout_normal) + list(f.holdout_defect),
            f.bank, rebuild.bank,
            current_threshold=self.threshold, candidate_threshold=self.new_threshold,
            embedder=f.embedder, root=f.root,
        )
        self.outcome.shadow = shadow
        self.outcome.stages.append(
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
        return {
            "total": shadow.total,
            "review_count": shadow.review_count,
            "summary": shadow.summary(),
            "next": "prepare_release",
        }

    # ── 도구 8. 승인 요청 ───────────────────────────────────────────────

    def prepare_release(self) -> dict:
        """배포 패키지와 승인 요청 문서를 만든다. 배포하지 않는다."""
        o = self.outcome
        if o.rebuild is None or o.rebuild.bank is None or o.gate is None or o.shadow is None:
            raise RuntimeError("재구성·게이트·섀도가 모두 끝나야 한다.")

        reproducibility = check_reproducibility(
            lambda: decide(self.evidence).to_dict()["cause"], runs=10
        )
        o.reproducibility = reproducibility

        package = prepare_release(
            self.factory.root / "release" / o.rebuild.bank.version,
            bank=o.rebuild.bank, record=o.rebuild.record, diagnosis=o.diagnosis, plan=o.plan,
            gate=o.gate, shadow=o.shadow, reproducibility=reproducibility,
            issue_text=self.issue_text,
        )
        o.package = package
        o.approval_markdown = package.approval_document.read_text(encoding="utf-8")
        o.stages.append(
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
        return {
            "created": True,
            "approved_for_deploy": False,
            "blocking_reasons": package.blocking_reasons,
            "next": "끝. 배포 여부는 사람이 정한다.",
        }

    # ── 등록 ────────────────────────────────────────────────────────────

    def registry(self) -> ToolRegistry:
        return ToolRegistry([
            Tool(INTAKE_SPEC, self.intake_issue),
            Tool(CHECKS_SPEC, self.run_checks),
            Tool(DIAGNOSE_SPEC, self.diagnose_issue),
            Tool(PLAN_SPEC, self.plan_curation),
            Tool(REBUILD_SPEC, self.rebuild_bank, mutates_bank=True),
            Tool(GATE_SPEC, self.evaluate_gate),
            Tool(SHADOW_SPEC, self.shadow_compare),
            Tool(RELEASE_SPEC, self.prepare_release),
        ])


def _replay_fixed_sequence(session: _DemoSession, registry: ToolRegistry) -> AgentRun:
    """모델이 없을 때 같은 도구들을 고정 순서로 부른다.

    도구가 앞 단계를 요구하며 실패하면 거기서 멈춘다. 실패를 삼키고 계속
    가면 화면에 빈 칸이 생기고 왜 비었는지 알 수 없게 된다.
    """
    run = AgentRun(prompt="(고정 순서 — 언어 모델 미연결)")
    for index, (name, arguments) in enumerate(FALLBACK_SEQUENCE, start=1):
        result = registry.execute(ToolCall(id=f"fixed-{index}", name=name, arguments=dict(arguments)))
        run.tool_results.append(result)
        run.steps = index
        if not result.ok:
            run.stopped_reason = f"{name} 에서 멈췄다 — {result.error}"
            return run
        # 도구가 "여기서 멈춘다"고 하면 따른다. 재구성이 답이 아닌 원인이 그렇다.
        nxt = result.output.get("next", "") if isinstance(result.output, dict) else ""
        if "멈춘다" in nxt or "중단" in nxt:
            run.stopped_reason = f"{name} 이후 진행하지 않았다 — {nxt}"
            return run
    run.stopped_reason = "고정 순서를 끝까지 실행했다."
    return run


def run_pipeline(
    factory: DemoFactory,
    issue_text: str = DEFAULT_ISSUE,
    patch_override: str | None = "defect",
    adapters: tuple[ModelAdapter, ModelAdapter] | None = None,
    threshold: float = 2.20,
    context: dict[str, Any] | None = None,
) -> RunOutcome:
    """이슈 한 건을 접수부터 승인 요청까지 돌린다.

    patch_override
        판별 5번(뱅크 패치가 결함인가)을 손으로 지정한다. 시연에서 이 값을
        바꿔 가며 같은 이미지·같은 점수에서 조치가 정반대로 갈리는 것을
        보여줄 수 있다. None 이면 역추적이 가리킨 자리를 잘라 시각 언어 모델에
        묻고, 모델이 없으면 판정이 보류된다.
    context
        웹 양식에서 받은 라인·품목·결함 유형. 인테이크가 추측으로 채우지
        않게 하려면 사람이 고른 값이 있어야 한다.

    언어 모델이 붙어 있으면 모델이 도구 순서를 정하고, 없으면 같은 도구들을
    고정 순서로 재생한다. 어느 쪽이었는지는 outcome.driver 에 남는다.
    """
    llm, vlm = adapters or build_adapters()
    session = _DemoSession(
        factory, issue_text, dict(context or DEFAULT_CONTEXT),
        patch_override, (llm, vlm), threshold,
    )
    registry = session.registry()

    run = run_agent(AGENT_PROMPT.format(issue=issue_text), llm, registry, max_steps=12)
    if run.tool_results:
        session.outcome.driver = "model"
        session.outcome.driver_note = (
            f"{llm.describe()} 가 도구 {len(run.tool_results)}개를 순서대로 불렀습니다."
        )
    else:
        run = _replay_fixed_sequence(session, registry)
        session.outcome.driver = "fallback"
        session.outcome.driver_note = (
            "언어 모델이 연결되지 않아 같은 도구들을 고정 순서로 실행했습니다. "
            "순서를 모델이 정한 것이 아닙니다."
        )

    session.outcome.agent_run = run
    return session.outcome
