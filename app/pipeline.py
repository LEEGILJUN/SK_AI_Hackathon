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
from datetime import date
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
    INSPECT_SPEC,
    INTAKE_SPEC,
    MES_SPEC,
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
from lookup.base import RETRIEVAL_KIND, DefectDistribution, ImageRecord

def _evidence_value(value: Any) -> str:
    """판별 항목 값을 사람이 읽는 한 줄로 만든다.

    4번 최근접 패치는 딕셔너리라 그대로 찍으면 파이썬 repr 이 화면에 나온다.
    **이 항목이 진단의 핵심 근거인데 화면에서 가장 안 읽히는 자리가 된다.**
    무엇을 어디서 얼마나 가까이 찾았는지만 남긴다. 원본은 진단 계층이 그대로
    들고 있으므로 여기서 줄여도 판정에는 영향이 없다.
    """
    if isinstance(value, dict) and "source_image" in value:
        parts = [Path(str(value["source_image"])).name]
        row, col = value.get("row"), value.get("col")
        if row is not None and col is not None:
            parts.append(f"격자 ({row},{col})")
        distance = value.get("distance")
        if isinstance(distance, (int, float)):
            parts.append(f"거리 {distance:.4f}")
        return " · ".join(parts)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


DEMO_CONFIG = FeatureConfig(backbone="resnet18", resize=64, crop=64)
DEFAULT_ISSUE = "2라인 캡슐 표면 찍힘이 며칠째 계속 빠집니다. 육안으로는 명확한데 검사에서 양품으로 나옵니다."


def default_issue(factory: "DemoFactory") -> str:
    """시연용 이슈 원문. **제품명을 본문에 넣는다.**

    현장 이슈는 이미지를 첨부해서 오기보다 "이 제품이 계속 빠진다"로 온다.
    제품명이 본문에 있어야 언어 모델이 뽑을 것이 생기고, MES 조회가 의미를
    가진다. 양식에 제품명 칸만 두고 본문에서 빼면 추출이 할 일이 없어진다.
    """
    return f"{DEFAULT_ISSUE} 제품 {factory.reported_product} 건입니다."

#: 모델이 없을 때 재생할 고정 순서. 언어 모델이 붙으면 이 순서를 모델이 정한다.
FALLBACK_SEQUENCE: list[tuple[str, dict[str, Any]]] = [
    ("intake_issue", {}),
    ("lookup_mes", {}),
    ("run_inspection", {}),
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

먼저 접수하고, MES 에서 해당 제품·로트의 이미지를 찾아 그 품목의 뱅크로
추론하라. 미검이 나오면 판별 항목을 모아 진단하라. 원인이 뱅크 재구성으로
풀리는 것이 아니면 재구성을 부르지 말고 거기서 멈춰라."""


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
    #: MES 조회·추론으로 가려낸 미검 건. 로트 집중도 집계의 재료다.
    missed_records: list[ImageRecord] = field(default_factory=list)
    distribution: DefectDistribution | None = None
    #: 진단 근거를 화면에 그리기 위한 것들. 전부 실제 추론 결과다.
    inference: Any = None                 # InferenceResult — 히트맵과 점수
    query_image: str = ""                 # 저장소 기준 상대 경로
    threshold: float = 0.0
    bank_version: str = ""
    grid: tuple[int, int] = (0, 0)
    #: 어떤 조회를 어떤 방식으로 했는가. lookup 이 남긴 실제 호출 기록이다.
    retrievals: list[dict[str, Any]] = field(default_factory=list)

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


#: 시연 공장의 품목 구성. (라인, 품목) → 합성 무늬 이름.
#:
#: **뱅크는 품목마다 따로 있습니다.** 캡슐의 정상 패치로 PCB 를 판정할 수
#: 없습니다. 품목이 하나뿐이면 그 사실이 코드에서 드러나지 않아 뱅크가
#: 하나뿐인 전제가 조용히 박힙니다.
DEMO_ITEMS: list[tuple[str, str, str]] = [
    ("line_02", "capsules", "capsules"),
    ("line_02", "pcb1", "pcb1"),
    ("line_03", "macaroni1", "macaroni1"),
]

#: 오염을 넣을 품목 하나. 나머지는 깨끗하다.
#: 전 품목을 오염시키면 "이 품목만 문제다"를 보여줄 수 없습니다.
CONTAMINATED_ITEM = ("line_02", "capsules")


@dataclass
class ItemLine:
    """품목 하나의 이미지와 뱅크.

    뱅크·홀드아웃·오염원이 품목 단위로 묶입니다. 이 묶음이 흐려지면 다른
    품목의 이미지를 다른 품목 뱅크로 재는 실수가 조용히 지나갑니다.
    """

    line: str
    object_name: str
    bank: MemoryBank
    bank_normal: list[Path]
    holdout_normal: list[Path]
    holdout_defect: list[Path]
    contaminants: list[Path]

    @property
    def key(self) -> tuple[str, str]:
        return (self.line, self.object_name)


class DemoFactory:
    """합성 이미지로 만든 가상 공장. 한 번 만들어 두고 재사용한다.

    품목이 여럿이고 **품목마다 뱅크가 따로** 있습니다. MES 가 아는 이미지
    목록(`catalog`)도 함께 만들어, 제품명·로트로 이미지를 찾는 경로가 실제로
    돌아갑니다. 이동현의 가상 공장이 오면 이 클래스가 빠집니다.
    """

    def __init__(self, normal_count: int = 16, defect_count: int = 6, contaminants: int = 2):
        from tests.synthetic import write_set

        self.root = Path(tempfile.mkdtemp(prefix="shvo_demo_"))
        self.embedder = PatchEmbedder(DEMO_CONFIG)
        self.items: dict[tuple[str, str], ItemLine] = {}
        self.catalog: list[ImageRecord] = []

        for index, (line, object_name, variant) in enumerate(DEMO_ITEMS):
            base = self.root / line / object_name
            normal = write_set(base / "normal", normal_count, "normal",
                               seed_offset=index * 1000, variant=variant)
            defect = write_set(base / "defect", defect_count, "defect",
                               seed_offset=index * 1000 + 500, variant=variant)

            dirty = (line, object_name) == CONTAMINATED_ITEM
            mixed_in = list(defect[:contaminants]) if dirty else []
            bank_normal = normal[:-4]

            bank = build_bank(
                list(bank_normal) + mixed_in,
                self.embedder,
                coreset_ratio=0.25,
                seed=42,
                bank_version=f"{object_name}-v3",
                root=self.root,
            )
            self.items[(line, object_name)] = ItemLine(
                line=line, object_name=object_name, bank=bank,
                bank_normal=list(bank_normal),
                holdout_normal=list(normal[-4:]),
                holdout_defect=list(defect[contaminants:]) if dirty else list(defect),
                contaminants=mixed_in,
            )
            self._register(line, object_name, normal, defect, index)

        contaminated = self.items[CONTAMINATED_ITEM]
        #: 이슈로 접수될 미검 제품. MES 조회로 찾아내는 대상이다.
        self.reported_product = self._product_id(
            CONTAMINATED_ITEM[0], CONTAMINATED_ITEM[1], contaminated.holdout_defect[0]
        )

    # ── MES 가 아는 것 ──────────────────────────────────────────────────

    def _product_id(self, line: str, object_name: str, path: Path) -> str:
        return f"{object_name.upper()}-{line[-2:]}-{Path(path).stem}"

    def _register(self, line: str, object_name: str, normal, defect, index: int) -> None:
        """이미지마다 MES 레코드를 만든다. 로트와 설비를 붙여 집계가 되게 한다."""
        lots = [f"LOT-2026060{index + 1}-{n:03d}" for n in (1, 2)]
        for group, kind in ((normal, "pass"), (defect, "defect")):
            for i, path in enumerate(group):
                #: 결함은 첫 로트에 몰아 넣는다. 로트 집중도가 화면에 뜨는지
                #: 확인하려면 실제로 몰려 있는 데이터가 있어야 한다.
                lot = lots[0] if kind == "defect" else lots[i % len(lots)]
                self.catalog.append(
                    ImageRecord(
                        product_id=self._product_id(line, object_name, path),
                        path=self.relative(path),
                        line=line,
                        object_name=object_name,
                        lot=lot,
                        captured_at=date(2026, 6, index + 1),
                        # 미검 상황을 만든다 — 실제로는 결함인데 설비가 양품이라 했다.
                        verdict="pass",
                        ground_truth=kind,
                        equipment=f"CAM-{line[-2:]}-{(i % 2) + 1}",
                    )
                )

    def bank_versions(self) -> dict[tuple[str, str], str]:
        return {key: item.bank.version for key, item in self.items.items()}

    def item_for(self, line: str, object_name: str) -> ItemLine | None:
        return self.items.get((line, object_name))

    def relative(self, path: Path) -> str:
        return Path(path).relative_to(self.root).as_posix()

    def resolve(self, relative_path: str) -> Path:
        return self.root / relative_path

    @property
    def contaminant_names(self) -> set[str]:
        return {self.relative(p)
                for item in self.items.values() for p in item.contaminants}


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
        self.lookup = MockLookup(
            threshold=threshold,
            catalog=factory.catalog,
            banks=factory.bank_versions(),
        )
        self.outcome = RunOutcome(issue_text=issue_text, patch_override=patch_override)

        self.evidence: list[Any] | None = None
        self.inference = None
        self.new_threshold: float | None = None
        self.baseline_curve = None
        #: MES 조회로 특정된 것들. 여기가 비면 뒤 단계가 돌지 않는다.
        self.item: ItemLine | None = None
        self.records: list[ImageRecord] = []
        self.query: Path | None = None

    # ── 도구 1. 인테이크 ────────────────────────────────────────────────

    def intake_issue(self, line: str = "", object_name: str = "",
                     defect_type: str = "", product_id: str = "", lot: str = "") -> dict:
        """자연어 이슈를 구조화하고 진단으로 넘길지 판단한다.

        **자연어가 주 입력이다.** 언어 모델이 이슈 원문에서 라인·품목·제품명을
        뽑고, 양식에서 받은 값은 모델이 못 뽑은 자리만 채운다. 양식이 다 채워져
        있으면 추출이 할 일이 없어져 언어 모델을 쓰는 의미가 사라진다.

        모델이 없으면 추출이 비고, 그때는 양식 값이 그대로 쓰인다. 그것도 없으면
        인테이크가 되묻는다 — 추측으로 채우면 엉뚱한 라인의 뱅크를 건드린다.
        """
        # 1) 언어 모델이 원문에서 먼저 뽑는다. known 은 주지 않는다.
        intake = receive(self.issue_text, self.llm, lookup=self.lookup)
        report = intake.report
        extracted = {k: getattr(report, k) for k in
                     ("line", "object_name", "defect_type", "product_id", "lot")}

        # 2) 모델이 못 뽑은 자리만 채운다. 도구 인자 → 양식 순.
        fallback = dict(self.context)
        for key, value in (("line", line), ("object_name", object_name),
                           ("defect_type", defect_type), ("product_id", product_id),
                           ("lot", lot)):
            if value:
                fallback[key] = value

        filled_by_form = []
        for key, value in fallback.items():
            if value and hasattr(report, key) and not getattr(report, key):
                setattr(report, key, value)
                filled_by_form.append(key)

        # 3) 채우고 나서 다시 충분성을 본다.
        intake = receive(self.issue_text, self.llm, lookup=self.lookup,
                         known={k: getattr(report, k) for k in extracted})
        self.outcome.intake = intake

        source = (
            f"언어 모델이 {sum(1 for v in extracted.values() if v)}개 항목을 원문에서 뽑았습니다."
            if any(extracted.values()) else
            "언어 모델이 연결되지 않아 원문에서 뽑지 못했습니다. 양식 값을 씁니다."
        )
        if filled_by_form:
            source += f" 못 뽑은 {len(filled_by_form)}개는 양식에서 채웠습니다."

        self.outcome.stages.append(
            Stage(
                key="intake",
                title="1. 인테이크",
                status="done" if intake.verdict == "proceed" else "blocked",
                headline={"proceed": "진단으로 넘김", "need_more_info": "정보 부족 — 되물음",
                          "duplicate": "이미 해결된 사례 — 중단"}[intake.verdict],
                detail=f"{intake.note} {source}",
                rows=[(label, f"{getattr(intake.report, key) or '—'}"
                              f"{'  (양식)' if key in filled_by_form else '  (원문 추출)' if extracted.get(key) else ''}")
                      for key, label in (("line", "라인"), ("object_name", "품목"),
                                         ("defect_type", "결함 유형"),
                                         ("product_id", "제품명"), ("lot", "로트"))],
                note=intake.question or "라인·품목이 없으면 추측하지 않고 되묻습니다.",
            )
        )
        return {
            "verdict": intake.verdict,
            "line": intake.report.line,
            "object_name": intake.report.object_name,
            "product_id": intake.report.product_id,
            "lot": intake.report.lot,
            "extracted_from_text": [k for k, v in extracted.items() if v],
            "missing": intake.missing,
            "next": "lookup_mes" if intake.verdict == "proceed" else "중단. 사람에게 되물어야 한다.",
        }

    # ── 도구 2. MES 조회 ────────────────────────────────────────────────

    def lookup_mes(self, product_id: str = "", lot: str = "",
                   line: str = "", object_name: str = "") -> dict:
        """제품명·로트로 MES 에서 이미지를 찾고, 그 품목의 뱅크를 확인한다.

        이슈는 이미지가 아니라 제품명이나 로트로 온다. 이 단계가 없으면
        "어느 이미지를 볼 것인가"가 코드에 박히게 된다.

        **조인으로 답한다. 임베딩하지 않는다.** 언어 모델이 하는 일은 "MES 를
        조회해야겠다"고 판단해 이 도구를 부르는 데까지다.
        """
        intake = self.outcome.intake
        if intake is None:
            raise RuntimeError("먼저 intake_issue 를 불러야 한다.")
        if intake.verdict != "proceed":
            raise RuntimeError(f"인테이크가 진행하지 않기로 했다: {intake.verdict}")

        line = line or intake.report.line or ""
        obj = object_name or intake.report.object_name or ""
        product_id = product_id or intake.report.product_id or ""

        # 뱅크부터. 품목에 걸린 모델이 없으면 볼 것도 없다.
        profile = self.lookup.resolve_bank(line, obj)
        records = self.lookup.find_images(
            line=line or None, object_name=obj or None,
            lot=lot or None, product_id=product_id or None,
        )

        # 제품 하나만 지목됐으면 **그 제품이 속한 로트를 함께 가져온다.**
        # 한 장만 보고 "미검 1건"이라 하는 것은 의미가 없다. 같은 로트를 함께
        # 봐야 이 문제가 그 제품 하나인지 로트 전체인지 갈린다. 현장에서
        # 담당자가 하는 일도 같다.
        if product_id and len(records) == 1 and records[0].lot:
            batch = self.lookup.find_images(
                line=records[0].line, object_name=records[0].object_name,
                lot=records[0].lot,
            )
            if len(batch) > len(records):
                records = batch
                lot = lot or records[0].lot
        self.records = records
        self.item = self.factory.item_for(line, obj)

        missing = profile is None or self.item is None
        found_note = (
            f"{obj} 에 배포된 뱅크가 없다. 이 품목은 아직 검사 모델이 없다."
            if missing else
            f"뱅크 {profile.bank_version} · MES 이미지 {len(records)}장"
        )
        self.outcome.stages.append(
            Stage(
                key="mes",
                title="2. MES 조회",
                status="blocked" if missing or not records else "done",
                headline=found_note,
                detail=(
                    f"제품명·로트로 이미지를 찾았습니다. 조인으로 답한 값이며 "
                    f"벡터 검색이 아닙니다."
                ),
                rows=[("조회 조건", " · ".join(
                    f"{k}={v}" for k, v in
                    (("제품", product_id), ("로트", lot), ("라인", line), ("품목", obj)) if v) or "—"),
                    ("품목 뱅크", profile.bank_version if profile else "없음"),
                    ("찾은 이미지", f"{len(records)}장"),
                    ("결함으로 확인된 것", f"{sum(1 for r in records if r.ground_truth == 'defect')}장")],
                note="뱅크는 품목마다 다릅니다. 캡슐 뱅크로 PCB 를 판정할 수 없습니다.",
            )
        )
        if missing:
            raise RuntimeError(f"{line}/{obj} 에 배포된 뱅크가 없다.")
        if not records:
            raise RuntimeError("MES 에서 해당 조건의 이미지를 찾지 못했다.")

        return {
            "bank_version": profile.bank_version,
            "images": len(records),
            "defects": sum(1 for r in records if r.ground_truth == "defect"),
            "next": "run_inspection",
        }

    # ── 도구 3. 추론 ────────────────────────────────────────────────────

    def run_inspection(self) -> dict:
        """찾은 이미지를 그 품목 뱅크로 돌려 미검·과검을 가려낸다.

        사람이 개입하지 않습니다. 화면은 처리 과정을 보여줄 뿐입니다.
        """
        if self.item is None or not self.records:
            raise RuntimeError("먼저 lookup_mes 를 불러야 한다.")

        f = self.factory
        paths = [f.resolve(r.path) for r in self.records]
        results = score_images(paths, self.item.bank, f.embedder, root=f.root)
        by_path = {r.image: r for r in results}

        missed, overkill, rows = [], [], []
        for record in self.records:
            inferred = by_path.get(record.path)
            if inferred is None:
                continue
            verdict = inferred.verdict(self.threshold)
            if record.ground_truth == "defect" and verdict == "pass":
                missed.append((record, inferred))
            elif record.ground_truth == "pass" and verdict == "defect":
                overkill.append((record, inferred))

        #: 이상 점수가 이보다 낮으면 그 이미지 자신이 뱅크에 들어 있다는 뜻이다.
        #: 자기 패치와의 거리라 0 에 가깝게 나온다. 오염의 가장 뚜렷한 흔적이다.
        in_bank_score = 0.05

        for record, inferred in (missed + overkill)[:8]:
            kind = "미검" if record.ground_truth == "defect" else "과검"
            mark = "  ← 이 이미지가 뱅크 안에 있다" if inferred.score < in_bank_score else ""
            rows.append((f"{kind} · {record.product_id}",
                         f"점수 {inferred.score:.3f} (임계값 {self.threshold}) · "
                         f"로트 {record.lot}{mark}"))

        self_matched = [r for r, i in missed if i.score < in_bank_score]

        # 진단은 미검 한 건을 대표로 본다. 없으면 볼 것이 없다.
        self.query = f.resolve(missed[0][0].path) if missed else None
        self.inference = missed[0][1] if missed else None

        self.outcome.missed_records = [r for r, _ in missed]
        if missed:
            top_record, top_result = missed[0]
            self.outcome.inference = top_result
            self.outcome.query_image = top_record.path
            self.outcome.threshold = self.threshold
            self.outcome.bank_version = self.item.bank.version
            self.outcome.grid = (top_result.grid_h, top_result.grid_w)
        self.outcome.stages.append(
            Stage(
                key="inspect",
                title="3. 추론 — 미검·과검",
                status="done" if missed else "blocked",
                headline=f"{len(self.records)}장 중 미검 {len(missed)}장 · 과검 {len(overkill)}장",
                detail=(
                    f"{self.item.object_name} 뱅크 {self.item.bank.version} 로 판정했습니다. "
                    f"설비 판정과 사람 확인이 갈린 건만 추렸습니다."
                ),
                rows=rows or [("갈린 건", "없음")],
                note=(
                    (f"미검 {len(self_matched)}건은 이상 점수가 0 에 가깝습니다 — "
                     f"그 이미지 자신이 뱅크에 들어 있다는 뜻이고, 뱅크 오염의 "
                     f"가장 뚜렷한 흔적입니다. ") if self_matched else ""
                ) + "사람이 개입하지 않습니다. 처리 과정을 보는 화면입니다.",
            )
        )
        if not missed:
            raise RuntimeError("미검이 없다. 진단할 대상이 없다.")

        return {
            "total": len(self.records),
            "missed": len(missed),
            "overkill": len(overkill),
            "next": "run_checks",
        }

    # ── 도구 4. 판별 7항목 ──────────────────────────────────────────────

    def run_checks(self) -> dict:
        """판별 항목 일곱 가지를 모은다. 진단의 입력이다."""
        intake = self.outcome.intake
        if self.query is None or self.item is None or self.inference is None:
            raise RuntimeError("먼저 run_inspection 을 불러야 한다.")

        f = self.factory
        item = self.item
        line, obj = item.line, item.object_name

        result = self.inference
        baseline = self.lookup.get_quality_baseline(line, obj)
        quality = assess_quality([self.query], baseline.stats, min_images=1)
        visible = judge_defect_visible(
            self.vlm, self.query, reported_defect=intake.report.defect_type or "표면 결함"
        )

        patch_judgment = self._judge_nearest_patch(result)

        cell = (DEMO_CONFIG.crop / result.grid_h) * (DEMO_CONFIG.crop / result.grid_w)
        hot = sum(1 for row in result.patch_distances for v in row if v >= self.threshold * 0.8)

        self.evidence = collect_evidence(
            defect_visible=visible,
            quality=quality,
            inference=result,
            threshold=self.lookup.get_threshold(line, obj, item.bank.version),
            patch_judgment=patch_judgment,
            bank_profile=self.lookup.get_bank_profile(item.bank.version),
            conditions={"date": "2026-06-01"},
            criteria=self.lookup.get_criteria(line, obj, intake.report.defect_type or "dent"),
            defect_area=hot * cell,
        )
        self.outcome.stages.append(
            Stage(
                key="evidence",
                title="4. 판별 7항목",
                status="done",
                headline=f"{sum(1 for e in self.evidence if e.usable)}/7 확인",
                rows=[
                    (f"{e.item_no}. {e.name}", f"{'○' if e.usable else '×'}  {_evidence_value(e.value)}")
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

        diagnosis = decide(self.evidence, similar_issues=self.outcome.intake.similar,
                           current_line=self.outcome.intake.report.line)
        self.outcome.diagnosis = diagnosis
        self.outcome.stages.append(
            Stage(
                key="diagnose",
                title="5. 진단",
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
        item = self.item
        if item is None:
            raise RuntimeError("먼저 lookup_mes 를 불러야 한다.")
        missed = score_images(item.holdout_defect, item.bank, f.embedder, root=f.root)
        plan = plan_curation(diagnosis, bank=item.bank, missed_results=missed)
        self.outcome.plan = plan
        self.outcome.stages.append(
            Stage(
                key="curate",
                title="6. 데이터 큐레이션",
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
        item = self.item
        rebuild = execute_rebuild(
            plan, item.bank, DirectoryImageSource(f.root), f.embedder,
            triggered_by=self.issue_text,
        )
        self.outcome.rebuild = rebuild
        self.outcome.stages.append(
            Stage(
                key="rebuild",
                title="7. 뱅크 재구성",
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
        item = self.item
        normals = score_images(item.holdout_normal, rebuild.bank, f.embedder, root=f.root)
        defects = score_images(item.holdout_defect, rebuild.bank, f.embedder, root=f.root)
        curve = sweep_from_results(normals, defects)
        self.new_threshold = (curve.threshold_for_detection(1.0) or curve.points[0]).threshold

        self.baseline_curve = sweep_from_results(
            score_images(item.holdout_normal, item.bank, f.embedder, root=f.root),
            score_images(item.holdout_defect, item.bank, f.embedder, root=f.root),
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
                title="8. 평가 게이트",
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
        item = self.item
        shadow = shadow_compare(
            list(item.holdout_normal) + list(item.holdout_defect),
            item.bank, rebuild.bank,
            current_threshold=self.threshold, candidate_threshold=self.new_threshold,
            embedder=f.embedder, root=f.root,
        )
        self.outcome.shadow = shadow
        self.outcome.stages.append(
            Stage(
                key="shadow",
                title="9. 섀도 비교",
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

        # 결함이 한 로트에 몰려 있으면 자재나 설비를 먼저 봐야 한다. 그 판단을
        # 승인하는 사람이 하려면 집계가 문서에 있어야 한다.
        item = self.item
        o.distribution = self.lookup.defect_distribution(
            line=item.line if item else None,
            object_name=item.object_name if item else None,
        )

        package = prepare_release(
            self.factory.root / "release" / o.rebuild.bank.version,
            bank=o.rebuild.bank, record=o.rebuild.record, diagnosis=o.diagnosis, plan=o.plan,
            gate=o.gate, shadow=o.shadow, reproducibility=reproducibility,
            issue_text=self.issue_text,
            distribution=o.distribution, affected=o.missed_records,
        )
        o.package = package
        o.approval_markdown = package.approval_document.read_text(encoding="utf-8")
        o.stages.append(
            Stage(
                key="release",
                title="10. 승인 요청",
                status="done",
                headline="배포 대기 — 자동 반영 없음",
                detail=f"승인 요청 문서를 생성했습니다. 재현성 {reproducibility.runs}회 "
                       f"{'일치' if reproducibility.identical else '불일치'}.",
                rows=[("배포 승인", "아니오 (사람이 결정)"),
                      ("결함 분포", o.distribution.describe() if o.distribution else "—")]
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
            Tool(MES_SPEC, self.lookup_mes),
            Tool(INSPECT_SPEC, self.run_inspection),
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
        보충 입력. **비워 두는 것이 기본이다.** 언어 모델이 이슈 원문에서 뽑고,
        못 뽑은 자리만 여기서 채운다. 기본값을 코드에 두면 화면에 값이 어디선가
        나타나고, 사람이 "모델이 뽑은 것"으로 오해한다. 아무것도 없고 모델도
        없으면 인테이크가 되묻는다 — 그것이 옳은 동작이다.

    언어 모델이 붙어 있으면 모델이 도구 순서를 정하고, 없으면 같은 도구들을
    고정 순서로 재생한다. 어느 쪽이었는지는 outcome.driver 에 남는다.
    """
    llm, vlm = adapters or build_adapters()
    session = _DemoSession(
        factory, issue_text, dict(context or {}),
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
    # 어떤 조회를 실제로 했는지 화면에 그대로 띄운다. 지어낸 목록이 아니라
    # 조회 계층이 남긴 호출 기록이고, 방식 표시는 RETRIEVAL_KIND 에서 온다.
    # 실구현이 calls 를 안 남기면 이 칸은 비고, 그때는 비어 있는 것이 맞다.
    session.outcome.retrievals = [
        {
            "name": name,
            "kind": RETRIEVAL_KIND.get(name, "unknown"),
            "arguments": {k: v for k, v in arguments.items() if v},
        }
        for name, arguments in getattr(session.lookup, "calls", [])
    ]
    return session.outcome
