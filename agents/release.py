"""릴리즈 에이전트 — 배포 패키지와 승인 요청 (작업 20).

**여기에 배포하는 함수는 없다.** 앞으로도 만들지 않는다.

품질 검사 설비는 잘못 반영하면 불량을 흘려보내거나 라인을 세운다. 그래서
에이전트가 하는 일은 "사람이 판단할 수 있게 모아 주는 것"까지이고, 실제
장비 반영은 사람이 한다. 이 경계를 스스로 그은 것 자체가 제안의 설득
근거이기도 하다.

승인 요청 문서가 이 파일의 결과물이다. 담당자가 그 문서 하나만 읽고
승인할지 말지 정할 수 있어야 한다. 그러려면 판단에 필요한 것이 다 있고,
불리한 것도 숨기지 않아야 한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

from inspection.bank import MemoryBank
from inspection.shadow import ShadowReport
from lookup.base import DefectDistribution, ImageRecord

from .curate import CurationPlan
from .diagnose import CAUSE_LABEL_KO, DiagnosisResult
from .gate import GateResult, ReproducibilityResult
from .rebuild import RebuildRecord


@dataclass
class ReleasePackage:
    """배포 후보 묶음.

    approved 는 항상 False 로 만들어진다. 승인은 이 파일 밖에서, 사람이 한다.
    """

    version: str
    directory: Path
    approval_document: Path
    approved: bool = False
    blocking_reasons: list[str] = field(default_factory=list)

    @property
    def ready_for_review(self) -> bool:
        """사람이 검토할 수 있는 상태인가. 승인 가능 여부와 다르다."""
        return self.approval_document.exists()

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "directory": str(self.directory),
            "approval_document": str(self.approval_document),
            "approved": self.approved,
            "blocking_reasons": self.blocking_reasons,
        }


#: 이 파일 기준 저장소 뿌리. `agents/release.py` 의 두 단계 위다.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def display_path(path: str | Path) -> str:
    """승인 문서에 적을 경로. **저장소 기준 상대경로로 줄인다.**

    절대경로를 그대로 쓰면 사용자명과 전체 폴더 구조가 문서에 박힌다.
    실측에서 `C:\\Users\\<사번>\\Desktop\\...` 이 승인 요청 문서에 찍혀
    나갔고, 하필 그 화면이 시연에서 보여주는 자리였다.

    승인자가 열어야 하는 것은 저장소 안의 폴더이므로 상대경로로 충분하다.
    저장소 밖이면 마지막 두 조각만 남긴다. 그것도 아니면 이름만 쓴다.
    """
    p = Path(path)
    shown = None
    try:
        shown = p.resolve().relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        pass
    # **시연은 윈도우에서 돌고 시험은 맥에서 돈다.** 맥의 `Path` 는 역슬래시를
    # 구분자로 보지 않아 `C:\Users\...` 를 통째로 한 조각으로 읽는다. 그러면
    # 아래 "마지막 두 조각" 이 경로 전체가 되어 사용자명이 그대로 남는다.
    if "\\" in str(p) and "/" not in str(p):
        p = Path(str(p).replace("\\", "/"))
    # 저장소 안이어도 사용자 폴더를 거쳐 들어온 경로가 있을 수 있고,
    # 다른 운영체제의 경로 문자열이 섞여 들어오면 위 계산이 통하지 않는다.
    # **어느 쪽이든 사용자명이 남으면 쓰지 않는다.**
    if shown is None or _looks_personal(shown):
        parts = p.parts[-2:]
        shown = Path(*parts).as_posix() if parts else p.name
    return shown


def _looks_personal(text: str) -> bool:
    """사람 폴더를 가리키는 조각이 남아 있는가."""
    lowered = text.lower().replace("\\", "/")
    return (
        lowered.startswith(("users/", "home/", "c:/", "d:/"))
        or "/users/" in lowered
        or "/home/" in lowered
    )


def _evidence_table(diagnosis: DiagnosisResult) -> str:
    lines = [
        "| # | 판별 항목 | 값 | 출처 | 확인 |",
        "|---|---|---|---|---|",
    ]
    source_ko = {"vlm": "시각 언어 모델", "lookup": "조회", "compute": "계산", "trace": "역추적"}
    for item in diagnosis.evidence:
        value = item.value
        if isinstance(value, dict):
            value = value.get("source_image", str(value))
        lines.append(
            f"| {item.item_no} | {item.name} | {value} | "
            f"{source_ko.get(item.source, item.source)} | {'○' if item.usable else '×'} |"
        )
    return "\n".join(lines)


def _gate_table(gate: GateResult) -> str:
    lines = ["| 항목 | 값 | 기준 | 결과 |", "|---|---|---|---|"]
    for check in gate.checks:
        lines.append(
            f"| {check.name} | {check.value} | {check.threshold} | "
            f"{'통과' if check.passed else '**미달**'} |"
        )
    return "\n".join(lines)


def write_approval_document(
    path: Path,
    *,
    bank: MemoryBank,
    record: RebuildRecord,
    diagnosis: DiagnosisResult,
    plan: CurationPlan,
    gate: GateResult,
    shadow: ShadowReport | None = None,
    reproducibility: ReproducibilityResult | None = None,
    issue_text: str = "",
    distribution: "DefectDistribution | None" = None,
    affected: "Sequence[ImageRecord]" = (),
) -> Path:
    """승인 요청 문서를 쓴다.

    담당자가 이 문서 하나로 판단할 수 있어야 한다. 그래서 불리한 것도
    같이 적는다 — 게이트에서 떨어진 항목, 섀도에서 새로 놓친 건, 근거를
    얻지 못한 판별 항목.
    """
    cause_label = CAUSE_LABEL_KO.get(diagnosis.cause or "", "판정 보류")

    parts: list[str] = []
    parts.append(f"# 뱅크 배포 승인 요청: {record.to_version}\n")
    parts.append(
        "> 이 문서는 자동 생성됐습니다. **배포는 실행되지 않았습니다.** "
        "아래를 검토하고 승인 여부를 결정해 주세요.\n"
    )

    # ── 요약 ────────────────────────────────────────────────────────
    parts.append("## 요약\n")
    parts.append(f"- **원인**: {cause_label}")
    parts.append(f"- **조치**: {plan.summary()}")
    parts.append(f"- **뱅크**: `{record.from_version}` → `{record.to_version}`")
    parts.append(f"- **게이트**: {'통과' if gate.passed else '**미통과**'}")
    if shadow:
        parts.append(f"- **사람 확인 필요**: {shadow.review_count}건")
    parts.append("")

    if issue_text:
        parts.append("## 접수된 이슈\n")
        parts.append(f"> {issue_text}\n")

    # ── 무엇이 걸렸나 ───────────────────────────────────────────────
    #
    # 결함이 한 로트에 몰려 있으면 자재나 설비를 먼저 의심해야 한다. 그때
    # 뱅크부터 다시 만들면 원인을 놔둔 채 증상만 덮는 셈이다. 승인하는 사람이
    # 그 판단을 하려면 집계가 문서에 있어야 한다.
    if affected:
        parts.append("## 대상 이미지\n")
        parts.append(f"미검으로 확인된 {len(affected)}건입니다.\n")
        parts.append("| 제품 | 라인 | 로트 | 설비 | 촬영일 |")
        parts.append("|---|---|---|---|---|")
        # 반복 변수를 record 로 두면 인자 record(RebuildRecord)를 덮어쓴다.
        # 파이썬 for 변수는 함수 스코프에 남는다.
        for image in list(affected)[:12]:
            parts.append(
                f"| `{image.product_id}` | {image.line} | {image.lot or '—'} | "
                f"{image.equipment or '—'} | "
                f"{image.captured_at.isoformat() if image.captured_at else '—'} |"
            )
        if len(affected) > 12:
            parts.append(f"| … | | | | 외 {len(affected) - 12}건 |")
        parts.append("")

    if distribution is not None and distribution.total:
        parts.append("## 결함이 어디에 몰렸나\n")
        parts.append(f"{distribution.describe()}\n")
        for title, counts in (("로트", distribution.by_lot),
                              ("라인", distribution.by_line),
                              ("설비", distribution.by_equipment)):
            if not counts:
                continue
            ordered = sorted(counts.items(), key=lambda kv: -kv[1])
            inline = ", ".join(
                f"{key} {n}건({n / distribution.total:.0%})" for key, n in ordered[:5]
            )
            parts.append(f"- **{title}**: {inline}")
        parts.append("")
        if distribution.concentrated_in():
            parts.append(
                "> 한쪽에 몰려 있습니다. **뱅크 재구성으로 덮기 전에 그쪽 원인을 "
                "먼저 확인해 주세요.** 자재나 설비 문제라면 뱅크를 다시 만들어도 "
                "같은 일이 반복됩니다.\n"
            )

    # ── 진단 ────────────────────────────────────────────────────────
    parts.append("## 진단\n")
    parts.append(f"{diagnosis.reasoning}\n")
    parts.append(_evidence_table(diagnosis))
    parts.append("")
    unusable = [e for e in diagnosis.evidence if not e.usable]
    if unusable:
        names = ", ".join(f"{e.item_no}번({e.name})" for e in unusable)
        parts.append(
            f"> 확인하지 못한 판별 항목: {names}. 이 항목들은 판정 근거에서 제외됐습니다.\n"
        )

    # ── 조치 내용 ───────────────────────────────────────────────────
    parts.append("## 뱅크에 무엇이 바뀌었나\n")
    parts.append(f"{record.reason}\n")
    if record.removed:
        parts.append(f"**제거 {len(record.removed)}장**\n")
        for image in record.removed[:20]:
            reason = next((c.reason for c in plan.remove if c.image == image), "")
            parts.append(f"- `{image}`: {reason}")
        if len(record.removed) > 20:
            parts.append(f"- … 외 {len(record.removed) - 20}장")
        parts.append("")
    if record.added:
        parts.append(f"**보충 {len(record.added)}장**\n")
        for image in record.added[:20]:
            parts.append(f"- `{image}`")
        if len(record.added) > 20:
            parts.append(f"- … 외 {len(record.added) - 20}장")
        parts.append("")
    parts.append(f"유지된 이미지 {record.kept_count}장, 최종 뱅크 {len(bank)}행\n")

    # ── 평가 ────────────────────────────────────────────────────────
    parts.append("## 평가 게이트\n")
    parts.append(_gate_table(gate))
    parts.append("")
    parts.append(f"{gate.reason}\n")

    if reproducibility:
        mark = "100%" if reproducibility.identical else "**불일치**"
        parts.append(f"**재현성**: {reproducibility.runs}회 반복, {mark}. {reproducibility.detail}\n")

    # ── 섀도 ────────────────────────────────────────────────────────
    if shadow:
        parts.append("## 섀도 비교\n")
        parts.append(
            "신규 뱅크를 실제 판정에 쓰지 않고 같은 이미지에 병렬로만 추론시킨 결과입니다. "
            "판정이 갈린 건만 확인하시면 됩니다.\n"
        )
        parts.append(f"{shadow.summary()}\n")

        if shadow.newly_missed:
            parts.append(f"### 새로 놓치는 건 {len(shadow.newly_missed)}건 (확인 필요)\n")
            parts.append("고치려던 문제가 나아져도 다른 것을 잃으면 개선이 아닙니다.\n")
            parts.append("| 이미지 | 현행 점수 | 후보 점수 |")
            parts.append("|---|---|---|")
            for d in shadow.newly_missed[:15]:
                parts.append(f"| `{d.image}` | {d.current_score:.4f} | {d.candidate_score:.4f} |")
            if len(shadow.newly_missed) > 15:
                parts.append(f"| … | | 외 {len(shadow.newly_missed) - 15}건 |")
            parts.append("")

        if shadow.newly_detected:
            parts.append(f"### 새로 잡는 건 {len(shadow.newly_detected)}건\n")
            parts.append("| 이미지 | 현행 점수 | 후보 점수 |")
            parts.append("|---|---|---|")
            for d in shadow.newly_detected[:15]:
                parts.append(f"| `{d.image}` | {d.current_score:.4f} | {d.candidate_score:.4f} |")
            if len(shadow.newly_detected) > 15:
                parts.append(f"| … | | 외 {len(shadow.newly_detected) - 15}건 |")
            parts.append("")

    # ── 승인 ────────────────────────────────────────────────────────
    parts.append("## 승인\n")
    if not gate.passed:
        parts.append(
            "**게이트를 통과하지 못했습니다.** 아래 항목이 기준에 미달합니다. "
            "그대로 승인하실 경우 근거를 남겨 주세요.\n"
        )
        for check in gate.failures:
            parts.append(f"- {check.name}: {check.value} (기준 {check.threshold}) — {check.detail}")
        parts.append("")

    parts.append("승인하시면 아래를 수동으로 진행합니다. **자동 반영은 없습니다.**\n")
    parts.append(f"1. 배포 패키지 확인: `{display_path(path.parent)}`")
    parts.append("2. 장비에 뱅크 반영 (담당자 수행)")
    parts.append("3. 반영 후 초기 물량 모니터링")
    parts.append("")
    parts.append("| | |")
    parts.append("|---|---|")
    parts.append("| 요청 시작 | " + (record.triggered_by or "—") + " |")
    parts.append(f"| 대상 뱅크 | `{record.to_version}` |")
    parts.append("| 승인자 | (서명) |")
    parts.append("| 승인 일시 | |")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def prepare_release(
    directory: str | Path,
    *,
    bank: MemoryBank,
    record: RebuildRecord,
    diagnosis: DiagnosisResult,
    plan: CurationPlan,
    gate: GateResult,
    shadow: ShadowReport | None = None,
    reproducibility: ReproducibilityResult | None = None,
    issue_text: str = "",
    distribution: "DefectDistribution | None" = None,
    affected: "Sequence[ImageRecord]" = (),
) -> ReleasePackage:
    """배포 패키지를 만든다. 배포하지는 않는다.

    담기는 것
      bank.npz / bank_meta.json   재구성된 뱅크
      rebuild_record.json          무엇을 왜 바꿨는가
      diagnosis.json               진단 결과와 근거
      gate.json                    평가 결과
      shadow.json                  섀도 비교 (있으면)
      승인요청.md                   사람이 읽는 문서
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    bank.save(directory)

    def dump(name: str, payload: Any) -> None:
        (directory / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    dump("rebuild_record.json", record.to_dict())
    dump("diagnosis.json", diagnosis.to_dict())
    dump("curation_plan.json", plan.to_dict())
    dump("gate.json", gate.to_dict())
    if shadow:
        dump("shadow.json", shadow.to_dict())
    if reproducibility:
        dump("reproducibility.json", reproducibility.to_dict())
    if distribution is not None:
        dump("defect_distribution.json", distribution.to_dict())
    if affected:
        dump("affected_images.json", [r.to_dict() for r in affected])

    document = write_approval_document(
        directory / "승인요청.md",
        bank=bank, record=record, diagnosis=diagnosis, plan=plan,
        gate=gate, shadow=shadow, reproducibility=reproducibility, issue_text=issue_text,
        distribution=distribution, affected=affected,
    )

    blocking: list[str] = []
    if not gate.passed:
        blocking.extend(f"게이트 미달: {c.name}" for c in gate.failures)
    if shadow and shadow.newly_missed:
        blocking.append(f"섀도에서 새로 놓치는 건 {len(shadow.newly_missed)}건")
    if reproducibility and not reproducibility.identical:
        blocking.append("재현성 불일치")

    return ReleasePackage(
        version=record.to_version,
        directory=directory,
        approval_document=document,
        approved=False,          # 언제나 False. 승인은 사람이 한다
        blocking_reasons=blocking,
    )
