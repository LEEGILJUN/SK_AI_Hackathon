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

import hashlib
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
    실측에서 윈도우 사용자 폴더 아래 전체 경로가 승인 요청 문서에 찍혀
    나갔고, 하필 그 화면이 시연에서 보여주는 자리였다.

    **이 설명에 예시 경로를 적지 않는다.** 사용자 폴더를 가리키는 문자열은
    공개 준비 검사가 유출로 잡는데, 잡아야 맞다 — 설명이라도 저장소에
    남으면 그 형태가 남는다.

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


def document_number(bank_version: str) -> str:
    """승인 요청 문서의 관리번호.

    **실무 문서에 번호가 없으면 되짚을 수가 없다.** 승인 이력을 쌓아 두고
    "그때 그 건"을 부르려면 부를 이름이 있어야 한다.

    뱅크 판 이름에서 만든다. 같은 판에는 같은 번호가 나오므로 문서를 다시
    생성해도 번호가 흔들리지 않는다. 시각을 넣으면 다시 만들 때마다 번호가
    달라져 같은 건이 두 개로 보인다.

        pcb1-01-v2_20260817-0252_cad872  →  AR-20260817-cad872
    """
    parts = str(bank_version).split("_")
    # **`hash()` 를 쓰지 않는다.** 파이썬 해시는 실행마다 씨앗이 달라져
    # 같은 판에 다른 번호가 나온다. 문서번호가 흔들리면 부를 이름이 못 된다.
    fingerprint = parts[2] if len(parts) > 2 else hashlib.sha1(
        str(bank_version).encode("utf-8")).hexdigest()[:6]
    if len(parts) > 1:
        return f"AR-{parts[1].split('-')[0]}-{fingerprint}"
    # 시각이 이름에 없는 판(합성·시험)은 그 자리를 비운다. `00000000` 을
    # 채우면 날짜를 아는 것처럼 보인다.
    return f"AR-{fingerprint}"


def bank_settings(bank: MemoryBank, record: RebuildRecord,
                  gate: GateResult, threshold: float | None = None) -> dict[str, str]:
    """이 판정이 어떤 설정 위에서 나왔는가.

    **문서를 재활용하려면 이것이 있어야 한다.** 나중에 되짚을 때 임계값과
    입력 크기, 코어셋 비율을 모르면 같은 조건으로 다시 재볼 수가 없다.
    폴더 이름의 설정 지문과 같은 값을 사람이 읽는 형태로 적는다.

    **값이 없으면 그 줄을 빼지 않고 "기록 없음"으로 남긴다.** 줄이 사라지면
    읽는 사람이 "원래 없는 항목"으로 오해한다.
    """
    meta = dict(getattr(bank, "meta", {}) or {})
    config = dict(meta.get("feature_config") or {})
    grid = meta.get("grid") or []
    rows: dict[str, str] = {
        "판정 임계값": _fmt(threshold),
        "입력 크기": _fmt(config.get("crop") or config.get("resize")),
        "특징 추출기": _fmt(config.get("backbone")),
        "격자": f"{grid[0]}×{grid[1]}" if len(grid) == 2 else "기록 없음",
        "coreset 비율": _fmt(meta.get("coreset_ratio")),
        "뱅크 행수": _fmt(_bank_rows(bank)),
        "구성 이미지 수": _fmt(meta.get("source_image_count")),
        "무작위 씨앗": _fmt(meta.get("seed")),
        "이전 판": _fmt(record.from_version),
        "이번 판": _fmt(record.to_version),
    }
    if meta.get("coreset_capped"):
        rows["coreset 상한"] = f"{meta.get('max_bank_size')} 에 걸려 요청 비율대로 안 됨"
    return rows


def _bank_rows(bank: MemoryBank) -> int | None:
    """뱅크 행수. **numpy 배열에 `or` 를 쓰지 않는다.**

    `embeddings or []` 는 배열의 참·거짓을 물어 `ValueError` 가 난다.
    길이만 본다.
    """
    rows = getattr(bank, "embeddings", None)
    try:
        return int(len(rows))
    except TypeError:
        return None


def _fmt(value: Any) -> str:
    """값이 없으면 "기록 없음". **0 은 값이다.**

    전에 `value == 0` 으로 걸렀는데, 0 은 없는 것이 아니라 0 이다.
    그리고 numpy 배열에 `==` 을 쓰면 배열이 돌아와 참·거짓을 못 묻는다.
    """
    if value is None:
        return "기록 없음"
    if isinstance(value, str):
        return value if value.strip() else "기록 없음"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


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
            f"{'통과' if check.passed else '미달'} |"
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
    threshold: float | None = None,
) -> Path:
    """승인 요청 문서를 쓴다.

    담당자가 이 문서 하나로 판단할 수 있어야 한다. 그래서 불리한 것도
    같이 적는다 — 게이트에서 떨어진 항목, 섀도에서 새로 놓친 건, 근거를
    얻지 못한 판별 항목.
    """
    cause_label = CAUSE_LABEL_KO.get(diagnosis.cause or "", "판정 보류")

    doc_no = document_number(record.to_version)

    parts: list[str] = []
    parts.append(f"뱅크 배포 승인 요청  {record.to_version}")
    parts.append(f"문서번호  {doc_no}\n")
    parts.append(
        "이 문서는 자동 생성됐습니다. 배포는 실행되지 않았습니다. "
        "아래를 검토하고 승인 여부를 결정해 주세요.\n"
    )

    # ── 요약 ────────────────────────────────────────────────────────
    parts.append("\n1. 요약\n")
    parts.append(f"- 원인: {cause_label}")
    parts.append(f"- 조치: {plan.summary()}")
    parts.append(f"- 뱅크: {record.from_version} → {record.to_version}")
    parts.append(f"- 성능 검증: {'통과' if gate.passed else '미통과'}")
    if shadow:
        parts.append(f"- 사람이 확인할 건: {shadow.review_count}건")
    parts.append("")

    if issue_text:
        parts.append("\n2. 접수된 이슈\n")
        parts.append(f"  {issue_text}\n")

    # ── 무엇이 걸렸나 ───────────────────────────────────────────────
    #
    # 결함이 한 로트에 몰려 있으면 자재나 설비를 먼저 의심해야 한다. 그때
    # 뱅크부터 다시 만들면 원인을 놔둔 채 증상만 덮는 셈이다. 승인하는 사람이
    # 그 판단을 하려면 집계가 문서에 있어야 한다.
    if affected:
        parts.append("\n3. 대상 이미지\n")
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
        parts.append("\n4. 결함이 어디에 몰렸나\n")
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
            parts.append(f"- {title}: {inline}")
        parts.append("")
        if distribution.concentrated_in():
            parts.append(
                "  한쪽에 몰려 있습니다. 뱅크 재구성으로 덮기 전에 그쪽 원인을 "
                "먼저 확인해 주세요. 자재나 설비 문제라면 뱅크를 다시 만들어도 "
                "같은 일이 반복됩니다.\n"
            )

    # ── 진단 ────────────────────────────────────────────────────────
    parts.append("\n5. 원인 규명\n")
    parts.append(f"{diagnosis.reasoning}\n")
    parts.append(_evidence_table(diagnosis))
    parts.append("")
    unusable = [e for e in diagnosis.evidence if not e.usable]
    if unusable:
        names = ", ".join(f"{e.item_no}번({e.name})" for e in unusable)
        parts.append(
            f"  확인하지 못한 판별 항목: {names}. 이 항목들은 판정 근거에서 제외됐습니다.\n"
        )

    # ── 조치 내용 ───────────────────────────────────────────────────
    parts.append("\n6. 뱅크에 무엇이 바뀌었나\n")
    parts.append(f"{record.reason}\n")
    if record.removed:
        parts.append(f"제거 {len(record.removed)}장\n")
        for image in record.removed[:20]:
            reason = next((c.reason for c in plan.remove if c.image == image), "")
            parts.append(f"- `{image}`: {reason}")
        if len(record.removed) > 20:
            parts.append(f"- … 외 {len(record.removed) - 20}장")
        parts.append("")
    if record.added:
        parts.append(f"보충 {len(record.added)}장\n")
        for image in record.added[:20]:
            parts.append(f"- `{image}`")
        if len(record.added) > 20:
            parts.append(f"- … 외 {len(record.added) - 20}장")
        parts.append("")
    parts.append(f"유지된 이미지 {record.kept_count}장, 최종 뱅크 {len(bank)}행\n")

    # ── 평가 ────────────────────────────────────────────────────────
    parts.append("\n7. 성능 검증\n")
    parts.append(_gate_table(gate))
    parts.append("")
    parts.append(f"{gate.reason}\n")

    if reproducibility:
        mark = "100%" if reproducibility.identical else "**불일치**"
        parts.append(
            f"재현성: 같은 근거로 {reproducibility.runs}회 다시 판정했고 {mark}. "
            f"{reproducibility.detail}\n"
        )

    # ── 섀도 ────────────────────────────────────────────────────────
    if shadow:
        parts.append("\n8. 신구 비교\n")
        parts.append(
            "신규 뱅크를 실제 판정에 쓰지 않고 같은 이미지에 병렬로만 추론시킨 결과입니다. "
            "판정이 갈린 건만 확인하시면 됩니다.\n"
        )
        parts.append(f"{shadow.summary()}\n")

        if shadow.newly_missed:
            parts.append(f"새로 놓치는 건 {len(shadow.newly_missed)}건 (확인 필요)\n")
            parts.append("고치려던 문제가 나아져도 다른 것을 잃으면 개선이 아닙니다.\n")
            parts.append("| 이미지 | 현행 점수 | 후보 점수 |")
            parts.append("|---|---|---|")
            for d in shadow.newly_missed[:15]:
                parts.append(f"| `{d.image}` | {d.current_score:.4f} | {d.candidate_score:.4f} |")
            if len(shadow.newly_missed) > 15:
                parts.append(f"| … | | 외 {len(shadow.newly_missed) - 15}건 |")
            parts.append("")

        if shadow.newly_detected:
            parts.append(f"새로 잡는 건 {len(shadow.newly_detected)}건\n")
            parts.append("| 이미지 | 현행 점수 | 후보 점수 |")
            parts.append("|---|---|---|")
            for d in shadow.newly_detected[:15]:
                parts.append(f"| `{d.image}` | {d.current_score:.4f} | {d.candidate_score:.4f} |")
            if len(shadow.newly_detected) > 15:
                parts.append(f"| … | | 외 {len(shadow.newly_detected) - 15}건 |")
            parts.append("")

    # ── 무엇으로 판정했나 ───────────────────────────────────────────
    #
    # **이 절이 없으면 문서를 재활용할 수 없다.** 나중에 이 승인을 되짚을 때
    # "그때 임계값이 얼마였나", "어떤 해상도로 뽑은 뱅크인가"를 알 수 없으면
    # 같은 조건으로 다시 재볼 수가 없다. 뱅크 폴더 이름의 설정 지문과 같은
    # 값을 사람이 읽는 형태로 함께 적는다.
    parts.append("\n9. 무엇으로 판정했나\n")
    parts.append("| 항목 | 값 |")
    parts.append("|---|---|")
    for label, value in bank_settings(bank, record, gate, threshold).items():
        parts.append(f"| {label} | {value} |")
    parts.append("")

    # ── 승인 ────────────────────────────────────────────────────────
    parts.append("\n10. 승인\n")
    if not gate.passed:
        parts.append(
            "성능 검증을 통과하지 못했습니다. 아래 항목이 기준에 미달합니다. "
            "그대로 승인하실 경우 근거를 남겨 주세요.\n"
        )
        for check in gate.failures:
            parts.append(f"- {check.name}: {check.value} (기준 {check.threshold}): {check.detail}")
        parts.append("")

    parts.append("승인하시면 아래를 수동으로 진행합니다. 자동 반영은 없습니다.\n")
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
    threshold: float | None = None,
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
        threshold=threshold,
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
