"""데이터 큐레이션 에이전트 — 뱅크에 무엇을 넣고 뺄 것인가 (작업 15).

진단이 원인을 정하면 여기서 조치를 계획한다. 판단하는 것은 하나다.

    **뱅크를 건드릴 것인가, 건드린다면 무엇을.**

여섯 원인 중 넷은 뱅크를 다시 만드는 것이 답이 아니다. 그 넷에서는 빈 계획을
돌려주고 왜 건드리지 않는지를 남긴다. 이 서비스의 가치가 불필요한 재학습을
막는 데 있으므로, **아무것도 하지 않기로 하는 것도 하나의 결정**이며 근거가
남아야 한다.

계획은 실행과 분리되어 있다. 여기서는 무엇을 할지만 정하고, 실제 재구성은
rebuild.py 가 한다. 사람이 계획을 보고 승인하거나 되돌릴 수 있어야 하기
때문이다.

── 제거 단위는 이미지다. 패치가 아니다 ──────────────────────────────────

VisA 실측(2026-08-14)에서 혼입 이미지 5장이 뱅크에 남긴 패치 404개 중 실제
결함 위는 **6개뿐**이었다. 이미지 단위로 빼면 멀쩡한 정상 표면 398개(98.5%)를
함께 버린다. "패치 6개만 빼면 되지 않나"가 자연스러운 반응이고, 그럼에도
이미지 단위를 유지한다. 이유가 셋이다.

1. **이웃 평균이 결함을 번지게 한다.** `FeatureConfig.neighborhood=3` 이라 각
   패치의 임베딩에 주변 여덟 칸이 섞인다. 결함 칸만 빼면 그 둘레의 패치들은
   결함 신호를 머금은 채 뱅크에 남는다. 해상도 실험에서 결함이 이웃 평균에
   희석되던 것과 같은 기제가 반대로 작용한다.
2. **재현성이 깨진다.** 재구성은 "구성 이미지 목록 + 같은 시드"로 처음부터
   다시 만든다. 그래야 달라진 것이 구성뿐이고 성능 차이를 구성 변화로 읽을 수
   있다. 특정 행만 사후에 덜어내면 그 뱅크는 같은 입력으로 재현되지 않고,
   평가 게이트의 재현성 검사가 의미를 잃는다.
3. **운영에는 마스크가 없다.** 위의 "6개"는 VisA 정답 마스크로 갈라서 안 값이다.
   현장에서 어느 칸이 결함인지는 판별 5번이 패치마다 답해야 알 수 있고, 뱅크
   수천 행에 그것을 다 물을 수는 없다.

**대신 버리는 비용을 계획에 적는다.** 제거가 커버리지를 얼마나 깎는지 숫자로
남기고, 특정 조건이 비게 되면 제거와 보충을 짝지어 요구한다. 뱅크 오염을 고치려다
커버리지 부족을 만드는 것은 여섯 원인 중 하나를 스스로 만드는 일이다.

── 고립도는 단독 근거가 못 된다 ──────────────────────────────────────────

같은 실측에서 `suspect_images()` 가 낸 후보 1개 중 혼입 이미지는 0개였다. 뱅크 오염은
패치 단위로 일어나는데 고립도는 이미지 단위 평균이라 정상부 패치에 희석된다.
그래서 고립도만으로는 제거 후보를 만들지 않고, **역추적이 이미 지목한 이미지를
보강하는 데만 쓴다.** 정상 이미지를 잘못 빼면 커버리지 부족을 스스로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Sequence

from inspection.bank import MemoryBank
from inspection.isolation import IsolationScore, suspect_images
from inspection.types import InferenceResult
from lookup.base import BankProfile

from .diagnose import CAUSE_LABEL_KO, REBUILD_REQUIRED, DiagnosisResult


@dataclass
class RemovalCandidate:
    """뱅크에서 뺄 후보 한 건.

    근거가 세 갈래로 들어온다. 역추적이 반복해서 지목한 것, 판별 5번이 결함으로
    확인한 것, 뱅크 안에서 고립도가 높은 것. 겹칠수록 오판 위험이 낮다.

    **고립도는 단독으로 후보를 만들지 못한다.** 실측에서 혼입 이미지를 못 짚었고,
    정상 이미지를 잘못 빼면 커버리지 부족을 스스로 만들기 때문이다. 역추적이
    이미 지목한 이미지에 붙어 확신을 올리는 역할만 한다.

    patch_count 는 이 이미지가 뱅크에 남긴 패치 수다. 빼는 비용이 얼마인지를
    계획에 적기 위해 들고 있는다.
    """

    image: str
    reason: str
    traced_hits: int = 0          # 미검출 건들이 이 이미지를 최근접으로 지목한 횟수
    isolation_z: float | None = None
    confirmed_by_vlm: bool = False
    patch_count: int = 0

    @property
    def evidence_count(self) -> int:
        """근거가 몇 갈래인가. 겹칠수록 오판 위험이 낮다."""
        return sum([self.traced_hits > 0, self.isolation_z is not None, self.confirmed_by_vlm])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"evidence_count": self.evidence_count}


@dataclass
class AdditionRequest:
    """뱅크에 채워야 할 조건 하나.

    이미지 목록이 아니라 **조건**으로 남긴다. 어떤 이미지가 그 조건에
    해당하는지는 조회 계층이 안다. 큐레이션이 파일 목록을 직접 들고 있으면
    데이터가 바뀔 때마다 계획이 낡는다.
    """

    condition_key: str
    condition_value: str
    reason: str
    minimum_images: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CoverageCost:
    """제거가 뱅크에서 무엇을 함께 가져가는가.

    혼입 이미지를 통째로 빼면 그 이미지의 멀쩡한 정상 패치도 같이 빠진다.
    VisA 실측에서 그 비율이 98.5% 였다 — 결함 위 6개를 빼려고 정상 표면 398개를
    버린 셈이다. 그래도 이미지 단위로 빼는 이유는 curate 모듈 첫머리에 있다.

    빼는 것 자체는 막지 않고, **얼마를 버리는지 숫자로 남긴다.** 커버리지
    부족은 이 과제가 다루는 여섯 원인 중 하나이므로, 뱅크 오염을 고치다 그것을
    만들고 있지 않은지 사람이 볼 수 있어야 한다.
    """

    removed_images: int = 0
    removed_patches: int = 0
    bank_patches_before: int = 0
    images_before: int = 0

    @property
    def patch_share(self) -> float:
        if not self.bank_patches_before:
            return 0.0
        return self.removed_patches / self.bank_patches_before

    @property
    def image_share(self) -> float:
        if not self.images_before:
            return 0.0
        return self.removed_images / self.images_before

    #: 이 비중을 넘게 빼면 커버리지가 흔들릴 수 있다고 본다. 자리표시 값이며
    #: 시나리오로 측정해 정해야 한다.
    WARN_PATCH_SHARE = 0.20

    @property
    def is_heavy(self) -> bool:
        return self.patch_share >= self.WARN_PATCH_SHARE

    def describe(self) -> str:
        if not self.removed_images:
            return ""
        text = (
            f"제거하면 뱅크 이미지 {self.images_before}장 중 {self.removed_images}장"
            f"({self.image_share:.1%}), 패치 {self.bank_patches_before:,}개 중 "
            f"{self.removed_patches:,}개({self.patch_share:.1%})가 빠진다."
        )
        if self.is_heavy:
            text += (
                " 빠지는 양이 많다. 빠진 조건을 보충하지 않으면 뱅크 오염을 고치다"
                " 커버리지 부족을 만들 수 있다."
            )
        return text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "patch_share": self.patch_share,
            "image_share": self.image_share,
            "is_heavy": self.is_heavy,
            "note": self.describe(),
        }


@dataclass
class CurationPlan:
    """뱅크를 어떻게 바꿀 것인가.

    touches_bank 가 False 면 아무것도 하지 않는 계획이다. 그것도 결정이므로
    reason 에 근거를 남긴다.
    """

    touches_bank: bool
    cause: str | None
    remove: list[RemovalCandidate] = field(default_factory=list)
    add: list[AdditionRequest] = field(default_factory=list)
    reason: str = ""
    needs_human: bool = True
    alternative_actions: list[str] = field(default_factory=list)
    coverage_cost: CoverageCost | None = None

    @property
    def is_empty(self) -> bool:
        return not self.remove and not self.add

    def summary(self) -> str:
        if not self.touches_bank:
            return f"뱅크를 건드리지 않는다: {self.reason}"
        parts = []
        if self.remove:
            parts.append(f"제거 {len(self.remove)}장")
        if self.add:
            parts.append(f"보충 조건 {len(self.add)}개")
        return f"뱅크 재구성: {', '.join(parts)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "touches_bank": self.touches_bank,
            "cause": self.cause,
            "summary": self.summary(),
            "reason": self.reason,
            "needs_human": self.needs_human,
            "alternative_actions": self.alternative_actions,
            "remove": [r.to_dict() for r in self.remove],
            "add": [a.to_dict() for a in self.add],
            "coverage_cost": self.coverage_cost.to_dict() if self.coverage_cost else None,
        }


#: 재구성이 답이 아닌 원인에서 대신 권고할 것.
ALTERNATIVES: dict[str, list[str]] = {
    "threshold": ["adjust_threshold"],
    "normal_overlap": ["redefine_criteria", "add_dedicated_detector", "improve_imaging"],
    "equipment_optics": ["request_equipment_check"],
    "criteria": ["redefine_criteria"],
}


def _blocked(cause: str, extra: str = "") -> CurationPlan:
    """뱅크를 건드리지 않는 계획. 왜 건드리지 않는지가 본문이다."""
    label = CAUSE_LABEL_KO.get(cause, cause)
    reason = (
        f"원인이 {label} 이므로 뱅크 재구성은 답이 아니다. "
        f"데이터를 더 넣거나 빼도 해결되지 않으며, 오히려 정상 패치를 잘못 제거하면 "
        f"과검이 늘어난다."
    )
    if extra:
        reason += f" {extra}"
    return CurationPlan(
        touches_bank=False,
        cause=cause,
        reason=reason,
        needs_human=False,
        alternative_actions=list(ALTERNATIVES.get(cause, [])),
    )


def _coverage_cost(bank: MemoryBank | None, images: Sequence[str]) -> CoverageCost | None:
    """이 이미지들을 빼면 뱅크에서 무엇이 함께 빠지는가.

    패치 수까지 세는 이유는 이미지 장수만으로는 비용이 안 보이기 때문이다.
    coreset 이 특정 이미지를 과대 대표하면 5장을 빼도 뱅크의 상당 부분이 빠진다.
    """
    if bank is None:
        return None
    counts = bank.contributing_images()
    total_patches = sum(counts.values())
    removed = sum(counts.get(name, 0) for name in images)
    return CoverageCost(
        removed_images=len([n for n in images if n in counts]),
        removed_patches=removed,
        bank_patches_before=total_patches,
        images_before=len(bank.images),
    )


def plan_curation(
    diagnosis: DiagnosisResult,
    bank: MemoryBank | None = None,
    missed_results: Sequence[InferenceResult] | None = None,
    bank_profile: BankProfile | None = None,
    missing_conditions: dict[str, str] | None = None,
    isolation_z_threshold: float = 1.0,
    max_removals: int = 10,
) -> CurationPlan:
    """진단 결과를 뱅크 조치 계획으로 옮긴다.

    missed_results
        같은 이슈에서 미검출된 이미지들의 추론 결과. 여러 건이 같은 뱅크
        이미지를 최근접으로 지목하면 그 이미지를 의심할 근거가 된다.
        한 장은 우연일 수 있으므로 반복이 근거다.
    missing_conditions
        커버리지 부족일 때 어느 조건이 비었는가. 진단의 판별 6번에서 나온다.
    """
    cause = diagnosis.cause

    # ── 진단이 보류됐으면 계획도 세우지 않는다 ─────────────────────────
    if cause is None:
        return CurationPlan(
            touches_bank=False,
            cause=None,
            reason=(
                "원인이 확정되지 않아 뱅크 조치를 계획하지 않는다. "
                f"{diagnosis.blocking_reason}"
            ),
            needs_human=True,
        )

    # ── 재구성이 답이 아닌 넷은 여기서 막는다 ──────────────────────────
    if not REBUILD_REQUIRED.get(cause, False):
        return _blocked(cause)

    # ── 뱅크 오염 — 무엇을 뺄 것인가 ───────────────────────────────────
    if cause == "bank_contamination":
        candidates: dict[str, RemovalCandidate] = {}

        # 근거 1. 역추적이 반복해서 지목한 이미지
        traced: dict[str, int] = {}
        for result in missed_results or []:
            top = result.top_match
            if top:
                traced[top.bank.source_image] = traced.get(top.bank.source_image, 0) + 1

        # 진단 자체가 지목한 패치도 한 표로 센다
        nearest = diagnosis.evidence_by_item(4)
        if nearest and nearest.usable and isinstance(nearest.value, dict):
            image = nearest.value.get("source_image")
            if image:
                traced[image] = traced.get(image, 0) + 1

        for image, hits in traced.items():
            candidates[image] = RemovalCandidate(
                image=image,
                reason=f"미검출 {hits}건이 이 이미지의 패치를 최근접 정상으로 지목했다",
                traced_hits=hits,
                confirmed_by_vlm=bool(
                    (patch := diagnosis.evidence_by_item(5)) and patch.usable and patch.value == "defect"
                ),
            )

        # 근거 2. 뱅크 안에서 고립된 이미지 — **보강만 한다.**
        #
        # 예전에는 고립도만으로도 후보를 만들었다. VisA 실측에서 후보 1개 중
        # 혼입 이미지가 0개로 나와 그 경로를 닫았다. 뱅크 오염은 패치 단위로 일어나는데
        # 고립도는 이미지 단위 평균이라 정상부 패치에 희석된다. 지목되지 않은
        # 이미지를 고립도만 보고 빼면 멀쩡한 정상품을 버리게 된다.
        if bank is not None and candidates:
            try:
                suspects: list[IsolationScore] = suspect_images(
                    bank, z_threshold=isolation_z_threshold, top_n=max_removals
                )
            except ValueError:
                suspects = []  # 뱅크가 너무 작으면 고립도를 못 잰다
            for score in suspects:
                existing = candidates.get(score.image)
                if existing:
                    existing.isolation_z = score.z_mean
                    existing.reason += f", 뱅크 내 고립도도 높다(z={score.z_mean:+.2f})"

        # 근거가 많은 순으로. 같으면 지목 횟수 순.
        ordered = sorted(
            candidates.values(),
            key=lambda c: (c.evidence_count, c.traced_hits, c.isolation_z or 0.0),
            reverse=True,
        )[:max_removals]

        if not ordered:
            return CurationPlan(
                touches_bank=False,
                cause=cause,
                reason=(
                    "뱅크 오염으로 진단됐으나 제거할 대상을 특정하지 못했다. "
                    "역추적이 반복해서 지목한 이미지가 없으므로 사람이 확인해야 한다. "
                    "고립도는 단독 근거로 쓰지 않는다. 실측에서 혼입 이미지를 짚지 못했다."
                ),
                needs_human=True,
            )

        # 빼는 비용을 센다. 막지는 않고 숫자로 남긴다.
        cost = _coverage_cost(bank, [c.image for c in ordered])
        if bank is not None:
            counts = bank.contributing_images()
            for candidate in ordered:
                candidate.patch_count = counts.get(candidate.image, 0)

        # 근거가 한 갈래뿐이면 사람 확인을 붙인다. 정상 이미지를 잘못 빼면
        # 커버리지 부족을 스스로 만드는 셈이 된다. 빼는 양이 많을 때도 마찬가지다.
        weak = [c for c in ordered if c.evidence_count < 2]
        reason = (
            f"뱅크 오염 후보 {len(ordered)}장을 뱅크에서 제거한 뒤 재구성한다. "
            f"근거가 겹친 것 {len(ordered) - len(weak)}장, 한 갈래뿐인 것 {len(weak)}장."
        )
        if cost and cost.describe():
            reason += f" {cost.describe()}"
        return CurationPlan(
            touches_bank=True,
            cause=cause,
            remove=ordered,
            reason=reason,
            needs_human=bool(weak) or bool(cost and cost.is_heavy),
            coverage_cost=cost,
        )

    # ── 커버리지 부족 — 무엇을 채울 것인가 ─────────────────────────────
    if cause == "coverage_gap":
        if not missing_conditions:
            return CurationPlan(
                touches_bank=False,
                cause=cause,
                reason=(
                    "커버리지 부족으로 진단됐으나 어느 조건이 비었는지 알 수 없다. "
                    "판별 6번의 결과가 필요하다."
                ),
                needs_human=True,
            )

        additions = [
            AdditionRequest(
                condition_key=key,
                condition_value=value,
                reason=f"{key}={value} 조건의 정상 패치가 뱅크 구성에 없다",
            )
            for key, value in missing_conditions.items()
        ]
        known = f" 현재 뱅크는 {bank_profile.source_image_count}장으로 구성돼 있다." if bank_profile else ""
        return CurationPlan(
            touches_bank=True,
            cause=cause,
            add=additions,
            reason=(
                f"비어 있는 조건 {len(additions)}개의 정상 이미지를 보충한 뒤 재구성한다."
                f"{known}"
            ),
            needs_human=False,
        )

    # 도달하지 않아야 한다. 원인이 늘어나면 여기서 걸린다.
    return CurationPlan(
        touches_bank=False,
        cause=cause,
        reason=f"원인 {cause} 에 대한 큐레이션 규칙이 정의되지 않았다.",
        needs_human=True,
    )
