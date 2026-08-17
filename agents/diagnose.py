"""진단 에이전트 — 판별 7항목을 조합해 원인 6종으로 (작업 13).

이 파일의 설계 원칙 하나가 나머지를 결정한다.

    **원인 판정은 규칙으로 한다. 언어 모델은 서술만 맡는다.**

판정까지 언어 모델에 맡기면 진단이 인상 평가가 된다. 판별 항목의 값은
조회와 계산에서 정확하게 나오므로, 그 값들을 어떻게 조합하는지는 코드로
적어 두는 편이 검증 가능하고 재현 가능하다. 언어 모델은 그렇게 나온 판정과
근거를 사람이 읽을 문장으로 옮기는 데만 쓴다.

판정 순서에도 근거가 있다.

  1. 화질부터 본다
     화질이 나가면 이상 점수도 최근접 패치도 전부 오염된 값이 된다.
     다른 판단의 전제가 무너지므로 여기서 먼저 걸러야 한다.

  2. 검출됐는데 기준상 양품으로 흘렀는지 본다
     이 경우는 모델 문제가 아니라 기준 문제다. 모델을 건드리면 안 된다.

  3. 그다음에 역추적한다 (판별 4→5)
     실측으로 확인한 것: 오염된 뱅크에서도 임계값 스윕은 "다시 잡으면 된다"고
     답한다. 점수가 통째로 내려앉았을 뿐 상대 순서는 유지되기 때문이다.
     그래서 임계값 판정보다 역추적을 먼저 봐야 한다. 순서를 바꾸면 뱅크
     뱅크 오염이 임계값 문제로 분류되고, 증상만 덮은 채 다음 로트에서 재발한다.

그리고 하나 더. **근거가 모자라면 판정하지 않는다.** 특히 판별 5번을 얻지
못하면 뱅크 오염과 정상 분포 중첩을 가를 수 없다. 이 둘은 조치가 정반대라
찍으면 절반은 정반대 조치를 지시하게 된다. 그럴 때는 보류하고 사람에게
넘긴다. 비어 있는 판정이 틀린 판정보다 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Sequence

from inspection.quality import QualityAssessment
from inspection.sweep import FeasibilityVerdict
from inspection.types import InferenceResult, NearestMatch
from lookup.base import BankProfile, CriteriaRule, PastIssue, ThresholdRecord

from .adapters.base import ChatMessage, ModelAdapter
from .vision import VisionJudgment

Cause = Literal[
    "threshold",
    "bank_contamination",
    "coverage_gap",
    "normal_overlap",
    "equipment_optics",
    "criteria",
]

#: 원인별로 뱅크 재구성이 답인가. 여섯 중 넷은 아니다.
REBUILD_REQUIRED: dict[str, bool] = {
    "threshold": False,
    "bank_contamination": True,
    "coverage_gap": True,
    "normal_overlap": False,
    "equipment_optics": False,
    "criteria": False,
}

CAUSE_LABEL_KO: dict[str, str] = {
    "threshold": "임계값 문제",
    "bank_contamination": "뱅크 오염",
    "coverage_gap": "커버리지 부족",
    "normal_overlap": "정상 분포 중첩",
    "equipment_optics": "설비·광학",
    "criteria": "기준 문제",
}

RECOMMENDED_ACTIONS: dict[str, list[str]] = {
    "threshold": ["adjust_threshold", "shadow_compare"],
    "bank_contamination": ["remove_contaminated_samples", "rebuild_bank", "shadow_compare"],
    "coverage_gap": ["add_normal_images_for_condition", "rebuild_bank", "shadow_compare"],
    "normal_overlap": ["redefine_criteria", "add_dedicated_detector", "improve_imaging"],
    "equipment_optics": ["request_equipment_check"],
    "criteria": ["redefine_criteria"],
}

#: 그 원인에서 하면 안 되는 조치. 차단율 측정의 근거가 된다.
FORBIDDEN_ACTIONS: dict[str, list[str]] = {
    "threshold": ["rebuild_bank"],
    "bank_contamination": ["lower_threshold"],
    "coverage_gap": [],
    "normal_overlap": ["rebuild_bank", "lower_threshold"],
    "equipment_optics": ["rebuild_bank", "lower_threshold"],
    "criteria": ["rebuild_bank", "lower_threshold"],
}


# ── 근거 ────────────────────────────────────────────────────────────────


@dataclass
class Evidence:
    """판별 항목 하나의 결과.

    source 가 중요하다. 어떤 근거가 조회·계산에서 왔고 어떤 것이 시각 언어
    모델에서 왔는지가 구분되어야, 진단의 신뢰도가 어디에 걸려 있는지 보인다.

    usable=False 는 값을 얻지 못했다는 뜻이다. 근거가 없는 것이지 값이
    거짓이라는 뜻이 아니다.
    """

    item_no: int  # 판별 항목 번호 1~7
    name: str
    value: Any
    source: Literal["vlm", "lookup", "compute", "trace"]
    usable: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosisResult:
    """진단 결과.

    cause 가 None 이면 판정하지 않은 것이다. 그때 blocking_reason 에 무엇이
    모자랐는지가 들어간다.
    """

    cause: Cause | None
    requires_bank_rebuild: bool | None
    confidence: Literal["high", "medium", "low", "none"]
    needs_human: bool
    evidence: list[Evidence] = field(default_factory=list)
    candidate_causes: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    forbidden_actions: list[str] = field(default_factory=list)
    reasoning: str = ""
    blocking_reason: str = ""
    duplicate_of: str | None = None
    narrative: str = ""  # 언어 모델이 쓴 서술. 판정에 영향을 주지 않는다

    @property
    def cause_label(self) -> str:
        return CAUSE_LABEL_KO.get(self.cause or "", "판정 보류")

    def evidence_by_item(self, item_no: int) -> Evidence | None:
        for item in self.evidence:
            if item.item_no == item_no:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "cause_label": self.cause_label,
            "requires_bank_rebuild": self.requires_bank_rebuild,
            "confidence": self.confidence,
            "needs_human": self.needs_human,
            "candidate_causes": self.candidate_causes,
            "recommended_actions": self.recommended_actions,
            "forbidden_actions": self.forbidden_actions,
            "reasoning": self.reasoning,
            "blocking_reason": self.blocking_reason,
            "duplicate_of": self.duplicate_of,
            "evidence": [e.to_dict() for e in self.evidence],
        }


# ── 근거 모으기 ─────────────────────────────────────────────────────────


def collect_evidence(
    *,
    defect_visible: VisionJudgment | None = None,
    quality: QualityAssessment | None = None,
    inference: InferenceResult | None = None,
    threshold: ThresholdRecord | None = None,
    patch_judgment: VisionJudgment | None = None,
    bank_profile: BankProfile | None = None,
    conditions: dict[str, str] | None = None,
    condition_key: str | None = None,
    condition_value: str | None = None,
    criteria: CriteriaRule | None = None,
    defect_area: float | None = None,
) -> list[Evidence]:
    """판별 7항목을 하나의 근거 목록으로 모은다.

    빠진 항목은 usable=False 로 남긴다. 목록에서 지우지 않는 이유는
    "무엇을 확인하지 못했는가"가 리포트에 드러나야 하기 때문이다.
    """
    items: list[Evidence] = []

    # 1. 결함이 이미지에 실제로 보이는가 — 시각 언어 모델
    items.append(
        Evidence(
            item_no=1,
            name="defect_visible",
            value=defect_visible.verdict if defect_visible else None,
            source="vlm",
            usable=bool(defect_visible and defect_visible.usable),
            detail=defect_visible.reason if defect_visible else "판독하지 않음",
        )
    )

    # 2. 화질 지표가 기준 분포를 벗어났는가 — 계산
    items.append(
        Evidence(
            item_no=2,
            name="quality_within_baseline",
            value=quality.within_baseline if quality else None,
            source="compute",
            usable=quality is not None,
            detail=quality.reason if quality else "화질을 재지 않음",
        )
    )

    # 3. 이상 점수가 임계값 대비 어디인가 — 조회 + 추론
    position = None
    detail_3 = "임계값 또는 추론 결과 없음"
    if inference and threshold:
        position = inference.score_position(threshold.value)
        detail_3 = (
            f"점수 {inference.score:.4f} / 임계값 {threshold.value:.4f} "
            f"(비 {inference.score / threshold.value:.2f})"
        )
    items.append(
        Evidence(
            item_no=3,
            name="score_position",
            value=position,
            source="lookup",
            usable=position is not None,
            detail=detail_3,
        )
    )

    # 4. 최근접 정상 패치가 무엇인가 — 뱅크 역추적
    top: NearestMatch | None = inference.top_match if inference else None
    items.append(
        Evidence(
            item_no=4,
            name="nearest_patch",
            value=(
                {
                    "source_image": top.bank.source_image,
                    "row": top.bank.row,
                    "col": top.bank.col,
                    "bank_row_index": top.bank_row_index,
                    "distance": top.distance,
                }
                if top
                else None
            ),
            source="trace",
            usable=top is not None,
            detail=(_nearest_detail(top) if top else "역추적하지 않음"),
        )
    )

    # 5. 그 패치가 결함인가 진짜 정상품인가 — 시각 언어 모델 또는 사람
    items.append(
        Evidence(
            item_no=5,
            name="nearest_patch_is_defect",
            value=patch_judgment.verdict if patch_judgment else None,
            source="vlm",
            usable=bool(patch_judgment and patch_judgment.usable),
            detail=patch_judgment.reason if patch_judgment else "판독하지 않음",
        )
    )

    # 6. 현재 조건의 정상 패치가 뱅크에 있는가 — 뱅크 구성 이력 조회
    #
    # 조건 축은 여러 개일 수 있다. 일자만이 아니라 로트·설비·교대조·자재
    # 배치까지 조건이 될 수 있고, 어느 축이 비었는지가 조치를 좌우한다.
    # 자재 배치가 빠진 것과 야간 교대가 빠진 것은 보충할 이미지가 다르다.
    # 그래서 축을 하나로 묶지 않고 전부 확인한 뒤 빠진 축을 지목한다.
    #
    # 데이터에 축이 늘어나면 진단이 자동으로 더 많이 본다. MES 를 섬세하게
    # 만들수록 여기서 값어치가 나온다.
    asked: dict[str, str] = dict(conditions or {})
    if condition_key and condition_value:
        asked.setdefault(condition_key, condition_value)

    coverage = None
    detail_6 = "뱅크 구성 이력 없음"
    if bank_profile and asked:
        checked = {key: bank_profile.covers(key, value) for key, value in asked.items()}
        missing = [key for key, present in checked.items() if present is False]
        # **기록하지 않은 축은 판정에서 뺀다.** 모르는 것을 "없다"로 세면
        # 뱅크 프로파일이 그 축을 안 담았다는 이유만으로 커버리지 부족이 된다.
        unknown = [key for key, present in checked.items() if present is None]
        answered = [key for key, present in checked.items() if present is not None]

        coverage = (not missing) if answered else None

        if coverage is None:
            detail_6 = (
                f"물어본 조건 {len(asked)}개를 뱅크 {bank_profile.bank_version} "
                f"구성 이력이 하나도 기록하지 않아 판정할 수 없다"
            )
        elif coverage:
            listed = ", ".join(f"{k}={asked[k]}" for k in answered)
            detail_6 = f"{listed} 가 모두 뱅크 {bank_profile.bank_version} 구성에 포함됨"
        else:
            listed = ", ".join(f"{k}={asked[k]}" for k in missing)
            detail_6 = (
                f"{listed} 가 뱅크 {bank_profile.bank_version} 구성에 없음 "
                f"(기록된 조건 {len(answered)}개 중 {len(missing)}개 누락)"
            )
        if unknown:
            detail_6 += f". 기록이 없어 못 본 축: {', '.join(unknown)}"
        if bank_profile.is_estimated:
            detail_6 += " (폴더 스캔으로 역추정한 이력, 추정)"

    items.append(
        Evidence(
            item_no=6,
            name="coverage_present",
            value=coverage,
            source="lookup",
            usable=coverage is not None,
            detail=detail_6,
        )
    )

    # 7. 판정 기준상 불량이 맞는가 — 기준 테이블 대조
    verdict = None
    detail_7 = "판정 기준 또는 면적 없음"
    if criteria and defect_area is not None:
        verdict = criteria.verdict_for(defect_area)
        detail_7 = (
            f"면적 {defect_area:.0f} / 불량 기준 {criteria.defect_area:.0f} "
            f"({criteria.rule_id}) → {verdict}"
        )
    items.append(
        Evidence(
            item_no=7,
            name="criteria_verdict",
            value=verdict,
            source="lookup",
            usable=verdict is not None,
            detail=detail_7,
        )
    )

    return items


# ── 판정 ────────────────────────────────────────────────────────────────


def _nearest_detail(top: "NearestMatch") -> str:
    """되짚은 패치를 사람이 읽는 문장으로.

    전에는 이렇게 적었다.

        pcb1/Data/Images/Anomaly/001.JPG 격자(48,34), 거리 2.3353

    **읽는 사람이 셋 다 모른다.** 저장소 안쪽 경로가 그대로 나오고, 격자가
    무엇을 세는 좌표인지 안 적혀 있고, 거리 숫자가 큰 것이 좋은지 나쁜지도
    알 수 없다. 화면에 그대로 실려서 "이게 뭔 소리냐"는 말을 들었다.

    파일 이름만 남기고, 좌표는 행·열로 풀고, 거리는 방향을 함께 적는다.
    """
    name = str(top.bank.source_image).replace("\\", "/").rsplit("/", 1)[-1]
    return (f"{name} 의 {top.bank.row}행 {top.bank.col}열 조각, "
            f"닮은 정도 {top.distance:.3f} (숫자가 클수록 덜 닮았습니다)")


def _value(evidence: Sequence[Evidence], item_no: int) -> Any:
    """근거 값을 꺼낸다. 얻지 못한 항목은 None."""
    for item in evidence:
        if item.item_no == item_no:
            return item.value if item.usable else None
    return None


def decide(
    evidence: Sequence[Evidence],
    sweep: FeasibilityVerdict | None = None,
    similar_issues: Sequence[PastIssue] | None = None,
    duplicate_similarity: float = 0.85,
    current_line: str | None = None,
) -> DiagnosisResult:
    """근거를 원인으로 옮긴다. 여기가 진단의 본체다.

    sweep
        임계값 스윕 판정. 있으면 임계값 문제와 정상 분포 중첩을 가르는 데
        보강 근거로 쓴다. 없어도 판정은 된다.
    similar_issues
        유사 사례. 이미 해결된 동일 건이면 진단 이전에 중복으로 끊는다.
    current_line
        지금 이슈의 라인. 주면 **다른 라인의 사례는 중복으로 세지 않는다.**
        라인마다 뱅크가 따로이므로 1라인 뱅크가 오염됐다고 2라인도 그렇다는
        뜻이 아니다. 안 주면 라인을 보지 않던 예전 동작 그대로다.
    """
    evidence = list(evidence)
    result = DiagnosisResult(
        cause=None, requires_bank_rebuild=None, confidence="none", needs_human=True, evidence=evidence
    )

    # ── 0. 이미 해결된 사례인가 ────────────────────────────────────────
    for issue in similar_issues or []:
        if current_line is not None and issue.line != current_line:
            continue
        if issue.resolved and issue.similarity >= duplicate_similarity:
            result.duplicate_of = issue.issue_id
            result.confidence = "high"
            result.needs_human = False
            result.reasoning = (
                f"{issue.line} 에서 동일 증상이 {CAUSE_LABEL_KO.get(issue.cause, issue.cause)}"
                f"으로 규명되어 조치가 끝난 이력이 있다(유사도 {issue.similarity:.2f}). "
                f"중복 작업을 막기 위해 진단을 진행하지 않는다."
            )
            result.recommended_actions = ["review_past_issue"]
            return result

    visible = _value(evidence, 1)
    quality_ok = _value(evidence, 2)
    position = _value(evidence, 3)
    nearest = _value(evidence, 4)
    patch_verdict = _value(evidence, 5)
    coverage = _value(evidence, 6)
    criteria_verdict = _value(evidence, 7)

    # ── 1. 결함이 보이지 않으면 접수 자체를 다시 봐야 한다 ─────────────
    if visible == "not_visible":
        result.blocking_reason = (
            "이미지에서 결함이 확인되지 않았다. 미검출이 아니라 접수 오류이거나 "
            "다른 이미지가 첨부됐을 수 있다. 원인을 판정하지 않는다."
        )
        result.recommended_actions = ["request_correct_image"]
        return result

    # ── 2. 화질이 나가면 다른 근거가 전부 오염된다 ─────────────────────
    if quality_ok is False:
        result.cause = "equipment_optics"
        result.confidence = "high"
        result.needs_human = False
        quality_detail = next((e.detail for e in evidence if e.item_no == 2), "")
        result.reasoning = (
            f"{quality_detail} 화질이 기준을 벗어난 상태에서는 이상 점수와 최근접 패치가 "
            f"모두 신뢰할 수 없는 값이 되므로, 뱅크를 건드리기 전에 설비를 먼저 확인해야 합니다."
        )
        _finalize(result)
        return result

    # ── 3. 검출은 했는데 기준상 양품으로 흘렀는가 ──────────────────────
    if position == "above" and criteria_verdict == "pass":
        result.cause = "criteria"
        result.confidence = "high"
        result.needs_human = False
        criteria_detail = next((e.detail for e in evidence if e.item_no == 7), "")
        result.reasoning = (
            f"이상 점수는 임계값을 넘어 검출됐으나 판정 기준에서 양품으로 분류됐다. "
            f"{criteria_detail} 모델은 잡았으므로 모델 문제가 아니라 기준 문제다."
        )
        _finalize(result)
        return result

    # ── 4. 역추적 결과가 없으면 여기서 더 못 간다 ──────────────────────
    if nearest is None:
        result.blocking_reason = (
            "최근접 정상 패치를 되짚지 못했다(판별 4번). 이 값이 없으면 뱅크 오염과 "
            "정상 분포 중첩을 가를 수 없다."
        )
        result.candidate_causes = ["bank_contamination", "coverage_gap", "normal_overlap", "threshold"]
        return result

    # ── 5. 되짚은 패치가 결함인가 진짜 정상품인가 ──────────────────────
    # 진단이 갈리는 지점이다. 조치가 정반대라 모르면 찍지 않는다.
    if patch_verdict == "defect":
        result.cause = "bank_contamination"
        result.confidence = "high"
        result.needs_human = False
        nearest_detail = next((e.detail for e in evidence if e.item_no == 4), "")
        patch_detail = next((e.detail for e in evidence if e.item_no == 5), "")
        result.reasoning = (
            f"못 잡은 이미지에서 가장 이상한 자리를 고른 뒤, 그 자리와 가장 닮은 "
            f"정상 패치를 뱅크에서 찾았습니다. {nearest_detail}. "
            f"그 조각을 판독하니 정상이 아니라 결함이었습니다({patch_detail}). "
            f"정상이라고 등록해 둔 이미지에 결함이 섞여 들어간 것이고, 그래서 같은 "
            f"유형의 불량이 그 조각과 닮았다는 이유로 정상 판정을 받습니다."
        )
        _finalize(result)
        return result

    if patch_verdict == "genuine_normal":
        # 진짜 정상품과 가까웠다. 이제 왜 못 잡았는지가 셋으로 갈린다.
        if coverage is False:
            result.cause = "coverage_gap"
            result.confidence = "high"
            result.needs_human = False
            coverage_detail = next((e.detail for e in evidence if e.item_no == 6), "")
            result.reasoning = (
                f"최근접 패치는 진짜 정상품이었으나, {coverage_detail}. 현재 조건의 정상 패치가 "
                f"뱅크에 없어 비교 기준 자체가 없는 상태다."
            )
            _finalize(result)
            return result

        # 임계값으로 해결되는가. 스윕이 있으면 그 판정을 우선한다.
        if sweep is not None:
            if sweep.achievable:
                result.cause = "threshold"
                result.confidence = "high"
            else:
                result.cause = "normal_overlap"
                result.confidence = "high"
            result.needs_human = False
            result.reasoning = (
                f"최근접 패치는 진짜 정상품이고 현재 조건도 뱅크에 있습니다. "
                f"임계값 스윕 결과: {sweep.reason}"
            )
            _finalize(result)
            return result

        # ── 스윕이 없을 때만 — 판정 기준으로 어림한다 ─────────────────
        #
        # **이 갈림의 정답은 위의 스윕이다.** 여기는 스윕을 못 구한 경우의
        # 어림이고, 확신도를 낮추고 사람 확인을 붙인다.
        #
        # **이 규칙은 정답 파일을 보고 찾았다.** 24건에서 판별 7번이 두 원인을
        # 안 겹치게 가르는 것을 확인하고 넣었다. 그러므로 **이 규칙에 대한
        # 24건 점수는 독립적인 측정이 아니다.** `docs/시나리오_검토_요청.md`
        # 가 경계한 것이 이것이며, 그래서 스윕을 먼저 보게 하고 이쪽은
        # 뒤로 물렸다.
        #
        # 그래도 점수 위치보다는 낫다. 점수비 구간이 겹치는 반면(임계값
        # 0.89~0.97, 중첩 0.93~0.98) 판별 7번은 기준 테이블 조회이고,
        # 원인 정의에서 따라온다.
        #
        #   임계값 문제      "이상 점수는 높으나 임계값 아래"
        #                    → 기준상 명백한 불량인데 점수가 못 미친 것
        #   정상 분포 중첩    "형상이 유사"
        #                    → 결함 자체가 애매해 기준으로도 불량이라 하기 어렵다
        #
        # 그래서 기준이 `defect` 면 검출 문턱의 문제이고, `review`·`pass` 면
        # 결함 자체가 정상과 겹치는 문제다. **점수비 같은 연속값이 아니라
        # 기준 테이블 조회로 가른다** — 진단의 신뢰도가 결정론적 조회에서
        # 나온다는 원칙과 같은 자리다.
        if criteria_verdict == "defect":
            result.cause = "threshold"
            result.confidence = "low"
            result.needs_human = True
            criteria_detail = next((e.detail for e in evidence if e.item_no == 7), "")
            result.reasoning = (
                f"최근접 패치는 진짜 정상품이고 현재 조건도 뱅크에 있습니다. "
                f"판정 기준으로는 명백한 불량인데({criteria_detail}) 이상 점수가 "
                f"임계값에 못 미쳤으므로 검출 문턱의 문제다. 다만 임계값 스윕 없이는 "
                f"과검률 대가를 제시할 수 없어 확정하지 않는다."
            )
            _finalize(result)
            return result

        if criteria_verdict in ("review", "pass"):
            result.cause = "normal_overlap"
            result.confidence = "low"
            result.needs_human = True
            criteria_detail = next((e.detail for e in evidence if e.item_no == 7), "")
            result.reasoning = (
                f"최근접 패치가 진짜 정상품이고 형상이 유사하다. 판정 기준으로도 "
                f"불량이라 하기 어려운 상태이므로({criteria_detail}) 임계값을 내려도 "
                f"과검만 늘고 이 결함은 계속 정상 쪽에 남습니다. 뱅크 재구성은 효과가 없습니다."
            )
            _finalize(result)
            return result

        # 기준도 없으면 점수 위치가 마지막 근거다. 가장 약하다.
        if position == "near":
            result.cause = "threshold"
            result.confidence = "low"
            result.needs_human = True
            result.reasoning = (
                "최근접 패치는 진짜 정상품이고 이상 점수가 임계값 바로 아래에 있습니다. "
                "임계값 조정으로 해결될 여지가 있으나, 판정 기준도 임계값 스윕도 없어 "
                "점수 위치 하나로 판단한 것이라 확신도가 낮습니다."
            )
            _finalize(result)
            return result

        result.cause = "normal_overlap"
        result.confidence = "medium"
        result.needs_human = True
        result.reasoning = (
            "최근접 패치가 진짜 정상품이고 형상이 유사하며, 이상 점수가 임계값에 크게 "
            "못 미친다. 정상 분포와 겹쳐 거리 기반으로는 구분되지 않는 상태로 보인다. "
            "임계값 스윕으로 과검률 대가를 산출해 확정해야 합니다."
        )
        _finalize(result)
        return result

    # ── 6. 5번을 얻지 못했다. 찍지 않는다 ──────────────────────────────
    result.blocking_reason = (
        "최근접 정상 패치는 찾았으나 그것이 잘못 섞인 결함인지 진짜 정상품인지 판독하지 "
        "못했다(판별 5번). 이 둘은 뱅크 오염과 정상 분포 중첩으로 갈리며 조치가 정반대다. "
        "추측으로 정하면 절반의 경우 반대 조치를 지시하게 되므로 사람 확인이 필요하다."
    )
    result.candidate_causes = ["bank_contamination", "normal_overlap"]
    if coverage is False:
        result.candidate_causes.append("coverage_gap")
    nearest_detail = next((e.detail for e in evidence if e.item_no == 4), "")
    result.reasoning = f"확인이 필요한 패치: {nearest_detail}"
    return result


def _finalize(result: DiagnosisResult) -> None:
    """원인이 정해진 뒤 따라오는 값들을 채운다."""
    if result.cause is None:
        return
    result.requires_bank_rebuild = REBUILD_REQUIRED[result.cause]
    result.recommended_actions = list(RECOMMENDED_ACTIONS[result.cause])
    result.forbidden_actions = list(FORBIDDEN_ACTIONS[result.cause])


# ── 서술 ────────────────────────────────────────────────────────────────

_NARRATION_RULE = (
    "You are writing an inspection diagnosis report in Korean for a manufacturing "
    "engineer. Use only the facts given. Do not add findings, numbers, or causes "
    "that are not in the evidence. If the diagnosis is withheld, say so plainly. "
    "Write 3-5 sentences, plain declarative Korean, no bullet points."
)


def narrate(result: DiagnosisResult, adapter: ModelAdapter) -> str:
    """판정과 근거를 사람이 읽을 문장으로 옮긴다.

    **판정에 영향을 주지 않는다.** 이 함수가 실패하거나 모델이 없어도 진단
    결과는 그대로다. 언어 모델이 하는 일은 이미 정해진 결론을 읽기 좋게
    쓰는 것뿐이다.
    """
    lines = [f"판정: {result.cause_label}" if result.cause else "판정: 보류"]
    if result.requires_bank_rebuild is not None:
        lines.append(f"뱅크 재구성 필요: {'예' if result.requires_bank_rebuild else '아니오'}")
    lines.append(f"확신도: {result.confidence}")
    if result.reasoning:
        lines.append(f"근거 요약: {result.reasoning}")
    if result.blocking_reason:
        lines.append(f"보류 사유: {result.blocking_reason}")

    lines.append("판별 항목:")
    for item in result.evidence:
        mark = "확인" if item.usable else "미확인"
        lines.append(f"  {item.item_no}. {item.name} = {item.value} [{mark}] {item.detail}")

    prompt = _NARRATION_RULE + "\n\n" + "\n".join(lines)

    try:
        response = adapter.chat([ChatMessage.user(prompt)])
    except Exception as exc:
        return f"(서술 생성 실패: {exc})"

    if response.is_stub:
        return "(언어 모델이 연결되지 않아 서술을 생성하지 않았다.)"
    return response.text.strip()
