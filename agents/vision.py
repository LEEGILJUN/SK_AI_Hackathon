"""시각 언어 모델을 쓰는 두 가지 판별.

일곱 판별 항목 중 시각 언어 모델을 쓰는 것은 1번과 5번뿐이다. 나머지는
조회와 계산으로 정확한 값을 얻는다. 이 비중이 뒤집히면 진단이 인상 평가가
되므로, 여기서 하는 일을 두 개로 좁혀 둔다.

  1번  결함이 이미지에 실제로 보이는가
       — 보이지 않으면 애초에 미검출이 아니라 접수 오류일 수 있다

  5번  뱅크의 그 패치가 결함인가 진짜 정상품인가
       — 결함이면 뱅크 오염, 진짜 정상품이면 정상 분포 중첩.
         조치가 정반대이므로 이 판별이 진단 전체를 가른다

두 함수 모두 판정을 지어내지 않는다. 모델이 없거나 응답이 깨지면
verdict="unknown" 을 돌려주고, 근거 계층은 그것을 근거에서 제외한다.
비어 있는 근거가 틀린 근거보다 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from .adapters.base import ChatMessage, ImagePart, ModelAdapter

Verdict = Literal["defect", "normal", "unknown"]

_JSON_RULE = (
    "Answer with a single JSON object and nothing else. "
    "Do not wrap it in code fences. Do not add commentary."
)


@dataclass
class VisionJudgment:
    """시각 판별 하나의 결과.

    usable 이 False 인 결과는 진단 근거로 쓰지 않는다. 모델이 없었거나
    (is_stub), 판단하지 못한 경우(unknown)다.

    call_failed 는 **모델에게 물어보지도 못한 경우**다. 호출이 예외로 죽으면
    verdict 가 unknown 으로 돌아오는데, 그것은 "모델이 모르겠다고 답한 것"과
    전혀 다르다. 앞은 모델의 판단이고 뒤는 모델이 그 일을 못 하는 것이다.
    진단 근거로는 둘 다 못 쓰지만(usable=False), 모델을 검사할 때는 갈라야
    한다. 시각 기능이 없는 모델(투영기 미탑재)에 이미지를 보내면 400 이
    떨어지는데, 그것을 "모를 때 지어내지 않았다"로 세면 통과가 뜬다.
    """

    verdict: Verdict
    confidence: float
    reason: str
    model: str
    is_stub: bool
    raw_text: str = ""
    call_failed: bool = False

    @property
    def usable(self) -> bool:
        return not self.is_stub and self.verdict != "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"usable": self.usable}


def _unknown(
    adapter: ModelAdapter, reason: str, raw: str = "", call_failed: bool = False
) -> VisionJudgment:
    return VisionJudgment(
        verdict="unknown",
        confidence=0.0,
        reason=reason,
        model=adapter.describe(),
        is_stub=adapter.is_stub,
        raw_text=raw,
        call_failed=call_failed,
    )


def _read(response_json: dict[str, Any], allowed: set[str]) -> tuple[Verdict, float, str]:
    """모델 응답에서 판정·확신도·근거를 꺼낸다. 형식이 어긋나면 unknown."""
    verdict = str(response_json.get("verdict", "")).strip().lower()
    if verdict not in allowed:
        return "unknown", 0.0, ""

    try:
        confidence = float(response_json.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    reason = str(response_json.get("reason", "")).strip()
    return verdict, confidence, reason  # type: ignore[return-value]


# ── 판별 항목 1번 ───────────────────────────────────────────────────────


def judge_defect_visible(
    adapter: ModelAdapter,
    image: str | Path | Image.Image,
    reported_defect: str = "",
    context_image: str | Path | Image.Image | None = None,
) -> VisionJudgment:
    """접수된 이미지에 결함이 실제로 보이는가.

    image
        **역추적이 가리킨 자리를 잘라 확대한 조각**을 넣는다. 전체 이미지를
        그대로 주면 못 본다 — VisA 결함은 30~45px 인데 원본이 1500×1000 이다.

        실측(`docs/실험_역추적크롭.md`)에서 갈렸다.

            여유 24px (63×64)    0/10   전부 "무엇을 보는지 모르겠다"
            여유 64px (143×144)  9/10   ← 이 값을 쓴다

    context_image
        조각 주변. 무엇을 보고 있는지가 드러나 판독이 안정된다. 판별 5번과
        같은 방식이다.

    reported_defect
        접수자가 말한 결함 종류. 있으면 프롬프트에 넣되, 그것만 찾도록
        몰아가지는 않는다. 접수자가 잘못 지목했을 수도 있기 때문이다.
    """
    hint = (
        f"The reporter described the defect as: {reported_defect}. "
        "Consider it, but judge what you actually see.\n"
        if reported_defect
        else ""
    )

    prompt = (
        "You are inspecting a product image from a manufacturing line.\n"
        f"{hint}"
        + (
            "The first image is the region flagged by the detector. "
            "The second image shows its surroundings for context.\n"
            if context_image is not None
            else ""
        )
        + "Decide whether a visible surface defect is present.\n"
        '"defect" means you can point to an actual anomaly (scratch, dent, '
        "contamination, crack, discoloration, missing part).\n"
        '"normal" means the surface looks acceptable.\n'
        '"unknown" means the image is too unclear to judge.\n\n'
        "Respond with:\n"
        '{"verdict": "defect|normal|unknown", "confidence": 0.0-1.0, '
        '"reason": "one short sentence naming what you saw and where"}\n'
        f"{_JSON_RULE}"
    )

    parts = [ImagePart(image)]
    if context_image is not None:
        parts.append(ImagePart(context_image))

    try:
        response = adapter.chat(
            [ChatMessage.user(prompt, images=parts)], json_object=True
        )
    except Exception as exc:
        return _unknown(adapter, f"시각 판독 호출이 실패했다: {exc}", call_failed=True)

    verdict, confidence, reason = _read(response.json(), {"defect", "normal", "unknown"})
    if verdict == "unknown" and not reason:
        reason = "모델 응답을 판정으로 읽지 못했다."

    return VisionJudgment(
        verdict=verdict,
        confidence=confidence,
        reason=reason,
        model=response.model or adapter.describe(),
        is_stub=response.is_stub,
        raw_text=response.text,
    )


# ── 판별 항목 5번 ───────────────────────────────────────────────────────


def judge_bank_patch(
    adapter: ModelAdapter,
    patch_image: str | Path | Image.Image,
    context_image: str | Path | Image.Image | None = None,
) -> VisionJudgment:
    """뱅크의 정상 패치가 실제로 정상인가, 잘못 들어간 결함인가.

    이 패치는 **정상으로 등록되어 뱅크에 들어간** 자리다. 그 전제를 프롬프트에
    밝히되, 정상이라고 답하도록 유도하지는 않는다. 유도하면 뱅크 오염을
    영영 찾지 못한다.

    context_image
        패치 주변을 함께 보여주면 무엇을 보고 있는지가 드러나 판독이 안정된다.
        inspection.crop.crop_with_context 로 만든다.

    판정이 갈리는 지점
        defect  → 뱅크 오염. 오염 샘플 제거 후 재구성
        normal  → 정상 분포 중첩. 재구성은 효과가 없다
    """
    prompt = (
        "This image region was registered as NORMAL in a defect-detection "
        "reference set. Your task is to check whether that registration was "
        "correct.\n"
        '"defect" means this region actually contains an anomaly and should '
        "not have been registered as normal.\n"
        '"normal" means it is a genuine good product surface.\n'
        '"unknown" means you cannot tell.\n\n'
        "Being registered as normal is not evidence that it is normal. "
        "Judge only from what you see.\n"
    )
    if context_image is not None:
        prompt += (
            "The first image is the region in question. "
            "The second image shows its surroundings for context.\n"
        )
    prompt += (
        "\nRespond with:\n"
        '{"verdict": "defect|normal|unknown", "confidence": 0.0-1.0, '
        '"reason": "one short sentence"}\n'
        f"{_JSON_RULE}"
    )

    images = [ImagePart(patch_image)]
    if context_image is not None:
        images.append(ImagePart(context_image))

    try:
        response = adapter.chat([ChatMessage.user(prompt, images=images)], json_object=True)
    except Exception as exc:
        return _unknown(adapter, f"패치 판독 호출이 실패했다: {exc}", call_failed=True)

    verdict, confidence, reason = _read(response.json(), {"defect", "normal", "unknown"})
    if verdict == "unknown" and not reason:
        reason = "모델 응답을 판정으로 읽지 못했다."

    return VisionJudgment(
        verdict=verdict,
        confidence=confidence,
        reason=reason,
        model=response.model or adapter.describe(),
        is_stub=response.is_stub,
        raw_text=response.text,
    )


def cause_from_patch_judgment(judgment: VisionJudgment) -> str | None:
    """패치 판독 결과를 원인 코드로 옮긴다.

    판단할 수 없으면 None 을 돌려준다. 여기서 억지로 하나를 고르면
    사람 확인이 필요한 건이 조용히 확정된다.
    """
    if not judgment.usable:
        return None
    if judgment.verdict == "defect":
        return "bank_contamination"
    if judgment.verdict == "normal":
        return "normal_overlap"
    return None
