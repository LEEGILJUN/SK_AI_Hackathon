"""모델 연결 점검.

시연 직전에 돌려서 언어 모델과 시각 언어 모델이 실제로 응답하는지 본다.
health_check 만으로는 부족하다. 도구 호출과 이미지 판독은 서버와 모델에
따라 되기도 하고 안 되기도 해서, 각각을 실제로 한 번씩 시켜 봐야 한다.

시연 중에 멈추는 것보다 여기서 걸리는 편이 낫다.

실행:
    .venv/bin/python scripts/check_models.py

설정 (없으면 스텁으로 떨어져 흐름만 확인된다)

로컬 모델
    SHVO_LLM_PROVIDER=openai_compat
    SHVO_LLM_BASE_URL=http://localhost:11434/v1
    SHVO_LLM_MODEL=<모델 이름>
    SHVO_VLM_* 도 같은 형태

Anthropic API
    SHVO_LLM_PROVIDER=anthropic
    SHVO_VLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=<키>          # 키는 환경 변수로만 둔다
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.adapters import (  # noqa: E402
    ChatMessage,
    ModelAdapter,
    ToolSpec,
    build_adapter,
    load_config,
)
from agents.vision import judge_defect_visible  # noqa: E402

OK = "  [통과]"
FAIL = "  [실패]"
SKIP = "  [건너뜀]"

# 도구 호출이 되는지 보려고 쓰는 최소 도구. 실제 조회 계층과는 무관하다.
PROBE_TOOL = ToolSpec(
    name="lookup_threshold",
    description="지정한 라인의 현재 이상 점수 임계값을 조회한다",
    parameters={
        "type": "object",
        "properties": {"line": {"type": "string", "description": "라인 ID"}},
        "required": ["line"],
    },
)


def check_responds(adapter: ModelAdapter) -> bool:
    alive, message = adapter.health_check()
    print(f"{OK if alive else FAIL} 응답 — {message}")
    return alive


def check_json(adapter: ModelAdapter) -> bool:
    """구조화된 출력이 되는가. 진단 근거는 전부 이 형식으로 오간다."""
    if adapter.is_stub:
        print(f"{SKIP} JSON 출력 — 스텁이라 의미 없음")
        return True
    try:
        response = adapter.chat(
            [
                ChatMessage.user(
                    'Return exactly this JSON object and nothing else: '
                    '{"verdict": "normal", "confidence": 0.5, "reason": "test"}'
                )
            ],
            json_object=True,
        )
    except Exception as exc:
        print(f"{FAIL} JSON 출력 — 호출 실패: {exc}")
        return False

    parsed = response.json()
    if parsed.get("verdict") == "normal":
        print(f"{OK} JSON 출력 — 구조를 지켜 응답함")
        return True
    print(f"{FAIL} JSON 출력 — 읽지 못함. 원문: {response.text[:80]!r}")
    return False


def check_tools(adapter: ModelAdapter) -> bool:
    """도구 호출이 되는가. 에이전트 오케스트레이션의 전제다."""
    if adapter.is_stub:
        print(f"{SKIP} 도구 호출 — 스텁이라 의미 없음")
        return True
    try:
        response = adapter.chat(
            [ChatMessage.user("line_02 라인의 현재 임계값이 얼마인지 확인해줘.")],
            tools=[PROBE_TOOL],
        )
    except Exception as exc:
        print(f"{FAIL} 도구 호출 — 호출 실패: {exc}")
        return False

    if response.tool_calls:
        call = response.tool_calls[0]
        print(f"{OK} 도구 호출 — {call.name}({call.arguments})")
        return True

    print(
        f"{FAIL} 도구 호출 — 도구를 부르지 않고 텍스트로 답했다. "
        f"이 모델·서버 조합은 도구 호출을 지원하지 않을 수 있다."
    )
    return False


def check_vision(adapter: ModelAdapter) -> bool:
    """이미지를 실제로 읽는가. 판별 항목 1번과 5번이 여기에 걸려 있다.

    판정 기준은 **틀린 답을 내지 않는가** 다.

      실패  결함에 normal 을 냈다 / 정상에 defect 를 냈다
      실패  호출이 죽어 물어보지도 못했다 (call_failed)
      보류  unknown — 실패가 아니다
      통과  틀린 답이 하나도 없다

    **두 번째 줄이 보류와 갈린다.** 시각 기능이 없는 모델(투영기 미탑재)에
    이미지를 보내면 400 이 떨어지고, vision.py 는 그 예외를 unknown 으로
    되돌린다. verdict 만 보면 "모를 때 지어내지 않았다"와 구분되지 않아
    통과가 떴다. 모델이 모르겠다고 답한 것과 그 일을 아예 못 하는 것은
    다르다. 후자는 판별 1·5번을 그 모델로 못 쓴다는 뜻이다.

    unknown 을 실패로 치면 안 된다. 모를 때 지어내지 않는 것은 설계 의도이고,
    그런 판정은 usable=False 로 빠져 사람에게 넘어간다. 실제로 이 검사가
    정상 이미지에 unknown 을 냈다고 모델을 탈락시킨 적이 있는데, 같은 모델이
    VisA 실데이터에서는 10장 중 9장을 맞혔다. 쓸 수 있는 모델을 "쓰지 마라"고
    판정한 것이다.

    합성 이미지는 체커보드 무늬라 "제품 표면이 아니다"라는 답이 오히려 정확할
    수 있다. 그래서 여기서는 **거짓 판정만** 잡고, 실제 성능은 VisA 로 잰다
    (scripts/measure_trace_crop.py).
    """
    from tests.synthetic import make_defect, make_normal

    if adapter.is_stub:
        print(f"{SKIP} 이미지 판독 — 스텁이라 의미 없음")
        return True

    # 판별 1번(`judge_defect_visible`)의 어휘다 — visible / not_visible.
    cases = [
        ("결함 이미지", make_defect(7), "visible", "not_visible"),
        ("정상 이미지", make_normal(7), "not_visible", "visible"),
    ]

    wrong: list[str] = []
    unreachable: list[str] = []
    held = 0

    for label, image, expected, forbidden in cases:
        judgment = judge_defect_visible(adapter, image)
        mark = "호출 실패" if judgment.call_failed else judgment.verdict
        print(
            f"       {label} → {mark} "
            f"(확신 {judgment.confidence:.2f}) {judgment.reason[:50]}"
        )
        if judgment.call_failed:
            unreachable.append(label)
        elif judgment.verdict == forbidden:
            wrong.append(f"{label}에 {forbidden}")
        elif judgment.verdict == "unknown":
            held += 1

    if unreachable:
        print(
            f"{FAIL} 이미지 판독 — 호출이 실패해 물어보지도 못했다 "
            f"({', '.join(unreachable)}). 시각 기능이 없는 모델일 수 있다 "
            f"(투영기 미탑재). 판별 1·5번을 이 모델로 쓸 수 없다."
        )
        return False

    if wrong:
        print(
            f"{FAIL} 이미지 판독 — 거짓 판정 {len(wrong)}건 ({', '.join(wrong)}). "
            f"판별 1·5번을 이 모델로 쓸 수 없다."
        )
        return False

    if held:
        print(
            f"{OK} 이미지 판독 — 거짓 판정 없음. 보류 {held}건은 실패가 아니다 "
            f"(모를 때 지어내지 않는 것이 설계 의도)"
        )
    else:
        print(f"{OK} 이미지 판독 — 결함과 정상을 구분함")

    print(
        "       합성 이미지는 사전 점검용이다. 실제 판독 성능은 VisA 로 재라 — "
        "scripts/measure_trace_crop.py"
    )
    return True


def main() -> int:
    config = load_config()
    print("설정")
    print(f"  언어 모델      {config.llm.provider} / {config.llm.model or '-'} / {config.llm.base_url or '-'}")
    print(f"  시각 언어 모델  {config.vlm.provider} / {config.vlm.model or '-'} / {config.vlm.base_url or '-'}")

    if config.llm.provider == "stub" and config.vlm.provider == "stub":
        print(
            "\n둘 다 스텁이다. 흐름은 돌지만 판정은 비어 있다.\n"
            "실제 모델을 붙이려면 SHVO_LLM_PROVIDER / SHVO_VLM_PROVIDER 를 "
            "openai_compat(로컬) 또는 anthropic(API) 으로 두라. 로컬은 모델 "
            "이름과 주소가, API 는 ANTHROPIC_API_KEY 가 필요하다."
        )

    results: list[bool] = []

    print("\n언어 모델")
    try:
        llm = build_adapter(config.llm)
        results += [check_responds(llm), check_json(llm), check_tools(llm)]
    except Exception as exc:
        print(f"{FAIL} 어댑터를 만들지 못했다: {exc}")
        results.append(False)

    print("\n시각 언어 모델")
    try:
        vlm = build_adapter(config.vlm)
        results += [check_responds(vlm), check_json(vlm), check_vision(vlm)]
    except Exception as exc:
        print(f"{FAIL} 어댑터를 만들지 못했다: {exc}")
        results.append(False)

    failed = results.count(False)
    print("\n" + "=" * 60)
    if failed:
        print(f"{failed}건 실패. 위 항목을 해결하기 전에는 시연에 쓰지 마라.")
        return 1
    print("모두 통과.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
