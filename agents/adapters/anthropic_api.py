"""Anthropic API 어댑터. 공식 SDK(`anthropic`)로 부른다.

**로컬 모델과 같은 인터페이스다.** 에이전트는 `ModelAdapter.chat` 만 보고,
뒤에 4090 의 gguf 가 붙었는지 API 가 붙었는지 모른다. 설정으로 바꾼다.

    SHVO_LLM_PROVIDER=anthropic
    SHVO_LLM_MODEL=claude-opus-5
    ANTHROPIC_API_KEY=...            # 키는 환경 변수로만. 저장소에 안 남긴다

**왜 이 어댑터가 필요한가.** 시연을 심사위원이 직접 눌러 보는데, 그때
언어 모델이 안 붙어 있으면 화면이 "순서를 모델이 정한 것이 아닙니다"로
뜬다. 그것을 지우는 것은 거짓말이고, 그대로 두면 도구 호출을 모델이
정한다는 이 과제의 주장을 심사위원이 확인할 수 없다. 키를 꽂으면 배포판
에서도 모델이 순서를 정한다.

── 로컬 어댑터와 다른 점 셋 ─────────────────────────────────────────────

1. **`temperature` 를 안 보낸다.** Claude Opus 5 는 sampling 인자를 받지
   않는다(400). 재현성은 `seed` 가 아니라 판정을 규칙이 내는 설계에서 온다.
2. **`max_tokens` 에 사고 토큰이 함께 든다.** 도구 호출 하나는 짧지만
   적응형 사고가 그 앞에 붙으므로 1024 로 두면 잘린다. 아래 하한을 둔다.
3. **거절(`stop_reason="refusal"`)이 예외가 아니다.** HTTP 200 으로 온다.
   그것을 빈 응답으로 넘기면 판정이 조용히 비므로 여기서 표시를 남긴다.
"""

from __future__ import annotations

import base64
from typing import Any, Sequence

from .base import ChatMessage, ChatResponse, ImagePart, ModelAdapter, ToolCall, ToolSpec

#: 기본 모델. **바꿀 때는 `docs/모델선정과_VRAM.md` 에 근거를 함께 적는다.**
DEFAULT_MODEL = "claude-opus-5"

#: `max_tokens` 하한. **사고 토큰이 여기 포함된다.**
#:
#: 로컬 어댑터의 기본값 1024 를 그대로 쓰면 적응형 사고가 그 안에서 끝나지
#: 않아 응답이 `stop_reason="max_tokens"` 로 잘린다. 도구 호출이 안 나오고
#: 빈 텍스트가 오므로 **오류가 아니라 조용히 아무 일도 안 일어난다.**
MIN_MAX_TOKENS = 4096

#: 사고 깊이. 도구 하나를 고르는 판단이라 낮게 둔다 — 시연에서 한 단계마다
#: 기다리는 시간이 심사위원 눈에 그대로 보인다. `SHVO_LLM_EFFORT` 로 올린다.
DEFAULT_EFFORT = "medium"


def _image_block(part: ImagePart) -> dict[str, Any]:
    """이미지 한 장을 Anthropic 형식으로.

    `ImagePart` 는 data URL 을 만든다(로컬 OpenAI 호환 서버가 그것을 받는다).
    여기서는 그 안의 base64 만 뽑아 쓴다. **외부 URL 은 애초에 안 받는다** —
    폐쇄망에서 돌아야 하고 이미지 출처가 저장소 밖으로 나가면 안 된다.
    """
    data_url = part.to_data_url()
    header, _, encoded = data_url.partition(",")
    media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": encoded},
    }


def _tool_block(call: ToolCall) -> dict[str, Any]:
    return {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}


def _build_messages(messages: Sequence[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    """우리 대화 표현을 Anthropic 형식으로 옮긴다.

    ── 도구 결과를 한 덩어리로 묶는 것이 중요하다 ─────────────────────

    도구를 여러 개 부른 턴의 결과를 **user 메시지 하나에 모아** 보내야 한다.
    나눠 보내면 모델이 병렬 호출을 그만두게 학습된다. 우리 파이프라인은
    한 번에 하나씩 부르지만, 모델이 둘을 함께 부르는 순간 여기가 갈린다.

    system 은 메시지가 아니라 최상위 인자다. 여러 개면 이어 붙인다.
    """
    system_parts: list[str] = []
    built: list[dict[str, Any]] = []
    pending_results: list[dict[str, Any]] = []

    def flush_results() -> None:
        if pending_results:
            built.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        if message.role == "system":
            if message.content:
                system_parts.append(message.content)
            continue

        if message.role == "tool":
            pending_results.append({
                "type": "tool_result",
                "tool_use_id": message.tool_call_id or "",
                "content": message.content or "",
            })
            continue

        flush_results()

        blocks: list[dict[str, Any]] = [_image_block(part) for part in message.images]
        # **빈 텍스트 블록은 거절된다.** 도구만 부른 턴은 content 가 빈
        # 문자열이라 그대로 넣으면 400 이 난다.
        if message.content:
            blocks.append({"type": "text", "text": message.content})
        if message.role == "assistant":
            blocks.extend(_tool_block(call) for call in message.tool_calls)

        if blocks:
            built.append({"role": message.role, "content": blocks})

    flush_results()
    return "\n\n".join(system_parts), built


class AnthropicAdapter(ModelAdapter):
    """Anthropic API 로 부르는 어댑터.

    `anthropic` 패키지가 없으면 만들 때 바로 알린다. 시연 중에 첫 호출에서
    터지면 그 자리에서 발표가 끊긴다.
    """

    is_stub = False

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout: float = 120.0,
        max_tokens: int = MIN_MAX_TOKENS,
        effort: str = DEFAULT_EFFORT,
        base_url: str | None = None,
        client: Any = None,
    ):
        """
        client
            시험이 가짜 클라이언트를 끼우는 자리다. **없으면 실제 SDK 를
            만든다.** 키는 여기서 받지 않고 환경 변수에서 SDK 가 찾게 두는
            것이 기본이다(`ANTHROPIC_API_KEY`).
        """
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max(int(max_tokens), MIN_MAX_TOKENS)
        self.effort = effort or DEFAULT_EFFORT

        if client is not None:
            self._client = client
            return

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - 설치 안 된 환경
            raise RuntimeError(
                "anthropic 패키지가 없다. `pip install anthropic` 를 하거나 "
                "SHVO_LLM_PROVIDER 를 stub 또는 openai_compat 으로 두라."
            ) from exc

        options: dict[str, Any] = {"timeout": timeout}
        if api_key:
            options["api_key"] = api_key
        if base_url:
            options["base_url"] = base_url
        self._client = anthropic.Anthropic(**options)

    # ── 계약 ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        json_object: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """대화 한 번.

        temperature
            **받기는 하되 안 보낸다.** 인터페이스가 로컬 어댑터와 같아야
            하는데 Claude Opus 5 는 sampling 인자를 거절한다(400). 인자를
            지우면 부르는 쪽이 어댑터마다 달라진다.
        json_object
            JSON 객체만 내놓으라고 요구한다. 지시로 건다 — 응답 형식을
            스키마로 못박는 방법도 있지만, 판정 쪽이 이미 코드 펜스까지
            벗겨 읽으므로(`parse_json_object`) 여기서는 지시로 충분하다.
        """
        system, built = _build_messages(messages)
        if json_object:
            instruction = "답은 JSON 객체 하나만 내라. 설명이나 코드 펜스를 붙이지 마라."
            system = f"{system}\n\n{instruction}" if system else instruction

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(int(max_tokens or self.max_tokens), MIN_MAX_TOKENS),
            "messages": built,
            # 적응형 사고. Opus 5 는 이것이 기본이고 `budget_tokens` 는 거절된다.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.parameters,
                }
                for spec in tools
            ]

        response = self._client.messages.create(**request)
        return self._to_chat_response(response)

    def _to_chat_response(self, response: Any) -> ChatResponse:
        """응답에서 텍스트와 도구 호출을 뽑는다.

        **`stop_reason` 을 먼저 본다.** 거절은 예외가 아니라 200 으로 오고,
        그것을 빈 응답으로 넘기면 판정이 조용히 빈다. 잘린 것도 마찬가지다 —
        `max_tokens` 에 걸리면 도구 호출이 안 나온 채로 성공처럼 보인다.
        """
        stop_reason = getattr(response, "stop_reason", None)

        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in getattr(response, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(getattr(block, "text", "") or "")
            elif kind == "tool_use":
                calls.append(
                    ToolCall(
                        id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )

        text = "".join(texts)
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            text = text or f"모델이 응답을 거절했다(분류: {category or '없음'})."
        elif stop_reason == "max_tokens" and not calls:
            text = text or (
                f"응답이 max_tokens({self.max_tokens})에서 잘렸다. "
                f"사고 토큰이 이 한도에 함께 든다."
            )

        return ChatResponse(
            text=text,
            tool_calls=calls,
            model=str(getattr(response, "model", self.model)),
            is_stub=False,
            raw=response,
        )

    def describe(self) -> str:
        return f"anthropic:{self.model} (effort={self.effort})"
