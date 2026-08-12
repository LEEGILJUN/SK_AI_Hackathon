"""OpenAI 호환 엔드포인트 어댑터.

이 어댑터 하나로 세 경우를 모두 덮는다.

    로컬 vLLM      http://localhost:8000/v1
    로컬 Ollama    http://localhost:11434/v1
    외부 API       https://api.openai.com/v1  등

셋 다 같은 규약을 쓰므로 base_url 과 model 만 바꾸면 된다. 폐쇄망 시연은
로컬 주소를, 개발 중에는 외부 API를 가리키게 두고 설정으로 교체한다.
코드가 갈리지 않는다는 점이 이 선택의 이유다.

주의: 도구 호출 지원은 서버와 모델에 따라 다르다. 로컬 모델을 쓸 때는
health_check 만이 아니라 도구 호출까지 한 번 확인해야 한다
(scripts/check_models.py).
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from .base import ChatMessage, ChatResponse, ModelAdapter, ToolCall, ToolSpec


class OpenAICompatAdapter(ModelAdapter):
    is_stub = False

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 0,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 설치 안내
            raise RuntimeError(
                "openai 패키지가 없다. `pip install -r requirements.txt` 를 실행하라."
            ) from exc

        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        # 재현성 목표 때문에 기본값을 0으로 둔다. 서버가 무시할 수도 있다.
        self.seed = seed

        # 로컬 서버는 키를 요구하지 않지만 클라이언트가 빈 값을 거부하므로
        # 자리표시 문자열을 넣는다.
        self.client = OpenAI(
            api_key=api_key or "not-needed-for-local",
            base_url=base_url,
            timeout=timeout,
        )

    # ── 변환 ────────────────────────────────────────────────────────────

    @staticmethod
    def _to_payload(message: ChatMessage) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "content": message.content,
            }

        if not message.images:
            return {"role": message.role, "content": message.content}

        parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
        for image in message.images:
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image.to_data_url(), "detail": image.detail},
                }
            )
        return {"role": message.role, "content": parts}

    # ── 호출 ────────────────────────────────────────────────────────────

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        json_object: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_payload(m) for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if self.seed is not None:
            request["seed"] = self.seed
        if tools:
            request["tools"] = [t.to_openai() for t in tools]
        if json_object:
            request["response_format"] = {"type": "json_object"}

        completion = self.client.chat.completions.create(**request)
        choice = completion.choices[0].message

        calls: list[ToolCall] = []
        for raw_call in getattr(choice, "tool_calls", None) or []:
            try:
                arguments = json.loads(raw_call.function.arguments or "{}")
            except json.JSONDecodeError:
                # 인자가 깨진 채로 오는 일이 있다. 조용히 버리면 진단이
                # 근거 없이 진행되므로 원문을 남긴다.
                arguments = {"_unparsed": raw_call.function.arguments}
            calls.append(
                ToolCall(
                    id=getattr(raw_call, "id", None) or str(uuid.uuid4()),
                    name=raw_call.function.name,
                    arguments=arguments,
                )
            )

        return ChatResponse(
            text=choice.content or "",
            tool_calls=calls,
            model=self.model,
            is_stub=False,
            raw=completion,
        )

    def describe(self) -> str:
        where = self.base_url or "api.openai.com"
        return f"{self.model} @ {where}"
