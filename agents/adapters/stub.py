"""모델 없이 도는 대체 어댑터.

쓰임은 두 가지다.

  1. 테스트 — 모델 서버 없이 에이전트 로직을 검증한다
  2. 배선 확인 — 모델을 붙이기 전에 흐름이 이어지는지 본다

기본 응답은 "판단하지 않음"이다. 그럴듯한 답을 지어내면 배선 오류가
정상 동작처럼 보이고, 더 나쁘게는 스텁의 답이 진단 근거로 리포트에
올라간다. 그래서 기본값은 판정을 비워 두고, 응답이 필요한 테스트는
scripted 로 명시해 넣는다.

모든 응답에 is_stub=True 가 붙는다. 판정 계층은 이 표시를 보고 근거에서
제외해야 한다.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Sequence

from .base import ChatMessage, ChatResponse, ModelAdapter, ToolCall, ToolSpec

#: 판정할 수 없음을 뜻하는 기본 응답. 판별 항목의 공통 형식을 따른다.
DEFAULT_ANSWER: dict[str, Any] = {
    "verdict": "unknown",
    "confidence": 0.0,
    "reason": "모델이 연결되지 않아 판단하지 않았다 (스텁 응답).",
}


class StubAdapter(ModelAdapter):
    is_stub = True

    def __init__(
        self,
        scripted: Sequence[str | dict[str, Any]] | None = None,
        tool_calls: Sequence[Sequence[ToolCall]] | None = None,
        on_call: Callable[[list[ChatMessage]], None] | None = None,
        name: str = "stub",
    ):
        """
        scripted
            호출 순서대로 돌려줄 응답. 문자열이면 그대로, 딕셔너리면 JSON 으로
            직렬화해 내보낸다. 다 쓰면 마지막 응답을 반복한다.
        tool_calls
            호출 순서대로 돌려줄 도구 호출 목록. 오케스트레이션 검증용이다.
        on_call
            호출될 때마다 받은 메시지를 넘겨준다. 프롬프트 검증에 쓴다.
        """
        self.scripted = list(scripted or [])
        self.scripted_tool_calls = [list(c) for c in (tool_calls or [])]
        self.on_call = on_call
        self.name = name
        self.calls: list[list[ChatMessage]] = []

    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        json_object: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        received = list(messages)
        self.calls.append(received)
        if self.on_call:
            self.on_call(received)

        index = len(self.calls) - 1

        if self.scripted:
            answer = self.scripted[min(index, len(self.scripted) - 1)]
        else:
            answer = DEFAULT_ANSWER
        text = json.dumps(answer, ensure_ascii=False) if isinstance(answer, dict) else answer

        calls: list[ToolCall] = []
        if self.scripted_tool_calls and index < len(self.scripted_tool_calls):
            calls = [
                ToolCall(id=c.id or str(uuid.uuid4()), name=c.name, arguments=c.arguments)
                for c in self.scripted_tool_calls[index]
            ]

        return ChatResponse(text=text, tool_calls=calls, model=self.name, is_stub=True)

    def describe(self) -> str:
        return f"{self.name} (스텁 — 실제 판단 아님)"

    def health_check(self) -> tuple[bool, str]:
        return True, f"{self.describe()} — 항상 응답하지만 판정 근거로 쓸 수 없다"
