"""언어 모델·시각 언어 모델 어댑터 인터페이스.

에이전트는 이 인터페이스만 보고, 뒤에 로컬 모델이 붙었는지 외부 API가
붙었는지 알지 못한다. 설정으로 교체할 수 있어야 폐쇄망 시연이 가능하고,
개발 중에 모델을 바꿔도 에이전트 코드를 고치지 않는다.

노출하는 것은 하나다.

    chat(messages, tools=None, json_schema=None) -> ChatResponse

이미지는 메시지에 담아 보낸다. 별도의 vision 함수를 두지 않는 이유는,
시각 언어 모델도 결국 이미지가 섞인 대화이기 때문이다. 인터페이스가 하나면
어댑터를 하나만 구현하면 된다.
"""

from __future__ import annotations

import base64
import io
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from PIL import Image

Role = Literal["system", "user", "assistant", "tool"]


# ── 주고받는 값 ─────────────────────────────────────────────────────────


@dataclass
class ImagePart:
    """메시지에 붙는 이미지 한 장.

    경로든 PIL 이미지든 받아서, 어댑터가 보낼 때 data URL 로 만든다.
    외부 URL 은 받지 않는다. 폐쇄망에서 동작해야 하고, 이미지 출처가
    저장소 밖으로 나가는 경로를 만들지 않기 위해서다.
    """

    image: str | Path | Image.Image
    detail: str = "auto"

    def to_data_url(self, fmt: str = "PNG") -> str:
        if isinstance(self.image, Image.Image):
            source = self.image
            buffer = io.BytesIO()
            source.convert("RGB").save(buffer, format=fmt)
            raw = buffer.getvalue()
        else:
            path = Path(self.image)
            if not path.exists():
                raise FileNotFoundError(f"이미지를 찾지 못했다: {path}")
            with Image.open(path) as opened:
                buffer = io.BytesIO()
                opened.convert("RGB").save(buffer, format=fmt)
                raw = buffer.getvalue()

        encoded = base64.b64encode(raw).decode("ascii")
        return f"data:image/{fmt.lower()};base64,{encoded}"


@dataclass
class ChatMessage:
    role: Role
    content: str
    images: list[ImagePart] = field(default_factory=list)
    tool_call_id: str | None = None  # role="tool" 일 때 어느 호출에 대한 답인지

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, images: Sequence[ImagePart] | None = None) -> "ChatMessage":
        return cls(role="user", content=content, images=list(images or []))


@dataclass
class ToolSpec:
    """에이전트가 부를 수 있는 도구 하나의 명세.

    판별 항목들이 각각 하나의 도구가 된다. parameters 는 JSON Schema 다.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """모델의 응답.

    is_stub 는 실제 모델이 아니라 대체 구현이 답했다는 표시다. 진단 근거로
    올라가면 안 되는 값을 구분하기 위해 응답에 직접 붙여 둔다.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    is_stub: bool = False
    raw: Any = None

    def json(self) -> dict[str, Any]:
        """응답 본문을 JSON 으로 읽는다.

        모델이 코드 펜스로 감싸는 경우가 흔해서 그것까지 벗겨 낸다.
        구조가 깨지면 예외를 내지 않고 빈 딕셔너리를 돌려주며, 판정 쪽에서
        '읽지 못함'으로 처리하게 한다. 진단이 파싱 오류로 멈추면 안 된다.
        """
        return parse_json_object(self.text)


def parse_json_object(text: str) -> dict[str, Any]:
    """모델 출력에서 JSON 객체를 뽑아낸다. 실패하면 빈 딕셔너리."""
    if not text:
        return {}

    candidate = text.strip()
    if candidate.startswith("```"):
        lines = [ln for ln in candidate.splitlines() if not ln.strip().startswith("```")]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    # 앞뒤에 설명이 붙은 경우 가장 바깥 중괄호 구간만 잘라 다시 시도한다.
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# ── 어댑터 ──────────────────────────────────────────────────────────────


class ModelAdapter(ABC):
    """모든 어댑터가 지켜야 하는 계약."""

    #: 실제 모델이 아니라 대체 구현인가
    is_stub: bool = False

    #: 이 모델이 system 역할 메시지를 실제로 읽는가.
    #:
    #: **False 면 시스템 지시가 사용자 메시지에 실려 나간다.** 4090 의
    #: gemma 모델은 ollama 템플릿에 system 분기가 없어 시스템 프롬프트가
    #: 통째로 버려졌다 — 프롬프트를 고쳐도 모델이 본 적이 없었다.
    #: 환경변수 `SHVO_LLM_NO_SYSTEM=1` 로 켠다.
    carries_system: bool = True

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] | None = None,
        json_object: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """대화를 한 번 주고받는다.

        json_object=True 면 모델에 JSON 객체만 내놓으라고 요구한다. 진단
        근거는 구조화되어야 하므로 판정 호출은 대부분 이 모드를 쓴다.
        """

    @abstractmethod
    def describe(self) -> str:
        """리포트와 로그에 남길 한 줄 설명. 어떤 모델이 판단했는지 기록한다."""

    def health_check(self) -> tuple[bool, str]:
        """지금 이 어댑터가 실제로 응답하는지 확인한다.

        시연 직전에 돌려 모델이 살아 있는지 보는 용도다. 네트워크나 로컬
        서버가 죽어 있으면 여기서 걸러야 시연 중에 멈추지 않는다.
        """
        try:
            response = self.chat(
                [ChatMessage.user("Reply with the single word: ok")], max_tokens=16
            )
        except Exception as exc:  # 연결 실패, 인증 실패, 타임아웃 등
            return False, f"{self.describe()} — 응답 없음: {exc}"
        return True, f"{self.describe()} — 응답 확인 ({response.text.strip()[:40]})"
