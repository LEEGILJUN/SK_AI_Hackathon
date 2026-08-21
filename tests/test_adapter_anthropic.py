"""Anthropic 어댑터 — 요청을 어떤 모양으로 보내는가.

**네트워크를 안 쓴다.** 가짜 클라이언트를 끼워 넣고 무엇을 보냈는지만 본다.
키가 있어야 도는 시험은 CI 에서도 맥에서도 못 돌고, 그러면 안 도는 시험이
된다.

여기서 잡으려는 것은 넷이다.

    1. sampling 인자를 안 보낸다        Claude Opus 5 가 400 을 낸다
    2. max_tokens 하한                  사고 토큰이 여기 함께 든다
    3. 도구 결과를 한 덩어리로 묶는다     나눠 보내면 병렬 호출이 죽는다
    4. 거절·잘림을 조용히 넘기지 않는다   둘 다 HTTP 200 으로 온다
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agents.adapters.anthropic_api import MIN_MAX_TOKENS, AnthropicAdapter
from agents.adapters.base import ChatMessage, ImagePart, ToolCall, ToolSpec


# ── 가짜 클라이언트 ─────────────────────────────────────────────────────


@dataclass
class _Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class _Response:
    content: list
    stop_reason: str = "end_turn"
    model: str = "claude-opus-5"
    stop_details: Any = None


class _Messages:
    def __init__(self, response):
        self.response = response
        self.seen: dict = {}

    def create(self, **kwargs):
        self.seen = kwargs
        return self.response


class _Client:
    def __init__(self, response):
        self.messages = _Messages(response)


def _adapter(response, **kwargs):
    client = _Client(response)
    return AnthropicAdapter(client=client, **kwargs), client


TOOL = ToolSpec(
    name="intake_issue",
    description="이슈를 접수한다",
    parameters={"type": "object", "properties": {}, "required": []},
)


# ── 1. sampling 인자 ────────────────────────────────────────────────────


def test_sampling_인자를_보내지_않는다():
    """**보내면 400 이다.** 인터페이스는 temperature 를 받되 버린다.

    로컬 어댑터와 인자를 맞춰야 부르는 쪽이 어댑터를 안 가린다. 그래서
    받기는 하고, 실제 요청에는 안 싣는다.
    """
    adapter, client = _adapter(_Response(content=[_Block("text", text="ok")]))
    adapter.chat([ChatMessage.user("안녕")], temperature=0.7)

    sent = client.messages.seen
    for banned in ("temperature", "top_p", "top_k", "seed"):
        assert banned not in sent, f"{banned} 를 보내면 Claude Opus 5 가 거절한다"


# ── 2. max_tokens 하한 ──────────────────────────────────────────────────


def test_max_tokens_에_하한을_둔다():
    """사고 토큰이 이 한도에 함께 든다.

    로컬 어댑터의 기본값 1024 를 그대로 쓰면 응답이 잘리는데, **오류가
    아니라 빈 응답으로** 온다. 도구 호출이 안 나오고 아무 일도 안 일어난다.
    """
    adapter, client = _adapter(_Response(content=[]), max_tokens=1024)
    adapter.chat([ChatMessage.user("안녕")], max_tokens=512)

    assert client.messages.seen["max_tokens"] >= MIN_MAX_TOKENS


# ── 3. 대화를 옮기는 모양 ───────────────────────────────────────────────


def test_system_은_메시지가_아니라_최상위_인자다():
    adapter, client = _adapter(_Response(content=[_Block("text", text="ok")]))
    adapter.chat([ChatMessage.system("너는 검사 에이전트다"), ChatMessage.user("시작")])

    sent = client.messages.seen
    assert "너는 검사 에이전트다" in sent["system"]
    assert [m["role"] for m in sent["messages"]] == ["user"]


def test_도구_결과를_user_메시지_하나에_모은다():
    """**나눠 보내면 병렬 호출이 죽는다.**

    한 턴에 도구를 둘 부르면 결과도 한 덩어리로 돌려줘야 한다. 우리
    파이프라인은 하나씩 부르지만, 모델이 둘을 함께 부르는 순간 여기가 갈린다.
    """
    adapter, client = _adapter(_Response(content=[_Block("text", text="ok")]))
    adapter.chat([
        ChatMessage.user("시작"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="a", name="intake_issue", arguments={}),
            ToolCall(id="b", name="lookup_mes", arguments={}),
        ]),
        ChatMessage(role="tool", content='{"ok":true}', tool_call_id="a"),
        ChatMessage(role="tool", content='{"ok":true}', tool_call_id="b"),
    ], tools=[TOOL])

    sent = client.messages.seen
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["user", "assistant", "user"], f"메시지가 {roles} 로 나뉘었다"

    results = [b for b in sent["messages"][2]["content"] if b["type"] == "tool_result"]
    assert [b["tool_use_id"] for b in results] == ["a", "b"]


def test_도구만_부른_턴에_빈_텍스트를_넣지_않는다():
    """빈 텍스트 블록은 거절된다. content 가 빈 문자열인 턴이 그 자리다."""
    adapter, client = _adapter(_Response(content=[_Block("text", text="ok")]))
    adapter.chat([
        ChatMessage.user("시작"),
        ChatMessage(role="assistant", content="", tool_calls=[
            ToolCall(id="a", name="intake_issue", arguments={"line": "line_01"}),
        ]),
        ChatMessage(role="tool", content="{}", tool_call_id="a"),
    ])

    assistant = client.messages.seen["messages"][1]
    kinds = [b["type"] for b in assistant["content"]]
    assert kinds == ["tool_use"], f"빈 텍스트가 섞였다: {kinds}"
    assert assistant["content"][0]["input"] == {"line": "line_01"}


def test_이미지를_base64_블록으로_보낸다(tmp_path):
    """**외부 URL 로 안 보낸다.** 폐쇄망에서 돌아야 하고, 이미지 출처가
    저장소 밖으로 나가는 경로를 만들지 않는다."""
    from PIL import Image

    path = tmp_path / "crop.png"
    Image.new("RGB", (8, 8), (120, 120, 120)).save(path)

    adapter, client = _adapter(_Response(content=[_Block("text", text="ok")]))
    adapter.chat([ChatMessage.user("이 조각이 결함인가", images=[ImagePart(path)])])

    blocks = client.messages.seen["messages"][0]["content"]
    image = next(b for b in blocks if b["type"] == "image")
    assert image["source"]["type"] == "base64"
    assert image["source"]["media_type"] == "image/png"
    assert image["source"]["data"]


def test_도구_명세를_input_schema_로_옮긴다():
    adapter, client = _adapter(_Response(content=[_Block("text", text="ok")]))
    adapter.chat([ChatMessage.user("시작")], tools=[TOOL])

    tool = client.messages.seen["tools"][0]
    assert tool["name"] == "intake_issue"
    assert tool["input_schema"] == TOOL.parameters
    assert "function" not in tool, "OpenAI 모양으로 보내면 안 된다"


# ── 4. 조용히 비는 응답 ─────────────────────────────────────────────────


def test_도구_호출을_읽어_온다():
    adapter, _ = _adapter(_Response(
        content=[_Block("tool_use", id="t1", name="diagnose_issue", input={"a": 1})],
        stop_reason="tool_use",
    ))
    response = adapter.chat([ChatMessage.user("시작")], tools=[TOOL])

    assert [c.name for c in response.tool_calls] == ["diagnose_issue"]
    assert response.tool_calls[0].arguments == {"a": 1}
    assert response.is_stub is False


def test_거절을_빈_응답으로_넘기지_않는다():
    """거절은 예외가 아니라 HTTP 200 으로 온다.

    빈 텍스트로 넘기면 판정이 조용히 비고, 화면에는 아무 일도 없던 것처럼
    보인다. 이 저장소가 계속 경계해 온 "조용히 틀리는 답"이다.
    """
    @dataclass
    class _Details:
        category: str = "cyber"

    adapter, _ = _adapter(_Response(content=[], stop_reason="refusal",
                                    stop_details=_Details()))
    response = adapter.chat([ChatMessage.user("시작")])

    assert response.text, "거절이 빈 문자열로 나가면 안 된다"
    assert "거절" in response.text


def test_잘린_응답을_성공처럼_넘기지_않는다():
    adapter, _ = _adapter(_Response(content=[], stop_reason="max_tokens"))
    response = adapter.chat([ChatMessage.user("시작")])

    assert "max_tokens" in response.text


# ── 설정으로 갈아끼워지는가 ─────────────────────────────────────────────


def test_설정으로_어댑터가_바뀐다(monkeypatch):
    """에이전트 코드를 안 고치고 모델을 바꾼다는 것이 어댑터 계층의 전부다."""
    from agents.adapters import build_adapter
    from agents.adapters.config import ModelConfig

    monkeypatch.setenv("ANTHROPIC_API_KEY", "테스트용-가짜-키")
    pytest.importorskip("anthropic", reason="anthropic 패키지가 없으면 건너뛴다")

    adapter = build_adapter(ModelConfig(provider="anthropic"))
    assert adapter.describe().startswith("anthropic:claude-opus-5")
    assert adapter.is_stub is False


def test_provider_이름을_틀리면_바로_알린다(monkeypatch):
    """시연 중에 첫 호출에서 터지면 그 자리에서 발표가 끊긴다."""
    from agents.adapters import load_config

    monkeypatch.setenv("SHVO_LLM_PROVIDER", "anthropic_api")
    with pytest.raises(ValueError, match="anthropic"):
        load_config()
