"""도구 호출 계층 — 프롬프트에서 재구성까지 (작업 16).

"2라인 뱅크 다시 만들어줘" 같은 자연어 명령이 실제 뱅크 재구성으로 이어지는
경로다. 언어 모델은 **어떤 도구를 어떤 순서로 부를지**를 정하고, 각 도구가
하는 일은 결정론적이다.

이 분리가 중요하다. 언어 모델이 원인을 정하거나 무엇을 제거할지 고르면
진단이 인상 평가가 된다. 언어 모델은 "지금 진단해야겠다", "계획이 나왔으니
실행하자" 같은 **절차 판단**만 하고, 판정 내용은 규칙과 조회에서 나온다.

안전 장치가 둘 있다.

  1. 큐레이션이 뱅크를 건드리지 않기로 한 계획은 재구성 도구가 거부한다.
     언어 모델이 재구성을 부르더라도 실행되지 않는다.
  2. 배포하는 도구는 없다. 새 뱅크를 만들 뿐 실제 판정에 쓰지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .adapters.base import ChatMessage, ModelAdapter, ToolCall, ToolSpec


@dataclass
class ToolResult:
    """도구 실행 한 번의 결과."""

    name: str
    arguments: dict[str, Any]
    output: Any
    ok: bool = True
    error: str = ""

    def to_message(self, call_id: str) -> ChatMessage:
        payload = {"ok": self.ok, "result": self.output} if self.ok else {"ok": False, "error": self.error}
        return ChatMessage(
            role="tool",
            content=json.dumps(payload, ensure_ascii=False, default=str),
            tool_call_id=call_id,
        )


@dataclass
class Tool:
    """언어 모델이 부를 수 있는 도구 하나."""

    spec: ToolSpec
    run: Callable[..., Any]
    #: 뱅크를 바꾸는 도구인가. 기록과 승인 경계를 가르는 표시다.
    mutates_bank: bool = False


class ToolRegistry:
    """도구 모음. 이름으로 찾아 실행한다."""

    def __init__(self, tools: Sequence[Tool] = ()):
        self._tools: dict[str, Tool] = {t.spec.name: t for t in tools}
        #: 실행 이력. 무엇이 언제 불렸는지가 거버넌스 기록의 재료다.
        self.calls: list[ToolResult] = []

    def add(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    @property
    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def execute(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.name)
        if tool is None:
            result = ToolResult(
                name=call.name,
                arguments=call.arguments,
                output=None,
                ok=False,
                error=f"'{call.name}' 도구가 없다. 쓸 수 있는 도구: {sorted(self._tools)}",
            )
            self.calls.append(result)
            return result

        try:
            output = tool.run(**call.arguments)
            result = ToolResult(name=call.name, arguments=call.arguments, output=output)
        except TypeError as exc:
            # 인자가 안 맞는 경우. 모델이 고쳐 부를 수 있게 원문을 돌려준다.
            result = ToolResult(
                name=call.name, arguments=call.arguments, output=None, ok=False,
                error=f"인자가 맞지 않다: {exc}",
            )
        except Exception as exc:
            result = ToolResult(
                name=call.name, arguments=call.arguments, output=None, ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        self.calls.append(result)
        return result

    def mutations(self) -> list[ToolResult]:
        """뱅크를 바꾼 호출만. 승인 요청 문서에 실을 목록이다."""
        return [c for c in self.calls if self._tools.get(c.name, None) and self._tools[c.name].mutates_bank]


# ── 도구 명세 ───────────────────────────────────────────────────────────

DIAGNOSE_SPEC = ToolSpec(
    name="diagnose_issue",
    description=(
        "접수된 미검출 이슈의 원인을 판별 7항목으로 진단한다. "
        "원인 6종 중 하나로 분류하고 근거와 뱅크 재구성 필요 여부를 함께 돌려준다. "
        "근거가 모자라면 판정을 보류한다."
    ),
    parameters={
        "type": "object",
        "properties": {
            "line": {"type": "string", "description": "라인 ID (예: line_02)"},
            "object_name": {"type": "string", "description": "대상 품목 (예: capsules)"},
        },
        "required": ["line", "object_name"],
    },
)

PLAN_SPEC = ToolSpec(
    name="plan_curation",
    description=(
        "진단 결과를 뱅크 조치 계획으로 옮긴다. 뱅크에서 무엇을 빼고 무엇을 채울지 정한다. "
        "재구성이 답이 아닌 원인이면 건드리지 않는 계획을 돌려준다. "
        "진단을 먼저 실행해야 한다."
    ),
    parameters={"type": "object", "properties": {}},
)

REBUILD_SPEC = ToolSpec(
    name="rebuild_bank",
    description=(
        "큐레이션 계획대로 새 메모리 뱅크를 만든다. 계획이 뱅크를 건드리지 않기로 했으면 "
        "실행하지 않는다. 새 뱅크는 후보일 뿐이며 실제 판정에 쓰이지 않는다. "
        "배포하려면 평가 게이트를 통과하고 사람이 승인해야 한다. "
        "계획을 먼저 세워야 한다."
    ),
    parameters={
        "type": "object",
        "properties": {
            "confirm": {
                "type": "boolean",
                "description": "재구성을 실행할지. 계획을 확인한 뒤 true 로 부른다",
            }
        },
        "required": ["confirm"],
    },
)

COMPARE_SPEC = ToolSpec(
    name="compare_banks",
    description="재구성 전후 뱅크의 구성 차이를 비교한다. 재구성을 먼저 실행해야 한다.",
    parameters={"type": "object", "properties": {}},
)


# ── 에이전트 루프 ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an operations agent for an AI visual inspection system in a factory.
A defect was missed by the deployed model. Your job is to work through the tools
available to you and report what happened, in Korean.

Rules you must follow:
- Call diagnose_issue first. Do not guess the cause yourself; the tool decides it.
- If the diagnosis says a bank rebuild is not required, do NOT call rebuild_bank.
  Four of the six causes are not solved by rebuilding, and rebuilding anyway wastes
  effort and can make detection worse.
- Call plan_curation before rebuild_bank. Read the plan before confirming.
- You cannot deploy anything. A rebuilt bank is only a candidate.
- When you are done, summarise in Korean: the cause, the evidence, what you did,
  and what a human must decide next.
"""


@dataclass
class AgentRun:
    """에이전트 한 번의 실행 기록."""

    prompt: str
    final_text: str = ""
    steps: int = 0
    tool_results: list[ToolResult] = field(default_factory=list)
    stopped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "final_text": self.final_text,
            "steps": self.steps,
            "stopped_reason": self.stopped_reason,
            "tools": [
                {"name": r.name, "ok": r.ok, "arguments": r.arguments, "error": r.error}
                for r in self.tool_results
            ],
        }


def run_agent(
    prompt: str,
    adapter: ModelAdapter,
    registry: ToolRegistry,
    max_steps: int = 8,
    system: str = SYSTEM_PROMPT,
) -> AgentRun:
    """자연어 명령을 받아 도구를 호출하며 진행한다.

    max_steps 는 무한 루프 방지용이다. 모델이 같은 도구를 반복해서 부르며
    끝내지 못하는 경우가 있으므로 상한을 둔다.
    """
    run = AgentRun(prompt=prompt)
    messages: list[ChatMessage] = [ChatMessage.system(system), ChatMessage.user(prompt)]

    for step in range(max_steps):
        run.steps = step + 1
        try:
            response = adapter.chat(messages, tools=registry.specs)
        except Exception as exc:
            run.stopped_reason = f"모델 호출 실패: {exc}"
            return run

        if response.is_stub:
            run.stopped_reason = "언어 모델이 연결되지 않아 도구 호출을 진행하지 않았다."
            return run

        if not response.tool_calls:
            run.final_text = response.text
            run.stopped_reason = "모델이 도구를 더 부르지 않고 답을 냈다."
            return run

        messages.append(
            ChatMessage(role="assistant", content=response.text or "")
        )
        for call in response.tool_calls:
            result = registry.execute(call)
            run.tool_results.append(result)
            messages.append(result.to_message(call.id))

    run.stopped_reason = f"{max_steps}단계를 넘겨 중단했다."
    return run
