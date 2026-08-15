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
    #: 이 결과를 받으면 **루프를 끊고 사람에게 돌려준다.**
    #:
    #: 실패가 아니다. 인테이크가 "정보가 부족하다"고 되묻는 것은 설계된
    #: 정상 경로이고, 도구는 성공으로 돌아간다. 그런데 그것을 성공으로만
    #: 보면 **모델이 같은 도구를 같은 인자로 계속 부른다** — 4090 실측에서
    #: `intake_issue` 가 12번 불려 14분 52초를 태웠다. 아무것도 안 바뀐다.
    #:
    #: 사람이 답해야 진행되는 상태라면 여기에 이유를 담아 멈춘다.
    halt: bool = False
    halt_reason: str = ""

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
    #: 출력을 보고 "사람에게 넘겨야 하는가"를 판정한다. (출력) → 이유 또는 "".
    #: 도구가 성공했는데도 더 진행할 수 없는 상태를 잡는 자리다.
    halts_on: Callable[[Any], str] | None = None


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
            reason = tool.halts_on(output) if tool.halts_on else ""
            result = ToolResult(
                name=call.name, arguments=call.arguments, output=output,
                halt=bool(reason), halt_reason=reason,
            )
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
#
# 명세만 여기 둔다. 실행 함수는 부르는 쪽이 붙인다. 같은 도구 이름이 데모와
# 실운영에서 다른 구현을 가리킬 수 있어야 하기 때문이다.

INTAKE_SPEC = ToolSpec(
    name="intake_issue",
    description=(
        "현장에서 올라온 자연어 이슈를 구조화하고 진단으로 넘길지 판단한다. "
        "정보가 모자라면 무엇이 필요한지 돌려주고, 이미 해결된 사례면 여기서 끊는다. "
        "가장 먼저 부를 도구다."
    ),
    parameters={
        "type": "object",
        "properties": {
            "line": {"type": "string", "description": "라인 ID (예: line_01). 1라인 같은 말이 아니다"},
            "object_name": {"type": "string", "description": "품목 (예: pcb1). 제품 ID 를 넣지 않는다"},
            "defect_type": {"type": "string", "description": "결함 유형"},
        },
    },
)

MES_SPEC = ToolSpec(
    name="lookup_mes",
    description=(
        "제품명·로트·라인으로 MES 에서 이미지를 찾고, 그 품목에 배포된 뱅크를 확인한다. "
        "이슈는 이미지가 아니라 제품명이나 로트로 오므로 이 단계가 있어야 무엇을 볼지 정해진다. "
        "**뱅크는 품목마다 다르다** — 캡슐 뱅크로 PCB 를 판정할 수 없다. "
        "조인으로 답하는 결정론적 조회이며 유사도 검색이 아니다. 인테이크 다음에 부른다."
    ),
    parameters={
        "type": "object",
        "properties": {
            "product_id": {"type": "string", "description": "제품 ID (예: PCB1-01-002)"},
            "lot": {"type": "string", "description": "로트 번호 (예: LOT-20260601-001)"},
            "line": {"type": "string", "description": "라인 ID (예: line_01)"},
            "object_name": {"type": "string", "description": "품목 (예: pcb1)"},
        },
    },
)

INSPECT_SPEC = ToolSpec(
    name="run_inspection",
    description=(
        "찾은 이미지를 그 품목의 뱅크로 추론해 미검과 과검을 가려낸다. "
        "설비 판정과 사람이 확인한 값이 갈린 건만 추린다. "
        "여기서 미검이 나와야 진단할 대상이 생긴다. MES 조회를 먼저 해야 한다."
    ),
    parameters={"type": "object", "properties": {}},
)

CHECKS_SPEC = ToolSpec(
    name="run_checks",
    description=(
        "판별 항목 일곱 가지를 모은다 — 결함이 보이는가, 화질이 기준을 벗어났는가, "
        "이상 점수가 임계값 대비 어디인가, 최근접 정상 패치가 무엇인가, 그 패치가 "
        "결함인가, 현재 조건의 정상 패치가 뱅크에 있는가, 기준상 불량이 맞는가. "
        "진단의 입력이며 인테이크 다음에 부른다."
    ),
    parameters={"type": "object", "properties": {}},
)

ONTOLOGY_SPEC = ToolSpec(
    name="lookup_ontology",
    description=(
        "미검출 원인 6종과 판별 7항목의 정의를 조회한다 — 각 원인이 무엇을 뜻하는지, "
        "혼동하기 쉬운 원인과 무엇으로 갈리는지, 어떤 조치가 권고되고 어떤 조치가 금지인지, "
        "뱅크 재구성이 답인지. **이 도구는 원인을 정하지 않는다.** 이번 이슈의 원인은 "
        "diagnose_issue 가 판별 7항목으로 낸다. 여기서 읽은 정의가 그럴듯하다는 이유로 "
        "원인을 고르면 안 된다. 언제든 부를 수 있고 순서 제약이 없다."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cause": {
                "type": "string",
                "description": (
                    "원인 id. threshold | bank_contamination | coverage_gap | "
                    "normal_overlap | equipment_optics | criteria. 비우면 전체 요약"
                ),
            },
            "check_item": {
                "type": "integer",
                "description": "판별 항목 번호 1~7. 비우면 전체 요약",
            },
        },
    },
)

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
            "line": {"type": "string", "description": "라인 ID (예: line_01)"},
            "object_name": {"type": "string", "description": "대상 품목 (예: pcb1)"},
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

# `compare_banks` 는 도구로 내보내지 않는다. `agents/rebuild.py` 에 함수로
# 있고 테스트가 쓰지만, 어느 레지스트리에도 등록되지 않아 모델이 부를 수 없다.
# **정의만 있고 아무 데서도 등록되지 않은 ToolSpec 이 있었다** — 읽는 사람에게
# "이 도구가 있다"고 약속해 놓고 지키지 않는 셈이라 걷어냈다(2026-08-14).
# 재구성 전후 비교를 화면에 내보내기로 정하면 그때 스펙을 다시 만든다.

GATE_SPEC = ToolSpec(
    name="evaluate_gate",
    description=(
        "새 뱅크가 배포 후보가 될 만한지 홀드아웃으로 평가한다. "
        "통과해도 배포되지 않는다 — 사람이 승인해야 한다. 재구성을 먼저 실행해야 한다."
    ),
    parameters={"type": "object", "properties": {}},
)

SHADOW_SPEC = ToolSpec(
    name="shadow_compare",
    description=(
        "신규 뱅크를 실제 판정에 쓰지 않고 같은 이미지에 병렬로만 추론시켜, "
        "기존 뱅크와 판정이 갈리는 케이스만 뽑는다. 양산 데이터에는 정답이 없으므로 "
        "사람이 확인할 이미지 수를 줄이는 것이 목적이다. 게이트 다음에 부른다."
    ),
    parameters={"type": "object", "properties": {}},
)

RELEASE_SPEC = ToolSpec(
    name="prepare_release",
    description=(
        "배포 패키지와 승인 요청 문서를 만든다. **배포하지 않는다.** "
        "실제 장비 반영은 사람이 별도로 결정한다. 게이트와 섀도가 끝나야 한다."
    ),
    parameters={"type": "object", "properties": {}},
)


# ── 에이전트 루프 ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an operations agent for an AI visual inspection system in a factory.
A defect was missed by the deployed model. Work through the tools and report in Korean.

## Follow the pipeline. Do not skip and do not go back.

    intake_issue -> lookup_mes -> run_inspection -> run_checks -> diagnose_issue
                 -> plan_curation -> rebuild_bank -> evaluate_gate
                 -> shadow_compare -> prepare_release

**Every tool result contains a "next" field naming the tool to call next. Call
that tool. Never call the same tool twice.** A result with "ok": true succeeded
even if it looks incomplete to you — calling it again returns the same thing and
wastes a turn. If "next" tells you to stop, stop and summarise.

## Argument format

Use canonical IDs, not the words from the issue text.

    line: "line_01"              not "1라인"
    object_name: "pcb1"          not "PCB 기판", and never a product ID
    product_id: "PCB1-01-002"    the specific item
    lot: "LOT-20260601-001"

Convert Korean line names to IDs. Leave a field out rather than guessing it.

## Rules

- Do not decide the cause yourself. diagnose_issue decides it from the seven
  checks, so call it only after run_checks has produced them.
- If you are unsure what a cause means, what separates it from a similar one, or
  which actions are forbidden for it, call lookup_ontology. It describes the six
  causes and the seven checks. It never decides the cause of this issue.
- If the diagnosis says a bank rebuild is not required, do NOT call rebuild_bank.
  Most of the six causes are not solved by rebuilding, and rebuilding anyway wastes
  effort and can make detection worse. lookup_ontology tells you which ones.
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


#: 한 도구를 이만큼 부르면 인자가 달라도 멈춘다.
#:
#: 모델이 인자를 조금씩 바꿔 가며 같은 도구를 반복하면 위의 "같은 인자" 검사에
#: 안 걸린다. 정상 흐름에서 한 도구를 세 번 부를 일은 없다 — 도구 결과가
#: `next` 로 다음 단계를 알려 주기 때문이다.
REPEAT_TOOL_LIMIT = 3


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

        # **무엇을 불렀는지 함께 남긴다.** 도구만 부른 턴은 content 가 비어
        # 있어서, 이것을 빼면 뒤따르는 tool 메시지가 붕 뜬다.
        messages.append(
            ChatMessage(role="assistant", content=response.text or "",
                        tool_calls=list(response.tool_calls))
        )
        for call in response.tool_calls:
            result = registry.execute(call)
            run.tool_results.append(result)
            messages.append(result.to_message(call.id))

            # ── 도구가 "사람이 답해야 한다"고 말하면 거기서 끝낸다 ──────
            #
            # 실패가 아니다. 인테이크가 정보 부족으로 되묻는 것은 설계된
            # 정상 경로다. 그런데 성공으로만 보면 **모델이 같은 도구를 같은
            # 인자로 계속 부른다** — 4090 실측에서 12번 불려 14분 52초를
            # 태웠고 아무것도 안 바뀌었다.
            if result.halt:
                run.final_text = result.halt_reason
                run.stopped_reason = result.halt_reason
                return run

            # ── 같은 도구를 같은 인자로 또 불렀는데 결과도 같으면 끊는다 ─
            #
            # 위의 halt 로 잡히지 않는 경우까지 막는 그물이다. 아무것도
            # 바뀌지 않는 호출을 반복하는 것은 진행이 아니다.
            repeats = [
                r for r in run.tool_results
                if r.name == result.name and r.arguments == result.arguments
            ]
            if len(repeats) >= 2:
                run.stopped_reason = (
                    f"'{result.name}' 을 같은 인자로 {len(repeats)}번 불렀고 결과가 "
                    f"바뀌지 않았다. 더 진행해도 달라질 것이 없어 멈춘다."
                )
                return run

            # ── 인자가 흔들려도 같은 도구를 계속 부르면 끊는다 ──────────
            #
            # 위 검사는 인자가 **똑같을** 때만 걸린다. 4090 실측에서 모델이
            # defect_type 을 '미세 스크래치' → 'micro-scratch' 로 바꿔 가며
            # 같은 도구를 불러 한 바퀴를 더 돌았다. 진행이 아니라 제자리다.
            same_tool = [r for r in run.tool_results if r.name == result.name]
            if len(same_tool) >= REPEAT_TOOL_LIMIT:
                run.stopped_reason = (
                    f"'{result.name}' 을 {len(same_tool)}번 불렀다. 인자만 바뀌고 "
                    f"다음 단계로 가지 못하고 있어 멈춘다."
                )
                return run

    run.stopped_reason = f"{max_steps}단계를 넘겨 중단했다."
    return run
