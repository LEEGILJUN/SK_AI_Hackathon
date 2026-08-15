"""이슈 이력 그래프 — `data/issue_history.jsonl`.

**이 데이터가 하는 일은 중복 작업 차단 하나다.** 과거가 비슷하다고 이번
원인을 그것으로 정하면 진단이 유사도 맞히기가 된다. 원인은 `decide()` 가
판별 7항목으로 낸다.

중복 차단 조건은 넷이 모두 맞을 때다 — 같은 라인 · 같은 품목 · 같은
결함유형 · 해결됨. 유사도 가중치가 품목 0.45 + 결함 0.40 + 라인 0.15 이라
넷이 다 맞아야 1.00 이 되고, 인테이크가 0.95 이상에서 끊는다.

**데이터가 규칙을 시연할 수 있어야 한다.** 여기서 검사하는 것은 자료형이
아니라 그 시연이 실제로 성립하는가다.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY = REPO_ROOT / "data" / "issue_history.jsonl"

#: 웹 데모가 접수하는 건. `app/pipeline.py` 의 `CONTAMINATED_ITEM` 과 같다.
DEMO = ("line_01", "pcb1", "scratch")


@pytest.fixture(scope="module")
def records() -> list[dict]:
    if not HISTORY.exists():
        pytest.skip(f"{HISTORY.relative_to(REPO_ROOT)} 가 아직 없다")
    found = []
    for raw in HISTORY.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        node = json.loads(raw)
        if "_comment" not in node:
            found.append(node)
    return found


@pytest.fixture(scope="module")
def factory_lines() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "build_factory", REPO_ROOT / "data" / "build_factory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.VALID_LINES)


@pytest.fixture(scope="module")
def lookup():
    from lookup.factory import FactoryLookup

    return FactoryLookup()


def receive_with(lookup, line: str, object_name: str, defect_type: str):
    """접수까지만 돌린다. 로트를 주는 이유는 이미지 없이도 넘어가기 위해서다."""
    from agents.adapters.stub import StubAdapter
    from agents.intake import receive

    return receive(
        "미검 건입니다.",
        StubAdapter(),
        lookup=lookup,
        known={
            "line": line, "object_name": object_name,
            "defect_type": defect_type, "lot": "LOT-AAJ",
        },
    )


# ── 데이터가 성립하는가 ─────────────────────────────────────────────────


def test_every_record_uses_a_real_line_and_object(records, factory_lines):
    """없는 라인·품목의 과거 이슈는 유사 사례로 떠도 확인할 수가 없다."""
    for node in records:
        assert node["line"] in factory_lines, f"{node['issue_id']}: 없는 라인 {node['line']}"
        assert node["object_name"] in set(factory_lines.values()), (
            f"{node['issue_id']}: 없는 품목 {node['object_name']}"
        )


def test_issue_ids_do_not_collide(records):
    from collections import Counter

    counts = Counter(node["issue_id"] for node in records)
    assert not {k: n for k, n in counts.items() if n > 1}


def test_every_cause_appears(records):
    """원인 6종이 다 나온다.

    한 원인만 잔뜩 있으면 "이 이력은 뱅크 오염 전용"으로 보이고, 진단이
    여섯을 가른다는 주장과 어긋난다.
    """
    from agents.ontology import CAUSES

    seen = {node["cause"] for node in records}
    assert seen == set(CAUSES), f"빠진 원인: {set(CAUSES) - seen}"


def test_unresolved_issues_exist(records):
    """미해결 건이 있어야 "해결됨만 중복으로 끊는다"를 시연할 수 있다."""
    assert any(not node["resolved"] for node in records)


# ── 규칙이 실제로 걸리는가 ──────────────────────────────────────────────


def test_the_demo_issue_is_not_blocked(lookup):
    """웹 데모가 접수 단계에서 끊기지 않는다.

    **한 번 이것 때문에 데모 전체가 멈췄다.** 같은 라인·품목·결함의 해결된
    이력이 있으면 인테이크가 중복으로 끊는다. 시연할 건은 그 조합이 없어야
    한다 — 여기서는 `ISS-0044` 가 같은 조합이지만 **미해결**이라 안 끊는다.
    """
    result = receive_with(lookup, *DEMO)
    assert result.verdict == "proceed", (
        f"데모 건이 {result.verdict} 로 끊겼다: {result.note}"
    )


def test_a_different_line_is_related_but_not_duplicate(lookup):
    """다른 라인의 같은 증상은 중복이 아니다.

    **이 규칙의 근거가 실측으로 있다.** pcb1 뱅크로 pcb2 정상을 재면 AUROC
    1.000 으로 완전히 갈린다(`docs/실험_pcbAUROC.md` 3장). 라인마다 뱅크가
    따로이므로 1라인이 뱅크 오염됐다고 2라인도 그렇다는 뜻이 아니다.

    유사도만 보고 끊으면 실제로 있는 문제를 "이미 해결된 건"으로 덮는다.
    """
    line, object_name, defect_type = DEMO
    similar = lookup.find_similar_issues(line, object_name, defect_type)

    cross = [i for i in similar if i.line != line and i.resolved]
    assert cross, "다른 라인의 해결된 사례가 없어 이 규칙을 재지 못한다"
    assert max(i.similarity for i in cross) >= 0.8, (
        "유사도가 낮으면 '높은데도 안 끊는다'를 못 보여준다"
    )
    assert receive_with(lookup, *DEMO).verdict == "proceed"


def test_every_resolved_issue_blocks_its_own_triple(lookup, records):
    """해결된 건은 같은 라인·품목·결함으로 다시 접수하면 전부 끊긴다.

    유사도가 품목 0.45 + 결함 0.40 + 라인 0.15 = 1.00 이 되고 해결됐으므로
    중복이다. **하나라도 안 끊기면 규칙이나 데이터 중 하나가 틀린 것**이라
    전건을 본다 — 한 건만 골라 보면 우연히 통과할 수 있다.
    """
    resolved = [node for node in records if node["resolved"]]
    assert resolved, "해결된 건이 없어 차단을 시연할 수 없다"

    for node in resolved:
        result = receive_with(
            lookup, node["line"], node["object_name"], node["defect_type"]
        )
        assert result.verdict == "duplicate", (
            f"{node['issue_id']} ({node['line']}/{node['object_name']}/"
            f"{node['defect_type']}, 해결됨) 로 다시 접수했는데 {result.verdict} 다"
        )


def test_an_unresolved_issue_never_blocks(lookup, records):
    """미해결 건은 유사도가 1.00 이어도 안 끊는다.

    끊으면 진행 중인 문제가 접수 단계에서 사라진다.
    """
    unresolved = [node for node in records if not node["resolved"]]
    assert unresolved
    for node in unresolved:
        result = receive_with(
            lookup, node["line"], node["object_name"], node["defect_type"]
        )
        if result.verdict == "duplicate":
            assert result.duplicate_of != node["issue_id"], (
                f"{node['issue_id']} 는 미해결인데 중복으로 끊었다"
            )


def test_the_path_and_matched_edges_come_along(lookup):
    """유사도 숫자만 돌려주지 않는다.

    어느 간선이 겹쳐서 비슷하다고 봤는지를 함께 내야 검증할 수 있다.
    화면이 이슈→원인→조치→결과 경로를 그린다.
    """
    similar = lookup.find_similar_issues(*DEMO)
    assert similar

    top = similar[0]
    assert top.matched_on, "겹친 간선이 비었다"
    relations = [edge.relation for edge in top.path]
    for expected in ("발생_라인", "대상_품목", "진단_원인", "조치", "결과"):
        assert expected in relations, f"경로에 {expected} 간선이 없다"
