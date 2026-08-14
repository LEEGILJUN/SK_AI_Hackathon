"""라인↔품목 매핑이 한 벌인가.

**한 번 어긋나 있었다.** `data/build_factory.py` 와 시나리오 24건은
`line_01=pcb1 … line_04=pcb4` 인데 `app/pipeline.py` 의 `DEMO_ITEMS` 는
`line_02` 를 capsules 로, 또 `line_02` 를 pcb1 로 쓰고 있었다. 즉 같은
`line_02` 가 세 가지 품목이었다.

**그런데도 아무것도 안 터졌다.** `lookup/mock.py` 가 라인·품목을 안 보고
아무 값이나 돌려주기 때문이다. 조회 계층 실구현(`lookup/factory.py`)이
붙는 순간 `resolve_bank(line, object_name)` 이 없는 조합을 받아 `None` 을
내고 거기서 멈춘다.

**돌아간다는 것과 맞다는 것이 갈리지 않는 자리**라 시험으로 못 박는다.
출처는 `data/build_factory.py` 의 `VALID_LINES` 하나다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def factory_lines() -> dict[str, str]:
    """출처. `data/` 는 패키지가 아니라 importlib 으로 읽는다."""
    spec = importlib.util.spec_from_file_location(
        "build_factory", REPO_ROOT / "data" / "build_factory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.VALID_LINES)


@pytest.fixture(scope="module")
def scenario_targets() -> set[tuple[str, str]]:
    payload = yaml.safe_load((REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    found = set()
    for item in payload.get("scenarios", []):
        target = item.get("target") or {}
        if target.get("line") and target.get("object"):
            found.add((target["line"], target["object"]))
    return found


def test_the_demo_uses_the_factory_mapping(factory_lines):
    """웹 데모의 (라인, 품목)이 공장 구성과 같다."""
    from app.pipeline import DEMO_ITEMS

    for line, object_name, _category in DEMO_ITEMS:
        assert line in factory_lines, f"공장에 없는 라인이다: {line}"
        assert factory_lines[line] == object_name, (
            f"{line} 은 공장에서 {factory_lines[line]} 인데 데모는 {object_name} 을 쓴다"
        )


def test_one_line_means_one_object(factory_lines):
    """한 라인이 두 품목을 갖지 않는다.

    전에 `line_02` 가 capsules 이면서 pcb1 이었다. 라인마다 뱅크가 따로인
    설계에서 이러면 어느 뱅크로 판정할지가 정해지지 않는다.
    """
    from app.pipeline import DEMO_ITEMS

    seen: dict[str, str] = {}
    for line, object_name, _category in DEMO_ITEMS:
        if line in seen:
            assert seen[line] == object_name, (
                f"{line} 에 품목이 둘이다: {seen[line]} · {object_name}"
            )
        seen[line] = object_name


def test_the_category_matches_the_object(factory_lines):
    """VisA 카테고리 이름과 품목 이름이 같다.

    다르면 `resolve_bank` 가 찾는 품목과 실제로 읽은 이미지가 어긋난다.
    """
    from app.pipeline import DEMO_ITEMS

    for line, object_name, category in DEMO_ITEMS:
        assert object_name == category, f"{line}: 품목 {object_name} · 카테고리 {category}"


#: `data/scenarios.yaml` 맨 끝의 스키마 예시 템플릿. 실제 시나리오가 아니라
#: 형식을 보여주려고 둔 것이라 공장 구성에 없는 (라인, 품목)을 쓴다.
#: `load_scenarios` 가 매핑으로 걸러 내므로 채점에는 안 들어간다.
SCHEMA_TEMPLATE = ("line_02", "capsules")


def test_only_the_schema_template_disagrees(factory_lines, scenario_targets):
    """채점 기준이 공장 구성과 같다. 예외는 스키마 예시 하나뿐이다.

    **`data/scenarios.yaml` 은 고치지 않는다**(장영진 소유의 채점 기준).
    어긋나는 것이 하나 더 생기면 여기서 잡아 알려 줄 뿐이다.
    """
    mismatched = sorted(
        (line, obj) for line, obj in scenario_targets
        if factory_lines.get(line) != obj
    )
    assert mismatched == [SCHEMA_TEMPLATE], (
        f"공장 구성과 어긋나는 시나리오: {mismatched}\n"
        f"공장: {factory_lines}"
    )


def test_the_only_duplicate_scenario_id_is_the_template():
    """시나리오 id 가 겹치지 않는다. 겹치는 것은 템플릿 하나뿐이다.

    **`SC-BC-001` 이 두 번 쓰이고 있다** — 실제 시나리오(line_01/pcb1)와 맨
    끝의 스키마 예시(line_02/capsules)가 같은 id 다. 지금은 채점이 전건을
    훑어서 안 터지지만, **id 로 찾는 코드가 하나라도 생기면 둘 중 하나가
    조용히 이긴다.** 장영진 확인이 필요하다.

    새로 겹치는 것이 생기면 여기서 잡힌다.
    """
    from collections import Counter

    payload = yaml.safe_load((REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    counts = Counter(s.get("id") for s in payload.get("scenarios", []))
    duplicated = {key: n for key, n in counts.items() if n > 1}
    assert duplicated == {"SC-BC-001": 2}, (
        f"겹치는 시나리오 id: {duplicated}. 알려진 것은 SC-BC-001 (템플릿) 하나뿐이다"
    )


def test_the_contaminated_item_is_a_real_item(factory_lines):
    """오염을 넣는 자리가 실제 품목이고, 오염 시나리오와 같은 자리다."""
    from app.pipeline import CONTAMINATED_ITEM, DEMO_ITEMS

    line, object_name = CONTAMINATED_ITEM
    assert factory_lines.get(line) == object_name, (
        f"오염 품목 {line}/{object_name} 이 공장 구성에 없다"
    )
    assert (line, object_name) in {(l, o) for l, o, _ in DEMO_ITEMS}, (
        "오염을 넣는 품목이 데모에 없으면 화면에서 그 결과를 볼 수 없다"
    )

    payload = yaml.safe_load((REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    contaminated = {
        (s["target"]["line"], s["target"]["object"])
        for s in payload.get("scenarios", [])
        if (s.get("injection") or {}).get("method") == "bank_contamination"
        and (s.get("target") or {}).get("line")
    }
    assert CONTAMINATED_ITEM in contaminated, (
        f"오염 품목 {CONTAMINATED_ITEM} 을 정답으로 거는 시나리오가 없다. "
        f"있는 것: {sorted(contaminated)}"
    )


def test_the_issue_text_names_the_item_it_is_actually_about(factory_lines):
    """시연 이슈 원문의 라인·품목이 실제 대상과 같다.

    **한 번 어긋나 있었다.** 매핑을 pcb 로 옮긴 뒤에도 `DEFAULT_ISSUE` 가
    "2라인 캡슐"로 남아 있어서, 화면 첫 줄이 `제품 PCB1-01-...` 과 나란히
    떴다. **심사에서 제일 먼저 읽는 자리가 자기모순**이었다.

    문자열이라 다른 시험에 안 걸린다. 여기서 본다.
    """
    from app.pipeline import CONTAMINATED_ITEM, DEFAULT_ISSUE

    line, object_name = CONTAMINATED_ITEM
    number = line.split("_")[-1].lstrip("0") or "0"

    assert f"{number}라인" in DEFAULT_ISSUE, (
        f"이슈 원문이 {number}라인({line}) 건이 아니다: {DEFAULT_ISSUE!r}"
    )
    family = "".join(ch for ch in object_name if not ch.isdigit()).upper()  # pcb1 → PCB
    assert family in DEFAULT_ISSUE.upper(), (
        f"이슈 원문이 {object_name} 을 가리키지 않는다: {DEFAULT_ISSUE!r}"
    )


def test_the_issue_history_uses_real_lines(factory_lines):
    """이슈 이력 그래프가 실제 라인을 쓴다.

    없는 라인의 과거 이슈는 유사 사례로 떠도 확인할 수가 없다. 품목까지
    맞을 필요는 없다 — **라인 재배치 이력**이 있어야 중복 차단이 왜 라인을
    봐야 하는지 보일 수 있다(`ISS-0042`).
    """
    from lookup.mock import ISSUE_GRAPH

    for node in ISSUE_GRAPH:
        assert node["line"] in factory_lines, (
            f"{node['issue_id']}: 공장에 없는 라인 {node['line']}"
        )
        assert node["object_name"] in set(factory_lines.values()), (
            f"{node['issue_id']}: 공장에 없는 품목 {node['object_name']}"
        )
