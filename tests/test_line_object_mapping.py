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


def test_scenario_ids_do_not_collide():
    """시나리오 id 가 겹치지 않는다.

    **한 번 겹쳐 있었다.** 파일 끝의 스키마 예시가 실제 시나리오와 같은
    `SC-BC-001` 을 썼다. 지금은 채점이 전건을 훑어서 안 터지지만, **id 로
    찾는 코드가 하나라도 생기면 둘 중 하나가 조용히 이긴다.**
    예시를 `SC-TEMPLATE-000` 으로 떼어 냈다 — 정답값은 건드리지 않았다.
    """
    from collections import Counter

    payload = yaml.safe_load((REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    counts = Counter(s.get("id") for s in payload.get("scenarios", []))
    duplicated = {key: n for key, n in counts.items() if n > 1}
    assert not duplicated, f"겹치는 시나리오 id: {duplicated}"


def test_the_scenario_count_is_what_scoring_sees():
    """채점에 들어가는 시나리오가 24건이다.

    파일 항목은 25개인데 하나는 스키마 예시다. 이 둘을 헷갈려 문서에
    20 · 24 · 25 세 숫자가 동시에 적혀 있던 적이 있다.
    """
    payload = yaml.safe_load((REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    items = payload.get("scenarios", [])
    real = [s for s in items if not str(s.get("id", "")).startswith("SC-TEMPLATE")]
    assert len(real) == 24, f"채점 대상이 {len(real)}건이다"
    assert len(items) - len(real) == 1, "스키마 예시는 하나여야 한다"


def test_the_contaminated_item_is_a_real_item(factory_lines):
    """혼입 이미지를 넣는 자리가 실제 품목이고, 뱅크 오염 시나리오와 같은 자리다."""
    from app.pipeline import CONTAMINATED_ITEM, DEMO_ITEMS

    line, object_name = CONTAMINATED_ITEM
    assert factory_lines.get(line) == object_name, (
        f"뱅크 오염 품목 {line}/{object_name} 이 공장 구성에 없다"
    )
    assert (line, object_name) in {(l, o) for l, o, _ in DEMO_ITEMS}, (
        "혼입 이미지를 넣는 품목이 데모에 없으면 화면에서 그 결과를 볼 수 없다"
    )

    payload = yaml.safe_load((REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    contaminated = {
        (s["target"]["line"], s["target"]["object"])
        for s in payload.get("scenarios", [])
        if (s.get("injection") or {}).get("method") == "bank_contamination"
        and (s.get("target") or {}).get("line")
    }
    assert CONTAMINATED_ITEM in contaminated, (
        f"뱅크 오염 품목 {CONTAMINATED_ITEM} 을 정답으로 거는 시나리오가 없다. "
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


def test_no_stale_item_name_survives_in_code(factory_lines):
    """공장에 없는 품목 이름이 코드의 문자열에 남아 있지 않다.

    **이 시험이 있는 이유가 있다.** `DEMO_ITEMS` 를 pcb 로 옮겼을 때 매핑은
    맞췄는데 그것을 **가리키는 문자열들이 안 따라왔다** — 이슈 원문이
    "2라인 캡슐"인 채로 제품은 `PCB1-01-...` 이 나갔고, 이슈 이력 그래프도
    화면 양식 예시도 도구 설명도 전부 옛 품목이었다. 매핑 시험은 초록이었다.
    **바꾼 값을 검사하는 것과 그 값을 부르는 곳을 검사하는 것은 다르다.**

    주석과 독스트링은 보지 않는다. 왜 바뀌었는지를 적어 두는 자리이고,
    거기까지 막으면 이력을 못 남긴다. **동작에 쓰이는 문자열만** 본다.
    """
    import ast

    live = set(factory_lines.values())
    # VisA 12 카테고리 중 공장 구성에 없는 것들
    stale = {
        "candle", "capsules", "cashew", "chewinggum", "fryum",
        "macaroni1", "macaroni2", "pipe_fryum",
    } - live

    found: list[str] = []
    for folder in ("app", "agents", "lookup"):
        for path in sorted((REPO_ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            docstrings = {
                id(node.body[0].value)
                for node in ast.walk(tree)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            }
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in docstrings:
                    continue
                for name in stale:
                    if name in node.value:
                        rel = path.relative_to(REPO_ROOT)
                        found.append(f"{rel}:{node.lineno} — {name!r} in {node.value[:60]!r}")

    assert not found, (
        "공장에 없는 품목 이름이 코드 문자열에 남아 있다:\n  " + "\n  ".join(found)
    )


def test_the_bank_name_rule_is_not_two_different_rules():
    """뱅크를 만드는 쪽과 조회하는 쪽이 같은 이름을 쓴다.

    **두 벌이었다.** 데모는 `pcb1-v3`, 조회 계층은 `pcb1-01-v1` 을 썼다.
    그러면 `get_bank_profile(version)` 이 조용히 `None` 을 돌려주고
    **판별 6번 커버리지가 통째로 비게 된다** — 화면에는 그냥 값이 없는
    것처럼 보이고 왜 없는지는 안 나온다.

    규칙은 `lookup.base.bank_version_for` 하나뿐이다.
    """
    from app.pipeline import DEMO_ITEMS, DemoFactory
    from lookup.base import bank_version_for

    try:
        factory = DemoFactory(visa_root=Path(__file__).parent / "_no_visa_here")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    for line, object_name, _category in DEMO_ITEMS:
        made = factory.items[(line, object_name)].bank.version
        assert made == bank_version_for(line, object_name), (
            f"{line}/{object_name}: 데모가 만든 뱅크 이름이 규칙과 다르다"
        )


def test_the_lookup_answers_with_the_same_bank_name():
    """조회 계층이 데모가 만든 뱅크와 같은 이름을 답한다.

    실구현을 끼웠을 때 이름이 갈리면 임계값도 커버리지도 못 찾는다.
    공장 데이터가 없으면 건너뛴다.
    """
    from app.pipeline import DEMO_ITEMS, DemoFactory

    if not (REPO_ROOT / "data" / "manifest.csv").exists():
        pytest.skip("공장 데이터가 아직 없다")

    from lookup.factory import FactoryLookup

    try:
        factory = DemoFactory(visa_root=Path(__file__).parent / "_no_visa_here")
    except RuntimeError as exc:
        pytest.skip(str(exc))

    lookup = FactoryLookup()
    for line, object_name, _category in DEMO_ITEMS:
        made = factory.items[(line, object_name)].bank.version
        profile = lookup.resolve_bank(line, object_name)
        assert profile is not None, f"{line}/{object_name}: 조회가 뱅크를 못 찾는다"
        assert profile.bank_version == made
        assert lookup.get_bank_profile(made) is not None, (
            "만든 이름으로 되물었는데 프로파일이 없다"
        )
