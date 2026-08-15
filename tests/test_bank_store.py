"""뱅크 저장소 — 보관·비교·원복, 그리고 전환은 사람이 한다."""

from __future__ import annotations

import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from inspection.bank import MemoryBank
from inspection.features import FeatureConfig
from inspection.store import (
    CURRENT_FILE,
    config_id,
    current_bank,
    folder_name,
    list_banks,
    load_current,
    save_bank,
    write_current,
)
from lookup.base import bank_item_key, bank_version_for

REPO_ROOT = Path(__file__).resolve().parent.parent

CONFIG = FeatureConfig(backbone="resnet18", crop=64)
OTHER = FeatureConfig(backbone="resnet18", crop=128)


def make_bank(version: str, config: FeatureConfig = CONFIG) -> MemoryBank:
    return MemoryBank(
        embeddings=np.zeros((4, 8), dtype=np.float32),
        origins=np.zeros((4, 3), dtype=np.int32),
        images=["a.png", "b.png"],
        meta={
            "bank_version": version,
            "grid": [2, 2],
            "feature_config": config.fingerprint(),
        },
    )


# ── 이름 규칙 ──────────────────────────────────────────────────────────


def test_the_item_key_is_the_same_rule_as_the_bank_version():
    """저장소 폴더와 뱅크 이름이 같은 규칙에서 나온다.

    두 벌이 되면 저장한 자리와 찾는 자리가 갈린다. 예전에 `pcb1-v3` 대
    `pcb1-01-v1` 로 갈려 판별 6번이 통째로 비었던 적이 있다.
    """
    key = bank_item_key("line_01", "pcb1")
    assert key == "pcb1-01"
    assert bank_version_for("line_01", "pcb1").startswith(key + "-")


def test_the_folder_name_carries_version_time_and_config():
    name = folder_name("v2", datetime(2026, 8, 15, 14, 30), CONFIG)
    version, stamp, fingerprint = name.split("_")

    assert version == "v2"
    assert stamp == "20260815-1430"
    assert fingerprint == config_id(CONFIG)


def test_a_different_input_size_gives_a_different_name():
    """이름만 보고 못 쓰는 뱅크를 거를 수 있어야 한다.

    448 에서 512 로 바꿨을 때 옛 뱅크가 검사를 통과해 조용히 틀린 점수를
    낸 적이 있다. 지문이 이름에 있으면 폴더를 열기 전에 갈린다.
    """
    assert config_id(CONFIG) != config_id(OTHER)


# ── 저장과 목록 ────────────────────────────────────────────────────────


def test_saving_does_not_change_what_is_in_use(tmp_path):
    """**저장과 전환은 다른 일이다.**

    새로 만든 뱅크가 저장만으로 판정에 쓰이면 게이트도 섀도도 지나지 않은
    것이 운영에 들어간다.
    """
    save_bank(make_bank("v2"), "pcb1-01", root=tmp_path, built_at=datetime(2026, 8, 15, 14, 30))

    assert current_bank("pcb1-01", root=tmp_path) is None
    assert not (tmp_path / "pcb1-01" / CURRENT_FILE).exists()


def test_versions_come_back_newest_first(tmp_path):
    for number, hour in ((1, 9), (2, 10), (10, 11)):
        save_bank(make_bank(f"v{number}"), "pcb1-01", root=tmp_path,
                  built_at=datetime(2026, 8, 15, hour, 0))

    found = [b.version for b in list_banks("pcb1-01", root=tmp_path)]
    assert found == ["v10", "v2", "v1"], "v10 이 v2 보다 뒤에 오면 문자열로 정렬한 것이다"


def test_banks_built_with_another_setting_are_filtered_out(tmp_path):
    save_bank(make_bank("v1"), "pcb1-01", root=tmp_path, built_at=datetime(2026, 8, 15, 9, 0))
    save_bank(make_bank("v2", OTHER), "pcb1-01", root=tmp_path,
              built_at=datetime(2026, 8, 15, 10, 0), config=OTHER)

    usable = list_banks("pcb1-01", root=tmp_path, config=CONFIG)
    assert [b.version for b in usable] == ["v1"]


def test_a_hand_made_folder_does_not_break_the_listing(tmp_path):
    """사람이 손으로 만든 폴더가 섞여도 무너지지 않아야 한다."""
    save_bank(make_bank("v1"), "pcb1-01", root=tmp_path, built_at=datetime(2026, 8, 15, 9, 0))
    (tmp_path / "pcb1-01" / "백업_옛날것").mkdir()

    assert [b.version for b in list_banks("pcb1-01", root=tmp_path)] == ["v1"]


def test_an_unknown_item_is_an_empty_list_not_an_error(tmp_path):
    assert list_banks("없는품목-99", root=tmp_path) == []
    assert current_bank("없는품목-99", root=tmp_path) is None
    assert load_current("없는품목-99", root=tmp_path) is None


# ── 전환과 원복 ────────────────────────────────────────────────────────


def test_rolling_back_only_moves_the_pointer(tmp_path):
    """**원복이 파일 이동이 아니다.** 가리키는 이름만 되돌린다.

    파일을 옮기면 도중에 죽었을 때 어느 것이 운영본인지 알 수 없게 된다.
    """
    first = save_bank(make_bank("v1"), "pcb1-01", root=tmp_path,
                      built_at=datetime(2026, 8, 15, 9, 0))
    second = save_bank(make_bank("v2"), "pcb1-01", root=tmp_path,
                       built_at=datetime(2026, 8, 16, 9, 0))

    write_current("pcb1-01", second.name, root=tmp_path)
    assert current_bank("pcb1-01", root=tmp_path).version == "v2"

    write_current("pcb1-01", first.name, root=tmp_path)
    assert current_bank("pcb1-01", root=tmp_path).version == "v1"

    # 되돌린 뒤에도 v2 는 그대로 있다 — 다시 앞으로 갈 수 있어야 한다.
    assert second.is_dir()
    assert {b.version for b in list_banks("pcb1-01", root=tmp_path)} == {"v1", "v2"}


def test_pointing_at_a_missing_version_says_what_is_there(tmp_path):
    save_bank(make_bank("v1"), "pcb1-01", root=tmp_path, built_at=datetime(2026, 8, 15, 9, 0))

    with pytest.raises(FileNotFoundError) as caught:
        write_current("pcb1-01", "v9_20260101-0000_abcdef", root=tmp_path)

    assert "v1_20260815-0900" in str(caught.value), "있는 것을 알려줘야 사람이 고른다"


def test_load_current_returns_none_when_the_setting_changed(tmp_path):
    """설정이 바뀌면 **오류가 아니라 None** 이다.

    쓸 만한 것이 없다는 것은 "새로 세워야 한다"는 뜻이지 고장이 아니다.
    """
    saved = save_bank(make_bank("v1"), "pcb1-01", root=tmp_path,
                      built_at=datetime(2026, 8, 15, 9, 0))
    write_current("pcb1-01", saved.name, root=tmp_path)

    assert load_current("pcb1-01", root=tmp_path, config=CONFIG) is not None
    assert load_current("pcb1-01", root=tmp_path, config=OTHER) is None


def test_current_bank_reports_a_stale_pointer_rather_than_hiding_it(tmp_path):
    """가리키는 것은 있는데 못 쓰는 상태를 구분해야 한다.

    "가리키는 것이 없다"(최초 구성)와 "가리키는데 설정이 다르다"(재구성
    필요)는 다음 할 일이 다르다.
    """
    saved = save_bank(make_bank("v1"), "pcb1-01", root=tmp_path,
                      built_at=datetime(2026, 8, 15, 9, 0))
    write_current("pcb1-01", saved.name, root=tmp_path)

    stored = current_bank("pcb1-01", root=tmp_path)
    assert stored is not None
    assert stored.matches(OTHER) is False


# ── 경계 ───────────────────────────────────────────────────────────────


def _referenced_names(package: str) -> set[str]:
    names: set[str] = set()
    for path in (REPO_ROOT / package).rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
    return names


@pytest.mark.parametrize("package", ["agents", "scheduler"])
def test_agents_never_switch_the_bank_in_use(package):
    """**전환은 사람이 한다.**

    `CURRENT` 가 가리키는 것이 실제 판정에 쓰인다. 에이전트가 그것을 바꾸면
    무인 배포다. 릴리즈가 승인 요청 문서까지만 만드는 것과 같은 경계이고,
    자동 실행되는 스케줄러 쪽이 더 위험하다.
    """
    assert "write_current" not in _referenced_names(package), (
        f"{package}/ 에서 뱅크 전환을 부르고 있다. 저장(save_bank)은 자동이지만 "
        f"전환은 사람이 실행하는 스크립트에서만 한다."
    )


# ── 가상 공장이 저장소를 쓴다 ──────────────────────────────────────────


def _factory(store, **kwargs):
    from app.pipeline import DemoFactory

    return DemoFactory(visa_root="/tmp/_no_visa_here", store_root=store, **kwargs)


def test_the_second_start_loads_instead_of_rebuilding(tmp_path):
    """**같은 뱅크를 두 번 세우지 않는다.**

    VisA 로 서면 뱅크 셋 구성에 4090 실측 108초가 든다. 프로세스마다 새로
    세울 이유가 없다.
    """
    first = _factory(tmp_path)
    assert first.loaded_from_store == [], "처음에는 저장소가 비어 있어야 한다"

    second = _factory(tmp_path)

    assert second.loaded_from_store, "두 번째는 저장소에서 불러와야 한다"
    for key, item in first.items.items():
        assert len(second.items[key].bank) == len(item.bank)


def test_a_changed_setting_rebuilds_rather_than_loading_a_wrong_bank(tmp_path):
    """설정이 다르면 불러오지 않고 다시 세운다.

    거리 척도가 다른 뱅크를 쓰면 점수가 조용히 틀린다. 448 에서 512 로
    바꿨을 때 옛 뱅크가 검사를 통과한 적이 있다.
    """
    from inspection.features import FeatureConfig as FC

    _factory(tmp_path)
    item_key = bank_item_key("line_01", "pcb1")

    assert list_banks(item_key, root=tmp_path), "처음에 저장은 됐어야 한다"
    assert load_current(item_key, root=tmp_path, config=FC(backbone="resnet18", crop=32)) is None, (
        "설정이 다른데 불러오면 거리 척도가 다른 뱅크로 판정하게 된다"
    )


def test_the_factory_never_pulls_an_approved_rebuild_back_to_v1(tmp_path):
    """**승인해 넘긴 판을 가상 공장이 무르지 않는다.**

    진단 뒤 재구성한 뱅크는 v2 부터다. `CURRENT` 가 그것을 가리키고 있으면
    사람이 승인해 넘긴 것이고, 공장이 v1 로 되돌리면 승인을 무르는 셈이다.
    """
    _factory(tmp_path)
    item_key = bank_item_key("line_01", "pcb1")

    approved = save_bank(make_bank("v2"), item_key, root=tmp_path,
                         config=CONFIG, built_at=datetime(2026, 8, 16, 9, 0))
    write_current(item_key, approved.name, root=tmp_path)

    _factory(tmp_path)

    assert current_bank(item_key, root=tmp_path).version == "v2", (
        "가상 공장이 승인된 v2 를 v1 로 되돌렸다"
    )


def test_the_real_version_format_increments(tmp_path):
    """**실제 버전 형식에서 판 번호가 올라간다.**

    `_next_version` 이 이름 전체를 `v3` 형태로 보고 `startswith("v")` 로
    걸렀는데, 실제 값은 `pcb1-01-v1` 이라 한 번도 안 걸렸다. 재구성할 때마다
    `-rebuilt` 가 붙어 `pcb1-01-v1-rebuilt-rebuilt` 가 됐고, 판 번호로 앞뒤를
    가릴 수 없어 **원복 대상이 정해지지 않았다.**
    """
    from agents.rebuild import _next_version

    assert _next_version("pcb1-01-v1") == "pcb1-01-v2"
    assert _next_version("pcb1-01-v9") == "pcb1-01-v10"
    assert _next_version("v3") == "v4"
    assert _next_version("사람이_지은_이름") == "사람이_지은_이름-rebuilt", (
        "규칙에 안 맞으면 조용히 번호를 붙이지 않는다 — 남의 판을 덮는다"
    )


def test_the_store_sorts_the_real_version_format_by_number(tmp_path):
    """`pcb1-01-v10` 이 `pcb1-01-v2` 보다 앞이어야 한다."""
    for number, hour in ((1, 9), (2, 10), (10, 11)):
        save_bank(make_bank(f"pcb1-01-v{number}"), "pcb1-01", root=tmp_path,
                  built_at=datetime(2026, 8, 15, hour, 0))

    found = [b.version for b in list_banks("pcb1-01", root=tmp_path)]
    assert found == ["pcb1-01-v10", "pcb1-01-v2", "pcb1-01-v1"]
