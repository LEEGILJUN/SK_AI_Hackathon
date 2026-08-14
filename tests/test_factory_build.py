"""`data/build_factory.py` 의 불변식.

이동현이 만든 공장 데이터 생성기를 이길준이 고쳤다(2026-08-14). 고친 것이
되돌아가지 않게 못 박는다. **이미지를 복사하지 않는다** — 순수 함수만 부른다.
전체 실행은 26,000장 5.2GB 라 테스트에서 돌릴 것이 아니다.

`data/` 는 패키지가 아니라 importlib 으로 읽는다.
"""

from __future__ import annotations

import importlib.util
import random
from datetime import date, datetime
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "data" / "build_factory.py"


@pytest.fixture(scope="module")
def bf():
    spec = importlib.util.spec_from_file_location("build_factory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _one_lot(bf, line="line_01", date_str="2026-06-10", lot_id="AAJ-01"):
    """로트 하나를 규칙 그대로 만든다. 이미지는 만들지 않고 (인덱스, 라벨)만."""
    rng = random.Random(bf.make_seed(line, date_str, lot_id, "base-lot-population"))
    indexes = set()
    while len(indexes) < bf.NORMAL_IMAGES_PER_LOT + bf.ANOMALY_IMAGES_PER_LOT:
        indexes.add(rng.randint(1, 9999))
    ordered = sorted(indexes)
    anomalies = bf.choose_background_anomaly_indexes(ordered, line, date_str, lot_id)
    return [(i, "defect" if i in anomalies else "normal") for i in ordered]


def test_the_bank_never_takes_a_defect_by_chance(bf):
    """뱅크에 결함이 우연히 섞이지 않는다.

    고치기 전에는 split 해시와 label 해시가 독립이라 `split="bank"` 안에
    결함이 9.1% 들어갔고, **모든 라인이 동시에 오염됐다.** 그러면 "이 라인만
    문제다"를 보여줄 수 없고 원인 여섯 중 뱅크 오염이 언제나 참이 된다.
    """
    day = date(2026, 6, 10)
    into_bank = [
        (index, label)
        for index, label in _one_lot(bf)
        if bf.pick_split(day, "line_01", index, label) == "bank"
    ]
    assert into_bank, "뱅크로 가는 이미지가 하나도 없다"
    assert all(label == "normal" for _, label in into_bank)


def test_defects_still_reach_operation_and_holdout(bf):
    """결함을 뱅크에서 막았다고 아예 사라지면 안 된다.

    운영 데이터에는 불량이 있어야 미검·과검을 가릴 수 있고, 홀드아웃에도
    있어야 게이트가 성능을 잰다.
    """
    day = date(2026, 6, 10)
    landed = {"bank": 0, "operation": 0, "holdout": 0}
    for index, label in _one_lot(bf):
        if label == "defect":
            landed[bf.pick_split(day, "line_01", index, label)] += 1
    assert landed["bank"] == 0
    assert landed["operation"] > 0
    assert landed["holdout"] > 0


def test_an_unknown_defect_never_gets_the_normal_code(bf):
    """모르는 결함에 "0000"(정상)이 붙지 않는다.

    손으로 적은 표라서 카테고리를 늘리면 없는 이름이 반드시 나온다. 그때
    정상 코드가 붙으면 MES 집계에서 불량이 통째로 사라진다.
    """
    assert bf.error_code_for("normal") == "0000"
    assert bf.error_code_for("scratch") == "ESA7"

    # VisA 다른 카테고리의 실제 결함 이름들 — pcb 표에 없다
    for unknown in ("bubble", "chunk of wax missing", "stuck together"):
        code = bf.error_code_for(unknown)
        assert code != "0000", f"{unknown} 에 정상 코드가 붙었다"
        assert code.startswith("ESB")
        # 결정론적이어야 한다. 다시 만들어도 같은 코드여야 재현된다.
        assert code == bf.error_code_for(unknown)


def test_every_issue_keyword_maps_to_a_defect_that_exists(bf):
    """이슈 키워드가 없는 결함을 가리키지 않는다.

    고치기 전에는 이물·오염·표면이 `dirt` 로 갔는데 pcb1~4 어디에도 없는
    이름이라 조용히 무시되고 최빈 결함으로 떨어졌다. "이물이 보입니다" 라고
    쓴 시나리오가 엉뚱한 결함 이미지를 받았다는 뜻이다.
    """
    if not bf.SOURCE_ROOT.exists():
        pytest.skip(f"VisA 원본이 없다: {bf.SOURCE_ROOT}")
    _, anomaly_pool = bf.load_source_samples()
    available = {s.primary_defect for pool in anomaly_pool.values() for s in pool}
    dangling = {k: v for k, v in bf.ISSUE_KEYWORD_TO_DEFECT.items() if v not in available}
    assert not dangling, f"어느 품목에도 없는 결함을 가리킨다: {dangling}"


def test_the_error_code_table_has_no_dead_entry(bf):
    """오류 코드 표에 쓰이지 않는 이름이 남아 있지 않다."""
    if not bf.SOURCE_ROOT.exists():
        pytest.skip(f"VisA 원본이 없다: {bf.SOURCE_ROOT}")
    _, anomaly_pool = bf.load_source_samples()
    available = {s.primary_defect for pool in anomaly_pool.values() for s in pool}
    dead = {k for k in bf.ERROR_CODE_BY_DEFECT_NAME if k != "normal" and k not in available}
    assert not dead, f"어느 품목에도 없는 결함에 코드가 붙어 있다: {dead}"


def test_the_shifts_are_named_the_usual_way(bf):
    """교대 이름이 통상 구간과 맞는다. 주간 08–16, 스윙 16–24, 야간 00–08."""
    assert bf.compute_shift(datetime(2026, 6, 10, 3, 0)) == "night"
    assert bf.compute_shift(datetime(2026, 6, 10, 9, 0)) == "day"
    assert bf.compute_shift(datetime(2026, 6, 10, 20, 0)) == "swing"


def test_the_scenario_injection_is_read(bf):
    """`injection` 절을 읽는다.

    장영진이 뱅크에 넣을 오염 이미지를 정확히 지정해 뒀는데 생성기가 읽지
    않고 있었다. 대신 해시가 아무 결함이나 뱅크에 넣었다.
    """
    scenarios, _, _, _ = bf.load_scenarios(REPO_ROOT / "data" / "scenarios.yaml")
    contaminated = [s for s in scenarios if s.injection_method == "bank_contamination"]
    assert contaminated, "bank_contamination 시나리오를 하나도 못 읽었다"
    for scenario in contaminated:
        assert scenario.contaminated_count > 0
        assert len(scenario.contaminated_images) <= scenario.contaminated_count


def test_named_contaminants_exist_in_visa(bf):
    """`contaminated_images` 가 가리키는 원본이 실제로 있다.

    없으면 오염이 조용히 다른 이미지로 채워지고, 채점 기준이 가리키는 것과
    뱅크에 들어간 것이 달라진다.
    """
    if not bf.SOURCE_ROOT.exists():
        pytest.skip(f"VisA 원본이 없다: {bf.SOURCE_ROOT}")
    _, anomaly_pool = bf.load_source_samples()
    known = {s.image_rel for pool in anomaly_pool.values() for s in pool}
    scenarios, _, _, _ = bf.load_scenarios(REPO_ROOT / "data" / "scenarios.yaml")
    for scenario in scenarios:
        for named in scenario.contaminated_images or []:
            assert named in known, f"{scenario.scenario_id}: VisA 에 없다 — {named}"


def test_the_bank_size_rule_is_not_two_different_numbers(bf):
    """뱅크를 몇 장으로 세우는가가 두 곳에 적혀 있고, 어긋나면 안 된다.

    이 규칙이 `app/pipeline.py` 안에만 있어서 공장 생성기 쪽에서는 알 길이
    없었다. 생성기는 이미지의 70% 를 `split="bank"` 로 찍는데(라인당 3,750장),
    뱅크는 거기서 150장만 뽑아 세운다. 이 차이를 모르면 오염률이 40배로
    희석돼 보인다 — 실제로 그렇게 잘못 읽었다.

    두 벌이 된 이상 한쪽만 고쳐지는 것을 막아야 한다.
    """
    from app.pipeline import VISA_NORMAL_COUNT

    assert bf.BANK_BUILD_SIZE == VISA_NORMAL_COUNT


def test_the_scenario_contamination_lands_near_what_we_measured(bf):
    """시나리오가 지정한 오염 장수가 실측 범위 안에 든다.

    VisA 실측에서 오염 3.2% 일 때 결함 위 패치가 뱅크의 0.1% 였다. 그보다
    훨씬 옅으면 coreset 을 거친 뒤 결함 패치가 한 장도 안 남아 역추적이
    오염원을 짚지 못하고, 그러면 정답을 걸어 둔 시나리오가 재현되지 않는다.

    **정답 파일을 고치지 않는다.** 이 시험은 어긋나면 알려 줄 뿐이다.
    """
    scenarios, _, _, _ = bf.load_scenarios(REPO_ROOT / "data" / "scenarios.yaml")
    contaminated = [s for s in scenarios if s.injection_method == "bank_contamination"]
    assert contaminated
    for scenario in contaminated:
        rate = scenario.contaminated_count / (bf.BANK_BUILD_SIZE + scenario.contaminated_count) * 100
        assert rate >= bf.MIN_DETECTABLE_CONTAMINATION_PCT, (
            f"{scenario.scenario_id}: 오염 {scenario.contaminated_count}장이면 "
            f"뱅크 {bf.BANK_BUILD_SIZE}장 기준 {rate:.2f}% 다. 검출 한계 아래라 "
            f"역추적이 못 짚을 수 있다 — 장영진 확인 필요"
        )


def test_the_lot_counts_are_totals_not_running_numbers(bf):
    """`inspected_count` · `defect_count` 가 로트 총량이다.

    고치기 전에는 결함마다 1, 2, 3… 이 들어가는 일련번호였다. 합계를 내면
    개수가 아니라 삼각수가 나와서 "이 로트 불량 N건"이 그대로 틀린다.
    """
    rows = [
        bf.ManifestRow(
            image_path=f"line_01/2026-06-10/AAJ-01/img_{i:04d}.png",
            line="line_01", object_name="pcb1", date_str="2026-06-10",
            lot_id="AAJ-01", equipment_id="EQ-01-A", split="operation",
            label="defect" if i < 3 else "normal", mask_path="",
            visa_source="", defect_name="scratch" if i < 3 else "normal",
            ercd="ESA7" if i < 3 else "0000",
        )
        for i in range(10)
    ]
    mes = bf.build_mes_rows(rows)
    assert len(mes) == 10
    assert {r["inspected_count"] for r in mes} == {"10"}
    assert {r["defect_count"] for r in mes} == {"3"}
