"""`data/build_factory.py` 의 불변식.

데이터 담당이 만든 공장 데이터 생성기를 에이전트 담당이 고쳤다(2026-08-14~15). 고친 것이
되돌아가지 않게 못 박는다. **이미지를 복사하지 않는다** — 순수 함수만 부른다.
전체 실행은 로트 36개 × 1,000장이라 테스트에서 돌릴 것이 아니다
(`--no-images` 로 CSV 만 만들면 1.6초다).

`data/` 는 패키지가 아니라 importlib 으로 읽는다.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime
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


@pytest.fixture(scope="module")
def calendar(bf):
    """(라인, 일자) → 구간. 시나리오에서 실제로 만들어지는 것과 같다."""
    scenarios, _, _, _ = bf.load_scenarios(REPO_ROOT / "data" / "scenarios.yaml")
    return bf.build_split_calendar(
        bf.bank_window_lots(scenarios),
        bf.collect_target_lots(scenarios),
        bf.pending_window_lots(scenarios),
    )


def test_the_bank_window_is_separate_from_the_operation_window(bf, calendar):
    """라인마다 뱅크 구간과 운영 구간이 날짜로 갈린다.

    고치기 전에는 `pick_split` 이 이미지마다 해시로 갈라서 **같은 날짜에
    뱅크·운영·홀드아웃이 전부 섞였다.** 그러면 화질 기준 분포를 뱅크
    구간에서만 뽑을 수가 없고, 열화가 주입된 운영 데이터가 기준에 섞여
    **설비·광학 원인을 영영 못 잡는다.**

    라인별로 본다 — 기준 분포가 라인·품목마다 따로이므로 2라인의 초기
    수집일이 1라인의 운영일과 같은 날인 것은 아무 문제가 아니다.
    """
    from collections import defaultdict

    windows = defaultdict(lambda: defaultdict(set))
    for (line, date_str), window in calendar.items():
        windows[line][window].add(date_str)

    assert windows, "구간 달력이 비었다"
    for line, by_window in sorted(windows.items()):
        assert by_window["bank"], f"{line}: 뱅크 구간이 없다"
        assert by_window["operation"], f"{line}: 운영 구간이 없다"
        overlap = by_window["bank"] & by_window["operation"]
        assert not overlap, f"{line}: 뱅크와 운영 일자가 겹친다 — {sorted(overlap)}"


def test_the_bank_window_comes_before_the_scenarios(bf, calendar):
    """초기 뱅크 구성 구간이 시나리오 발생보다 앞선다.

    현장 순서 그대로다 — 설비를 들이고 정상만 모아 초기 뱅크를 만든 뒤
    운영이 시작되고, 그러다 미검이 난다. 순서가 뒤집히면 "미검이 난 뒤에
    모은 데이터로 그 미검을 놓친 뱅크를 만들었다"가 된다.
    """
    from collections import defaultdict

    latest_bank, earliest_other = defaultdict(list), defaultdict(list)
    for (line, date_str), window in calendar.items():
        (latest_bank if window == "bank" else earliest_other)[line].append(date_str)

    for line in sorted(latest_bank):
        assert max(latest_bank[line]) < min(earliest_other[line]), (
            f"{line}: 뱅크 구간이 운영보다 뒤에 있다"
        )


def test_a_defect_in_the_bank_window_is_an_error_not_a_silent_move(bf, calendar):
    """뱅크 구성 구간에 결함이 생기면 조용히 넘기지 않고 멈춘다.

    다른 구간으로 슬쩍 옮기면 그 날짜가 두 구간에 걸쳐 일자 분리가
    무의미해진다. 생성 단계에서 안 만드는 것이 맞고, 그래도 오면 고장이다.
    """
    line, date_str = next((k for k, v in calendar.items() if v == "bank"))
    assert bf.pick_split(calendar, line, date_str, "normal") == "bank"
    with pytest.raises(ValueError):
        bf.pick_split(calendar, line, date_str, "defect")


def test_defects_still_reach_operation_and_holdout(bf, calendar):
    """결함을 뱅크에서 막았다고 아예 사라지면 안 된다.

    운영 데이터에는 불량이 있어야 미검·과검을 가릴 수 있고, 홀드아웃에도
    있어야 게이트가 성능을 잰다.
    """
    landed = {
        bf.pick_split(calendar, line, date_str, "defect")
        for (line, date_str), window in calendar.items()
        if window != "bank"
    }
    assert "operation" in landed
    assert "holdout" in landed
    assert "bank" not in landed


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

    고치기 전에는 이물·뱅크 오염·표면이 `dirt` 로 갔는데 pcb1~4 어디에도 없는
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

    도메인 담당이 뱅크에 넣을 혼입 이미지를 정확히 지정해 뒀는데 생성기가 읽지
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

    없으면 뱅크 오염이 조용히 다른 이미지로 채워지고, 채점 기준이 가리키는 것과
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
    뱅크는 거기서 150장만 뽑아 세운다. 이 차이를 모르면 혼입률이 40배로
    희석돼 보인다 — 실제로 그렇게 잘못 읽었다.

    두 벌이 된 이상 한쪽만 고쳐지는 것을 막아야 한다.
    """
    from app.pipeline import VISA_NORMAL_COUNT

    assert bf.BANK_BUILD_SIZE == VISA_NORMAL_COUNT


def test_the_scenario_contamination_lands_near_what_we_measured(bf):
    """시나리오가 지정한 뱅크 오염 장수가 실측 범위 안에 든다.

    VisA 실측에서 혼입률 3.2% 일 때 결함 위 패치가 뱅크의 0.1% 였다. 그보다
    훨씬 옅으면 coreset 을 거친 뒤 결함 패치가 한 장도 안 남아 역추적이
    혼입 이미지를 짚지 못하고, 그러면 정답을 걸어 둔 시나리오가 재현되지 않는다.

    **정답 파일을 고치지 않는다.** 이 시험은 어긋나면 알려 줄 뿐이다.
    """
    scenarios, _, _, _ = bf.load_scenarios(REPO_ROOT / "data" / "scenarios.yaml")
    contaminated = [s for s in scenarios if s.injection_method == "bank_contamination"]
    assert contaminated
    for scenario in contaminated:
        rate = scenario.contaminated_count / (bf.BANK_BUILD_SIZE + scenario.contaminated_count) * 100
        assert rate >= bf.MIN_DETECTABLE_CONTAMINATION_PCT, (
            f"{scenario.scenario_id}: 뱅크 오염 {scenario.contaminated_count}장이면 "
            f"뱅크 {bf.BANK_BUILD_SIZE}장 기준 {rate:.2f}% 다. 검출 한계 아래라 "
            f"역추적이 못 짚을 수 있다 — 도메인 담당 확인 필요"
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


# ── pending — 아직 검사 안 된 생산분 ────────────────────────────────────


def test_pending_comes_after_everything_else(bf, calendar):
    """pending 은 시나리오보다 뒤에 온다.

    "아직 검사 안 됐다"는 것은 **가장 최근 생산분**이라는 뜻이다. 앞에
    두면 "왜 그동안 안 돌렸나"가 되어 이야기가 어긋난다.
    """
    from collections import defaultdict

    days = defaultdict(lambda: defaultdict(list))
    for (line, date_str), window in calendar.items():
        days[line][window].append(date_str)

    for line, by_window in sorted(days.items()):
        assert by_window["pending"], f"{line}: pending 이 없다"
        others = [d for w, ds in by_window.items() if w != "pending" for d in ds]
        assert min(by_window["pending"]) > max(others), (
            f"{line}: pending 이 다른 구간보다 앞에 있다"
        )


def test_the_defect_pool_is_not_reused(bf, calendar):
    """결함 원본을 재사용하지 않는다.

    VisA 는 카테고리당 결함이 **100장뿐**이다. 라인당 필요량이 그것을 넘으면
    같은 이미지가 여러 번 나오고, **홀드아웃에서 중복되면 게이트 점수가
    부풀려진다.**

    로트 크기가 이 값을 정한다. 전에는 로트 1,000장이라 라인당 900장이
    필요해 9배 재사용됐다.
    """
    from collections import Counter

    VISA_DEFECTS_PER_CATEGORY = 100
    per_line = Counter(
        line for (line, _date), window in calendar.items() if window != "bank"
    )
    for line, days in sorted(per_line.items()):
        need = days * bf.ANOMALY_IMAGES_PER_LOT
        assert need <= VISA_DEFECTS_PER_CATEGORY, (
            f"{line}: 결함 {need}장이 필요한데 원본은 "
            f"{VISA_DEFECTS_PER_CATEGORY}장뿐이다 — 재사용이 생긴다. "
            f"ANOMALY_IMAGES_PER_LOT 을 줄이거나 일수를 줄여야 한다"
        )


def test_the_lot_is_ten_percent_defective(bf):
    """로트 구성이 결함 10% 다. pending 도 같다 — 아직 안 본 생산분이므로."""
    total = bf.NORMAL_IMAGES_PER_LOT + bf.ANOMALY_IMAGES_PER_LOT
    assert bf.ANOMALY_IMAGES_PER_LOT / total == 0.10
