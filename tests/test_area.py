"""판별 7번의 면적 — 뱅크마다 다른 기준으로 잰다."""

from __future__ import annotations

import numpy as np
import pytest

from inspection.area import DEFAULT_ROBUST_CUTOFF, RAW_BASIS, measured_area
from inspection.baseline_map import ROBUST, build_baseline
from inspection.mask import defect_area

CROP = 64
THRESHOLD = 2.2


def normal_maps(count: int, *, seed: int = 0, high: float = 1.7) -> list[list[list[float]]]:
    """자리마다 원래 크기가 다른 정상 이미지들.

    실측을 본뜬다 — pcb1 정상부 거리가 1.4~1.7 이고 **전역 컷오프 1.1 을
    이미 넘는다.** 그것이 면적이 화면 전체로 나오던 이유다.
    """
    rng = np.random.default_rng(seed)
    base = np.linspace(1.4, high, 64).reshape(8, 8)
    return [(base + rng.normal(0, 0.02, base.shape)).tolist() for _ in range(count)]


def with_defect(grid: list[list[float]], size: int = 2, lift: float = 0.6) -> list[list[float]]:
    arr = np.asarray(grid, dtype=np.float64).copy()
    arr[3:3 + size, 3:3 + size] += lift
    return arr.tolist()


# ── 왜 고쳐야 했는가 ───────────────────────────────────────────────────


def test_the_old_way_marks_almost_the_whole_frame():
    """**정상 이미지가 화면 전체로 나온다.**

    4090 실측에서 261,901px² 였다. 512x512 가 262,144 이므로 99.9% 다.
    무엇을 넣어도 "불량"이 되는 상수였다.
    """
    clean = normal_maps(1)[0]
    old = defect_area(clean, crop=CROP, threshold=THRESHOLD)

    assert old > CROP * CROP * 0.9, (
        f"정상인데 면적이 {old} — 이 시험이 실패하면 옛 방식의 문제가 사라진 것이다"
    )


def test_the_baseline_separates_normal_from_defect():
    """기준선을 빼면 정상은 거의 0, 결함은 남는다."""
    normals = normal_maps(30)
    baseline = build_baseline(normals)

    clean, _ = measured_area(normal_maps(1, seed=99)[0], crop=CROP,
                             threshold=THRESHOLD, baseline=baseline)
    dirty, _ = measured_area(with_defect(normal_maps(1, seed=99)[0]), crop=CROP,
                             threshold=THRESHOLD, baseline=baseline)

    assert clean < CROP * CROP * 0.05, f"정상인데 면적이 크다: {clean}"
    assert dirty > clean * 5, f"결함이 정상과 구분되지 않는다: {dirty} 대 {clean}"


def test_a_flatter_item_also_works():
    """**정상과 결함 차이가 작은 품목에서도 서야 한다.**

    capsules 는 정상 중앙 2.251 · 결함 중앙 2.460 으로 차이가 0.21 뿐이다.
    전역 컷오프로는 어떤 값을 잡아도 둘을 나누지 못한다.
    """
    normals = normal_maps(30, high=2.3)
    baseline = build_baseline(normals)
    query = normal_maps(1, seed=7, high=2.3)[0]

    clean, _ = measured_area(query, crop=CROP, threshold=THRESHOLD, baseline=baseline)
    dirty, _ = measured_area(with_defect(query, lift=0.21), crop=CROP,
                             threshold=THRESHOLD, baseline=baseline)

    assert dirty > clean, f"차이가 작은 품목에서 구분되지 않는다: {dirty} 대 {clean}"


# ── 기준으로 무엇을 썼는지 말한다 ──────────────────────────────────────


def test_it_says_which_basis_it_used():
    """**어느 기준으로 잰 값인지 모르면 그 숫자로 판정을 논할 수 없다.**

    기준선이 있는 값과 없는 값은 단위가 다르다 — 앞은 자리별 변동 폭의
    배수이고 뒤는 픽셀 거리다. 숫자만 돌려주면 그 사실이 사라진다.
    """
    baseline = build_baseline(normal_maps(30))
    query = normal_maps(1, seed=5)[0]

    _, with_base = measured_area(query, crop=CROP, threshold=THRESHOLD, baseline=baseline)
    _, without = measured_area(query, crop=CROP, threshold=THRESHOLD, baseline=None)

    assert with_base == ROBUST
    assert without == RAW_BASIS


def test_without_a_baseline_it_falls_back_rather_than_returning_zero():
    """기준선이 없다고 0 을 돌려주면 **전건이 양품**이 된다.

    예전에 실제로 그랬다 — 합성용 입력 크기를 실데이터에 써서 면적이 64배
    작게 나왔고, 판별 7번이 어떤 결함으로도 기준을 넘지 못했다.
    """
    query = with_defect(normal_maps(1, seed=3)[0])
    area, basis = measured_area(query, crop=CROP, threshold=THRESHOLD, baseline=None)

    assert basis == RAW_BASIS
    assert area == defect_area(query, crop=CROP, threshold=THRESHOLD)


def test_an_empty_map_is_zero_not_an_error():
    assert measured_area([], crop=CROP, threshold=THRESHOLD) == (0.0, RAW_BASIS)


# ── 기준선을 뱅크에 붙인다 ─────────────────────────────────────────────


def test_a_thin_baseline_is_refused_rather_than_used():
    """**모자란 기준선보다 없는 편이 낫다.**

    중앙값과 중앙절대편차를 몇 장으로 만들면 그 몇 장의 우연을 기준으로
    삼게 된다. 없으면 예전 방식으로 재고 화면이 그렇다고 적지만, 엉성한
    기준선은 그럴듯한 숫자를 내면서 틀린다.
    """
    from inspection.area import attach_baseline

    class FakeBank:
        meta: dict = {}

    bank = FakeBank()
    assert attach_baseline(bank, ["a.png", "b.png"], minimum=8) is False
    assert "baseline" not in bank.meta


def test_the_demo_factory_attaches_a_baseline(tmp_path):
    """가상 공장이 뱅크를 세울 때 기준선도 함께 만든다.

    **합성에서도 만들어야 한다.** 맥에서 이 경로가 한 번도 안 밟히면
    기준선이 깨져도 시험이 못 잡는다.
    """
    from app.pipeline import CONTAMINATED_ITEM, DemoFactory

    factory = DemoFactory(visa_root="/tmp/_no_visa_here", store_root=tmp_path)
    bank = factory.items[CONTAMINATED_ITEM].bank
    baseline = bank.meta.get("baseline")

    assert baseline is not None, "기준선이 안 붙었다"
    assert baseline["images"] >= 8
    assert tuple(baseline["grid"]) == bank.grid, "기준선 격자가 뱅크와 다르다"


def test_the_baseline_images_are_not_in_the_bank(tmp_path):
    """**기준선은 뱅크에 안 들어간 이미지로 만든다.**

    뱅크에 든 이미지는 자기 패치가 뱅크에 있어 거리가 낮게 나오고, 그 낮은
    값을 기준으로 삼으면 잔차가 부풀어 면적이 다시 커진다.
    """
    from app.pipeline import CONTAMINATED_ITEM, DemoFactory

    factory = DemoFactory(visa_root="/tmp/_no_visa_here", store_root=tmp_path)
    item = factory.items[CONTAMINATED_ITEM]

    in_bank = {factory.relative(p) for p in item.bank_normal}
    holdout = {factory.relative(p) for p in item.holdout_normal}
    assert not (in_bank & holdout)

    # 기준선용 이미지는 뱅크에도 홀드아웃에도 없어야 한다.
    all_normal = set(bank_image for bank_image in item.bank.images)
    assert all_normal <= in_bank | {factory.relative(p) for p in item.contaminants}, (
        "뱅크에 기준선용 이미지가 섞였다"
    )


@pytest.mark.parametrize("cutoff", [1.0, DEFAULT_ROBUST_CUTOFF, 10.0])
def test_a_higher_cutoff_never_grows_the_area(cutoff):
    """컷오프를 올리면 면적이 줄거나 같아야 한다. 늘면 부호가 뒤집힌 것이다."""
    baseline = build_baseline(normal_maps(30))
    query = with_defect(normal_maps(1, seed=11)[0])

    area, _ = measured_area(query, crop=CROP, threshold=THRESHOLD,
                            baseline=baseline, robust_cutoff=cutoff)
    tighter, _ = measured_area(query, crop=CROP, threshold=THRESHOLD,
                               baseline=baseline, robust_cutoff=cutoff + 5)
    assert tighter <= area
