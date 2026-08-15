"""자리별 기준선 — 늘 큰 자리를 걷어내고 이번에만 큰 자리를 남긴다.

4090 이 pcb1 에서 쟀다.

    역추적 중심이 결함 위    4/10      capsules 8/10
    마스크 크롭으로 defect   9/10      ← 모델은 문제없다

좌표가 행 43~45 에 몰렸다(10장 중 7장). 결함 위치는 이미지마다 다른데
히트맵은 매번 같은 자리를 가리켰다.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspection.baseline_map import (
    RAW,
    RESIDUAL,
    ROBUST,
    apply_baseline,
    build_baseline,
    hottest_cell,
    hottest_cells,
)


def _normals(rows=6, cols=6, hotspot=(0, 0), hot=5.0, quiet=1.0, count=9):
    """`hotspot` 만 늘 큰 정상 이미지들. 기판의 구조물을 흉내낸다."""
    maps = []
    for i in range(count):
        grid = np.full((rows, cols), quiet)
        grid[hotspot] = hot + (i % 3) * 0.1     # 그 자리는 변동도 크다
        maps.append(grid.tolist())
    return maps


# ── 전역 정규화로는 안 된다 ─────────────────────────────────────────────


@pytest.mark.parametrize("scale,shift", [(2.0, 0.0), (0.5, 0.0), (1.0, 10.0)])
def test_scaling_the_whole_map_never_moves_the_hottest_cell(scale, shift):
    """거리 전체에 같은 수를 곱하거나 더해도 **가장 큰 자리는 그대로다.**

    이상맵 정규화를 어떻게 고쳐도 역추적이 짚는 자리는 안 바뀐다. 면적(판별
    7번)은 달라지지만 위치(판별 1번)는 안 달라진다. **자리마다 다른 값을
    빼야** 순서가 바뀐다.
    """
    grid = np.array([[5.2, 1.0], [1.0, 3.0]])
    assert hottest_cell(grid.tolist()) == (0, 0)
    assert hottest_cell((grid * scale + shift).tolist()) == (0, 0)


# ── 자리별로 빼면 순서가 바뀐다 ─────────────────────────────────────────


def test_a_structural_hotspot_is_subtracted_away():
    """늘 큰 자리는 걷어내고 이번에만 큰 자리를 짚는다."""
    baseline = build_baseline(_normals())
    query = np.full((6, 6), 1.0)
    query[0, 0] = 5.2      # 구조물 — 정상보다 조금 큼
    query[4, 4] = 3.0      # 결함 — 정상보다 세 배

    assert hottest_cell(query.tolist(), baseline, RAW) == (0, 0)
    assert hottest_cell(query.tolist(), baseline, RESIDUAL) == (4, 4)
    assert hottest_cell(query.tolist(), baseline, ROBUST) == (4, 4)


def test_robust_divides_by_how_much_that_cell_normally_moves():
    """변동이 큰 자리는 조금 커진 것으로 놀라지 않는다.

    커넥터 핀처럼 원래 들쭉날쭉한 자리와, 늘 잔잔하던 자리를 같은 크기로
    보면 안 된다.
    """
    maps = []
    for i in range(9):
        grid = np.full((4, 4), 1.0)
        grid[0, 0] = 3.0 + i * 0.5      # 변동이 큰 자리
        grid[3, 3] = 1.0 + i * 0.01     # 잔잔한 자리
        maps.append(grid.tolist())
    baseline = build_baseline(maps)

    query = np.full((4, 4), 1.0)
    query[0, 0] = 7.0       # 변동 폭 안에서 큼
    query[3, 3] = 1.3       # 잔잔하던 자리가 튐

    assert hottest_cell(query.tolist(), baseline, RESIDUAL) == (0, 0)
    assert hottest_cell(query.tolist(), baseline, ROBUST) == (3, 3)


# ── 조용히 다른 값을 쓰지 않는다 ────────────────────────────────────────


def test_without_a_baseline_nothing_changes():
    """기준선이 없으면 입력을 그대로 쓴다.

    못 만든 상황에서 조용히 다른 값을 쓰면 무엇을 보고 판정했는지 흐려진다.
    """
    grid = [[5.0, 1.0], [1.0, 3.0]]
    assert apply_baseline(grid, None, RESIDUAL).tolist() == grid
    assert apply_baseline(grid, None, ROBUST).tolist() == grid
    assert hottest_cell(grid, None, ROBUST) == (0, 0)


def test_a_baseline_from_a_different_grid_is_refused():
    """격자가 다르면 예외다. 입력 크기나 백본이 다르면 못 쓴다."""
    baseline = build_baseline(_normals(rows=6, cols=6))
    with pytest.raises(ValueError, match="격자"):
        apply_baseline([[1.0, 1.0], [1.0, 1.0]], baseline, RESIDUAL)


def test_the_baseline_uses_the_median_not_the_mean():
    """한 장의 이상치에 기준이 끌려가면 안 된다.

    정상이라고 기록된 것 중에 실제로는 결함이 섞일 수 있고, **그것이 바로
    우리가 찾는 뱅크 오염이다.** 평균을 쓰면 오염된 자리의 기준이 올라가
    같은 자리의 결함을 영영 못 잡는다.
    """
    maps = [[[1.0, 1.0], [1.0, 1.0]] for _ in range(9)]
    maps.append([[99.0, 1.0], [1.0, 1.0]])          # 섞여 든 뱅크 오염 한 장
    baseline = build_baseline(maps)

    assert baseline["median"][0][0] == pytest.approx(1.0), "평균이면 10.8 이 된다"


def test_top_cells_skip_the_neighbours_of_a_pick():
    """상위 여러 자리를 뽑을 때 1등 옆칸은 건너뛴다.

    같은 곳을 다시 보는 것이라 크롭을 여러 장 주는 의미가 없다.
    """
    grid = np.full((20, 20), 1.0)
    grid[5, 5] = 9.0
    grid[5, 6] = 8.9      # 바로 옆
    grid[15, 15] = 8.0    # 멀리 떨어진 다른 후보

    picked = hottest_cells(grid.tolist(), 2)
    assert picked[0] == (5, 5)
    assert picked[1] == (15, 15), f"옆칸을 골랐다: {picked}"


def test_an_empty_stack_is_refused():
    """정상 이미지가 없으면 기준선을 만들 수 없다."""
    with pytest.raises(ValueError):
        build_baseline([])
