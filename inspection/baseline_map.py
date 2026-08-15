"""이상맵에서 **자리마다 원래 큰 곳**을 걷어낸다.

── 왜 필요한가 ─────────────────────────────────────────────────────────

4090 이 pcb1 에서 역추적 위치를 쟀다.

    역추적 중심이 결함 위    4/10      capsules 8/10
    마스크 크롭으로 defect   9/10      ← 모델은 문제없다

그리고 좌표가 몰려 있었다.

    (43,25) (45,27) (43,33) (43,37) (44,19) (45,23) (45,27)
    10장 중 7장이 행 43~45

**결함 위치는 이미지마다 다른데 히트맵은 매번 같은 자리를 가리킨다.** 기판의
특정 구조물(커넥터 핀이나 경계)로 보인다 — 정상 이미지에서도 그 자리는 변동이
커서 최근접 거리가 늘 크게 나온다.

── 전역 정규화로는 안 된다 ─────────────────────────────────────────────

거리 전체에 같은 수를 곱하거나 빼는 것은 **가장 큰 자리를 바꾸지 못한다.**
단조 변환이라 순서가 그대로다. 이상맵 정규화를 어떻게 고쳐도 역추적이 짚는
자리는 같다. 면적(판별 7번)은 달라지지만 위치(판별 1번)는 안 달라진다.

**자리마다 다른 값을 빼야** 순서가 바뀐다. 그것이 이 파일이다.

── 어떻게 하는가 ───────────────────────────────────────────────────────

정상 이미지 여러 장을 같은 뱅크로 돌려 **격자 칸마다 거리 분포**를 구한다.
그다음 검사 이미지의 거리에서 그 자리의 기준값을 뺀다.

    잔차[r,c] = 거리[r,c] - 기준[r,c]

늘 큰 자리는 기준도 크므로 잔차가 작아지고, **정상에서는 조용한데 이번에만
큰 자리**가 남는다. 그것이 결함일 가능성이 높은 자리다.

`robust=True` 면 변동 폭으로 한 번 더 나눈다.

    잔차[r,c] = (거리[r,c] - 중앙값[r,c]) / (중앙편차[r,c] + eps)

자리마다 변동 폭이 다를 때 필요하다. 커넥터 핀처럼 원래 들쭉날쭉한 자리는
조금 커진 것으로 놀라지 않고, 늘 잔잔하던 자리가 조금만 커져도 잡힌다.

── 아직 기본값이 아니다 ────────────────────────────────────────────────

**이것이 낫다고 확인되기 전까지 추론 경로를 바꾸지 않는다.** 합성에서 좋아
보이는 것이 VisA 에서 뒤집힌 전례가 여러 번 있다.
`scripts/measure_trace_crop.py --map` 으로 실데이터에서 재고, 그 표를 보고
정한다.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: 어떤 방식으로 기준선을 세울 것인가.
RAW = "raw"                 # 아무것도 안 한다 — 지금 동작
RESIDUAL = "residual"       # 자리별 중앙값을 뺀다
ROBUST = "robust"           # 뺀 뒤 자리별 변동 폭으로 나눈다

MODES = (RAW, RESIDUAL, ROBUST)

#: 0 으로 나누는 것을 막는 값. 변동이 거의 없는 자리를 무한대로 만들지 않는다.
_EPS = 1e-6


def build_baseline(distance_maps: Sequence[Sequence[Sequence[float]]]) -> dict:
    """정상 이미지들의 격자 거리에서 자리별 기준을 만든다.

    **정상 이미지만 넣어야 한다.** 결함이 섞이면 그 자리의 기준이 올라가서
    같은 자리의 결함을 영영 못 잡는다. 뱅크를 세울 때 쓴 이미지나 운영 구간의
    정상 판정 이미지를 쓴다.

    중앙값과 중앙절대편차(MAD)를 쓴다. 평균과 표준편차는 한 장의 이상치에
    끌려간다 — 정상이라고 기록된 것 중에 실제로는 결함인 것이 섞일 수 있고,
    그것이 바로 우리가 찾는 뱅크 오염이다.
    """
    stack = np.asarray(distance_maps, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[0] == 0:
        raise ValueError("정상 이미지들의 격자 거리 맵 목록이어야 한다")

    median = np.median(stack, axis=0)
    mad = np.median(np.abs(stack - median), axis=0)
    return {
        "median": median.tolist(),
        "mad": mad.tolist(),
        "images": int(stack.shape[0]),
        "grid": [int(stack.shape[1]), int(stack.shape[2])],
    }


def apply_baseline(
    patch_distances: Sequence[Sequence[float]],
    baseline: dict | None,
    mode: str = RAW,
) -> np.ndarray:
    """검사 이미지의 격자 거리에 기준선을 적용한다.

    baseline 이 없거나 mode 가 raw 면 **입력을 그대로 돌려준다.** 기준선을
    못 만든 상황에서 조용히 다른 값을 쓰면 무엇을 보고 판정했는지가 흐려진다.
    """
    grid = np.asarray(patch_distances, dtype=np.float64)
    if mode == RAW or baseline is None:
        return grid

    median = np.asarray(baseline["median"], dtype=np.float64)
    if median.shape != grid.shape:
        raise ValueError(
            f"기준선 격자 {median.shape} 와 이미지 격자 {grid.shape} 가 다르다. "
            f"같은 입력 크기·같은 백본으로 만든 기준선이어야 한다."
        )

    residual = grid - median
    if mode == RESIDUAL:
        return residual

    mad = np.asarray(baseline["mad"], dtype=np.float64)
    return residual / (mad + _EPS)


def hottest_cell(
    patch_distances: Sequence[Sequence[float]],
    baseline: dict | None = None,
    mode: str = RAW,
) -> tuple[int, int]:
    """가장 이상한 자리의 격자 좌표 (행, 열).

    역추적 크롭이 자를 자리다. `mode` 에 따라 **답이 달라진다** — 그것이 이
    파일의 요점이다.
    """
    scored = apply_baseline(patch_distances, baseline, mode)
    row, col = np.unravel_index(int(np.argmax(scored)), scored.shape)
    return int(row), int(col)


def hottest_cells(
    patch_distances: Sequence[Sequence[float]],
    count: int,
    baseline: dict | None = None,
    mode: str = RAW,
    min_separation: int = 3,
) -> list[tuple[int, int]]:
    """상위 여러 자리. 서로 너무 붙은 것은 건너뛴다.

    1등 옆칸은 같은 곳을 다시 보는 것이라 크롭을 여러 장 주는 의미가 없다.
    `min_separation` 칸 안쪽은 같은 자리로 본다.
    """
    scored = apply_baseline(patch_distances, baseline, mode)
    order = np.argsort(scored, axis=None)[::-1]
    picked: list[tuple[int, int]] = []
    for flat in order:
        row, col = np.unravel_index(int(flat), scored.shape)
        if any(abs(row - r) < min_separation and abs(col - c) < min_separation
               for r, c in picked):
            continue
        picked.append((int(row), int(col)))
        if len(picked) >= count:
            break
    return picked
