"""격자 좌표 → 원본 픽셀 영역 되돌리기 검증.

이 변환은 틀려도 예외가 나지 않는다. 그냥 엉뚱한 자리를 잘라서 판독에
넘기고, 진단은 조용히 틀린다. 그래서 좌표가 맞는지를 직접 재야 한다.

마지막 테스트가 본체다. 결함을 넣은 위치를 알고 있는 이미지를 추론에 넣고,
이상 점수가 가장 높은 격자 칸을 잘라 **그 안에 실제로 결함이 들어 있는지**를
픽셀로 확인한다. 추론과 좌표 변환이 같은 자리를 가리켜야만 통과한다.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from inspection import FeatureConfig, PatchEmbedder, build_bank, crop_patch, patch_box, score_image
from inspection.crop import crop_with_context
from inspection.types import PatchRef
from tests.synthetic import IMAGE_SIZE, make_normal, write_set

CONFIG = FeatureConfig(backbone="resnet18", crop=64)
GRID = (8, 8)


def ref(row: int, col: int) -> PatchRef:
    return PatchRef(source_image="x.png", row=row, col=col, patch_index=row * GRID[1] + col)


# ── 좌표 변환 자체 ─────────────────────────────────────────────────────


def test_corners_map_to_corners():
    """정사각 이미지에서 첫 칸은 좌상단, 마지막 칸은 우하단이어야 한다."""
    size = (IMAGE_SIZE, IMAGE_SIZE)

    left, top, right, bottom = patch_box(ref(0, 0), GRID, size, CONFIG)
    assert left == 0 and top == 0
    assert 0 < right < IMAGE_SIZE and 0 < bottom < IMAGE_SIZE

    left, top, right, bottom = patch_box(ref(7, 7), GRID, size, CONFIG)
    assert right == IMAGE_SIZE and bottom == IMAGE_SIZE
    assert 0 < left < IMAGE_SIZE and 0 < top < IMAGE_SIZE


def test_boxes_tile_without_gaps_or_overlap():
    """여유 없이 자르면 칸들이 빈틈없이 이어져야 한다."""
    size = (IMAGE_SIZE, IMAGE_SIZE)
    boxes = [patch_box(ref(0, c), GRID, size, CONFIG) for c in range(GRID[1])]

    for left_box, right_box in zip(boxes, boxes[1:]):
        assert left_box[2] == right_box[0], f"칸 사이가 어긋난다: {left_box} → {right_box}"

    assert boxes[0][0] == 0
    assert boxes[-1][2] == IMAGE_SIZE


def test_row_and_col_are_not_swapped():
    """행과 열을 뒤바꾸는 실수는 정사각 이미지에서 눈에 띄지 않는다.

    행을 늘리면 세로로, 열을 늘리면 가로로 움직여야 한다.
    """
    size = (IMAGE_SIZE, IMAGE_SIZE)
    origin = patch_box(ref(0, 0), GRID, size, CONFIG)
    down = patch_box(ref(3, 0), GRID, size, CONFIG)
    across = patch_box(ref(0, 3), GRID, size, CONFIG)

    assert down[1] > origin[1] and down[0] == origin[0]  # 아래로만
    assert across[0] > origin[0] and across[1] == origin[1]  # 오른쪽으로만


def test_a_wide_image_maps_across_its_whole_width():
    """가로가 긴 이미지도 격자가 **전체 폭**을 덮는다.

    전처리가 자르지 않고 정사각으로 리사이즈하므로 되짚기도 축마다 따로다.
    전에는 "짧은 변 맞춤 + 중앙 크롭" 기하로 되짚어서 양옆이 사각지대였다.

    **여기가 어긋나면 역추적이 엉뚱한 자리를 가리킨다** — 판별 1번과 5번이
    그 좌표로 이미지를 잘라 모델에 주므로, 조용히 틀린 곳을 보게 된다.
    """
    wide = (256, 128)  # 너비가 두 배
    first_left, _, _, _ = patch_box(ref(0, 0), GRID, wide, CONFIG, margin=0)
    _, _, last_right, _ = patch_box(ref(0, GRID[1] - 1), GRID, wide, CONFIG, margin=0)

    assert first_left == 0, "첫 칸이 왼쪽 끝에서 시작해야 한다"
    assert last_right == wide[0], "마지막 칸이 오른쪽 끝까지 가야 한다"


def test_each_axis_scales_on_its_own():
    """가로와 세로의 배율이 다르다. 한 배율로 되짚으면 세로가 어긋난다."""
    wide = (256, 128)
    _, top, _, bottom = patch_box(ref(0, 0), GRID, wide, CONFIG, margin=0)
    left, _, right, _ = patch_box(ref(0, 0), GRID, wide, CONFIG, margin=0)

    assert right - left == wide[0] // GRID[1]
    assert bottom - top == wide[1] // GRID[0]


def test_margin_widens_and_clamps_to_bounds():
    size = (IMAGE_SIZE, IMAGE_SIZE)
    tight = patch_box(ref(4, 4), GRID, size, CONFIG, margin=0)
    loose = patch_box(ref(4, 4), GRID, size, CONFIG, margin=10)

    assert loose[0] < tight[0] and loose[2] > tight[2]

    # 경계 칸에서도 이미지 밖으로 나가지 않는다
    edge = patch_box(ref(0, 0), GRID, size, CONFIG, margin=50)
    assert edge[0] >= 0 and edge[1] >= 0
    assert edge[2] <= IMAGE_SIZE and edge[3] <= IMAGE_SIZE


def test_out_of_range_coordinate_is_rejected():
    with pytest.raises(ValueError, match="격자 범위를 벗어난"):
        patch_box(ref(8, 0), GRID, (IMAGE_SIZE, IMAGE_SIZE), CONFIG)


# ── 잘라내기 ───────────────────────────────────────────────────────────


def test_crop_enlarges_small_patches_for_reading(tmp_path):
    """격자 한 칸은 수십 픽셀이라 그대로 보내면 판독이 어렵다."""
    path = tmp_path / "n.png"
    make_normal(0).save(path)

    raw = crop_patch(path, ref(3, 3), GRID, CONFIG, margin=0, enlarge_to=None)
    enlarged = crop_patch(path, ref(3, 3), GRID, CONFIG, margin=0, enlarge_to=96)

    assert min(raw.size) < 96
    assert min(enlarged.size) >= 96


def test_context_crop_is_larger_than_single_cell(tmp_path):
    path = tmp_path / "n.png"
    make_normal(0).save(path)

    single = crop_patch(path, ref(4, 4), GRID, CONFIG, margin=0, enlarge_to=None)
    with_context = crop_with_context(path, ref(4, 4), GRID, CONFIG, context_cells=2)

    assert with_context.size[0] > single.size[0]
    assert with_context.size[1] > single.size[1]


# ── 추론과 좌표가 같은 자리를 가리키는가 ───────────────────────────────


def test_top_patch_crop_actually_contains_the_defect(tmp_path):
    """이상 점수가 가장 높은 칸을 잘랐을 때 그 안에 결함이 있어야 한다.

    결함을 넣은 위치를 알고 있으므로 픽셀로 확인할 수 있다. 추론이 짚은
    격자 좌표와 crop 이 되돌린 픽셀 좌표가 어긋나면 여기서 걸린다.
    """
    embedder = PatchEmbedder(CONFIG)
    root = tmp_path / "factory"
    normal_paths = write_set(root / "normal", 10, "normal", seed_offset=0)

    # 결함을 정확히 아는 위치에 찍는다. 가장자리를 피해 중앙 아래쪽에 둔다.
    canvas = np.asarray(make_normal(99), dtype=np.float32)[:, :, 0]
    cy, cx, radius = 88, 40, 10
    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    canvas[((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2] = 20.0
    query = root / "query.png"
    Image.fromarray(np.stack([canvas.astype(np.uint8)] * 3, axis=-1)).save(query)

    bank = build_bank(normal_paths, embedder, coreset_ratio=0.3, seed=0, root=root)
    result = score_image(query, bank, embedder, root=root, top_k=1)

    top = result.matches[0].query
    box = patch_box(top, (result.grid_h, result.grid_w), (IMAGE_SIZE, IMAGE_SIZE), CONFIG, margin=6)

    # 잘라낸 영역 안에 어두운 결함 픽셀이 실제로 들어 있어야 한다
    region = canvas[box[1] : box[3], box[0] : box[2]]
    assert region.size > 0
    assert region.min() < 60, (
        f"짚은 칸({top.row},{top.col}) → 영역 {box} 안에 결함이 없다. "
        f"영역 최소 밝기 {region.min():.1f}"
    )

    # 결함 중심이 그 영역 안에 있어야 한다
    assert box[0] <= cx <= box[2] and box[1] <= cy <= box[3], (
        f"결함 중심 ({cx},{cy}) 이 잘라낸 영역 {box} 밖이다"
    )
