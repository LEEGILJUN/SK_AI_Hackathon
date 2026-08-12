"""격자 좌표를 원본 이미지의 픽셀 영역으로 되돌린다.

역추적이 알려주는 것은 "뱅크의 이 패치가 저 이미지의 격자 (4,5)였다"까지다.
그 자리를 사람이나 시각 언어 모델이 판독하려면 원본 이미지에서 해당 영역을
잘라내야 한다. 판별 항목 5번(그 패치가 결함인가 진짜 정상품인가)이 여기에
걸려 있다.

좌표 되돌리기가 까다로운 이유는 임베딩 전에 두 번 변환하기 때문이다.

    원본 (W0,H0) → Resize(짧은 변 = resize) → CenterCrop(crop) → 격자 (Hg,Wg)

이 변환을 역순으로 풀어야 원본 픽셀 좌표가 나온다. 여기서 어긋나면 엉뚱한
자리를 판독하게 되고, 진단이 조용히 틀린다.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .features import FeatureConfig
from .types import PatchRef

# 잘라낸 조각이 너무 작으면 사람도 모델도 판독하지 못한다.
# 판독용으로 확대할 최소 한 변 길이.
MIN_VIEW_SIZE = 96


def patch_box(
    ref: PatchRef,
    grid: tuple[int, int],
    image_size: tuple[int, int],
    config: FeatureConfig | None = None,
    margin: int = 0,
) -> tuple[int, int, int, int]:
    """격자 한 칸에 대응하는 원본 이미지 픽셀 영역을 구한다.

    image_size 는 원본 이미지의 (너비, 높이)다.
    margin 은 잘라낼 때 사방으로 더 볼 여유 픽셀. 문맥이 있어야 판독이 쉬워진다.

    반환은 PIL 의 crop 인자 형식 (left, top, right, bottom) 이다.
    """
    config = config or FeatureConfig()
    grid_h, grid_w = grid
    width, height = image_size

    if not (0 <= ref.row < grid_h and 0 <= ref.col < grid_w):
        raise ValueError(f"격자 범위를 벗어난 좌표다: ({ref.row},{ref.col}) / 격자 {grid}")

    # 1) 격자 → CenterCrop 된 정사각 이미지 안의 좌표
    cell_h = config.crop / grid_h
    cell_w = config.crop / grid_w
    left_c = ref.col * cell_w
    top_c = ref.row * cell_h
    right_c = left_c + cell_w
    bottom_c = top_c + cell_h

    # 2) CenterCrop 되돌리기 → Resize 직후 이미지 안의 좌표
    scale = config.resize / min(width, height)
    resized_w = width * scale
    resized_h = height * scale
    offset_x = (resized_w - config.crop) / 2
    offset_y = (resized_h - config.crop) / 2

    left_r = left_c + offset_x
    top_r = top_c + offset_y
    right_r = right_c + offset_x
    bottom_r = bottom_c + offset_y

    # 3) Resize 되돌리기 → 원본 좌표
    left = left_r / scale
    top = top_r / scale
    right = right_r / scale
    bottom = bottom_r / scale

    # 여유 픽셀을 붙이고 원본 경계 안으로 자른다.
    left = max(0, int(round(left - margin)))
    top = max(0, int(round(top - margin)))
    right = min(width, int(round(right + margin)))
    bottom = min(height, int(round(bottom + margin)))

    # 경계에서 폭이 0이 되는 것을 막는다.
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)

    return left, top, right, bottom


def crop_patch(
    image_path: str | Path,
    ref: PatchRef,
    grid: tuple[int, int],
    config: FeatureConfig | None = None,
    margin: int = 12,
    enlarge_to: int | None = MIN_VIEW_SIZE,
) -> Image.Image:
    """패치 자리를 원본 이미지에서 잘라낸다.

    enlarge_to
        잘라낸 조각의 짧은 변이 이 값보다 작으면 확대한다. 격자 한 칸은
        보통 수십 픽셀이라 그대로 보내면 판독이 어렵다. 확대는 정보를
        늘리지 않지만 판독 안정성에는 도움이 된다.
        None 이면 원 크기 그대로 둔다.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        box = patch_box(ref, grid, image.size, config=config, margin=margin)
        patch = image.crop(box)

    if enlarge_to:
        short_side = min(patch.size)
        if 0 < short_side < enlarge_to:
            factor = enlarge_to / short_side
            new_size = (max(1, round(patch.width * factor)), max(1, round(patch.height * factor)))
            patch = patch.resize(new_size, Image.LANCZOS)

    return patch


def crop_with_context(
    image_path: str | Path,
    ref: PatchRef,
    grid: tuple[int, int],
    config: FeatureConfig | None = None,
    context_cells: int = 2,
) -> Image.Image:
    """패치와 그 주변까지 함께 잘라낸다.

    한 칸만 떼어 보면 무엇을 보고 있는지 알기 어렵다. 주변 몇 칸을 함께
    보여주면 "이게 제품의 어느 부위인가"가 드러나 판독이 정확해진다.
    """
    config = config or FeatureConfig()
    grid_h, grid_w = grid
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size

        top_left = PatchRef(
            source_image=ref.source_image,
            row=max(0, ref.row - context_cells),
            col=max(0, ref.col - context_cells),
            patch_index=0,
        )
        bottom_right = PatchRef(
            source_image=ref.source_image,
            row=min(grid_h - 1, ref.row + context_cells),
            col=min(grid_w - 1, ref.col + context_cells),
            patch_index=0,
        )
        left, top, _, _ = patch_box(top_left, grid, (width, height), config=config)
        _, _, right, bottom = patch_box(bottom_right, grid, (width, height), config=config)
        return image.crop((left, top, right, bottom))
