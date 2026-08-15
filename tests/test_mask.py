"""이상맵에서 면적을 내는 값들 — 판별 7번의 바탕.

격자 한 칸이 몇 픽셀인지가 판별 7번 전체를 좌우한다. 칸 크기를 잘못 알면
"기준 150px² 를 몇 칸에서 넘는가" 가 통째로 달라지고, 그 서술이 문서와
기획서로 퍼진다. 실제로 퍼졌다.
"""

from __future__ import annotations

# ── 격자 한 칸이 몇 픽셀인가 ────────────────────────────────────────────
#
# **문서에 틀린 숫자가 두 곳 퍼져 있었다.** "448px 입력에 28×28 격자라 칸
# 하나가 256px²" 라고 적어 두고 기획서까지 옮겼는데, 실측은 56×56 이고 칸은
# 64px² 다. 4090 이 잡았다.
#
# 칸 크기는 백본 스트라이드가 정하므로 **입력 크기와 무관하게 8px** 이다.
# 이 값이 바뀌면 판별 7번의 면적 판정 전체가 바뀐다.


def test_a_grid_cell_is_eight_pixels_wide():
    """격자 한 칸 = 원본 8×8px = 64px². 입력 크기와 무관하다."""
    import os
    import tempfile

    import numpy as np
    from PIL import Image

    from inspection import FeatureConfig, PatchEmbedder, build_bank, score_image

    for crop in (64, 448):
        embedder = PatchEmbedder(FeatureConfig(resize=crop, crop=crop))
        folder = tempfile.mkdtemp()
        paths = []
        for i in range(2):
            path = os.path.join(folder, f"n{i}.png")
            Image.fromarray((np.random.rand(crop, crop, 3) * 255).astype("uint8")).save(path)
            paths.append(path)
        result = score_image(paths[0], build_bank(paths, embedder), embedder)

        assert result.grid_h == crop // 8, (
            f"crop {crop} 의 격자가 {result.grid_h} 다 — {crop // 8} 이어야 한다"
        )
        assert (crop / result.grid_h) ** 2 == 64.0


def test_the_criteria_threshold_takes_three_cells():
    """기준 150px² 는 두 칸(128)으로 못 넘고 세 칸(192)에서 넘는다.

    **"한 칸만 켜져도 불량" 이 아니다.** 면적 판정이 칸 경계에서 뛰는 계단
    함수라는 것이 실제 문제이고, 그 서술을 문서가 정확히 담아야 한다.
    """
    cell = 64.0
    assert 2 * cell < 150 <= 3 * cell
