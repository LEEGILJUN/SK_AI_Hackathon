"""테스트용 합성 이미지 생성.

VisA 가 준비되기 전에도 추론과 역추적이 도는지 확인하려고 만든 것이다.
가상 공장 데이터(data/build_factory.py, 이동현 담당)를 대신하지 않는다.
시나리오 검증은 실제 VisA 로 해야 한다.

정상은 규칙적인 격자 무늬, 결함은 그 위에 찍힌 어두운 얼룩이다. 사람 눈에
명확히 다르므로, 여기서 구분하지 못하면 추론 쪽 문제라고 볼 수 있다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 128
_TILE = 16


def _base_pattern() -> np.ndarray:
    """정상 제품의 기본 무늬. 규칙적인 체크 패턴."""
    canvas = np.full((IMAGE_SIZE, IMAGE_SIZE), 110, dtype=np.float32)
    for row in range(0, IMAGE_SIZE, _TILE):
        for col in range(0, IMAGE_SIZE, _TILE):
            if ((row // _TILE) + (col // _TILE)) % 2 == 0:
                canvas[row : row + _TILE, col : col + _TILE] = 165
    return canvas


def make_normal(seed: int) -> Image.Image:
    """정상 이미지. 미세한 노이즈와 밝기 흔들림만 준다."""
    rng = np.random.default_rng(seed)
    canvas = _base_pattern()
    canvas += rng.normal(0.0, 3.0, canvas.shape)
    canvas += rng.uniform(-6.0, 6.0)  # 장마다 조금씩 다른 조명
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([canvas] * 3, axis=-1))


def make_defect(seed: int, radius: int = 9) -> Image.Image:
    """결함 이미지. 정상 무늬 위에 어두운 얼룩이 하나 찍힌다.

    얼룩의 모양과 밝기는 일정하고 위치만 흔들린다. 그래야 오염된 뱅크가
    같은 종류의 결함을 정상으로 받아들이는 상황이 재현된다.
    """
    rng = np.random.default_rng(seed)
    canvas = np.asarray(make_normal(seed), dtype=np.float32)[:, :, 0]

    margin = radius + 12
    cy = int(rng.integers(margin, IMAGE_SIZE - margin))
    cx = int(rng.integers(margin, IMAGE_SIZE - margin))

    yy, xx = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    blob = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    canvas[blob] = 25.0

    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([canvas] * 3, axis=-1))


def write_set(
    directory: Path,
    count: int,
    kind: str = "normal",
    seed_offset: int = 0,
) -> list[Path]:
    """이미지를 폴더에 저장하고 경로 목록을 돌려준다."""
    directory.mkdir(parents=True, exist_ok=True)
    maker = make_normal if kind == "normal" else make_defect

    paths: list[Path] = []
    for i in range(count):
        image = maker(seed_offset + i)
        path = directory / f"{kind}_{i:03d}.png"
        image.save(path)
        paths.append(path)
    return paths
