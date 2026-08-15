"""이상맵에서 결함 면적을 낸다 — 판별 항목 7번의 입력.

`data/criteria.yaml` 이 면적으로 불량을 판정한다. 그 면적을 여기서 낸다.

── 왜 격자 칸을 세면 안 되는가 ─────────────────────────────────────────

전에는 이렇게 했다.

    cell = (crop / grid_h) * (crop / grid_w)
    hot  = 임계값을 넘은 격자 칸 수
    면적 = hot * cell

세 가지가 어긋났다.

  1. **격자 해상도로만 잰다.** 격자 한 칸은 원본에서 8×8px = 64px² 다
     (실측: crop 448 → 격자 56×56, crop 64 → 격자 8×8. 칸 크기는 백본
     스트라이드가 정하므로 입력 크기와 무관하게 8px 이다).

     기준이 150px² 이므로 **두 칸(128px²)은 통과, 세 칸(192px²)은 불량**이다.
     면적으로 판정한다는 것이 실제로는 **칸 경계에서 뛰는 계단 함수**가 된다.
     40px 결함이 어떻게 놓이느냐에 따라 두 칸에 걸치기도 세 칸에 걸치기도
     하는데, 그 차이로 판정이 갈린다
  2. **흩어진 칸을 다 더한다.** `criteria.yaml` 은 `largest_blob`(가장 큰
     덩어리)을 요구하는데 `total_area` 를 낸다. 잡음 다섯 칸이 흩어진 것과
     결함 다섯 칸이 뭉친 것이 같은 값이 된다
  3. **기준 파일의 컷오프를 안 읽는다.** `binarize_threshold` 가 있는데
     코드가 따로 정한 값을 썼다

── 어떻게 재는가 ───────────────────────────────────────────────────────

    격자 이상맵 → 이미지 해상도로 업샘플 → 정규화 → 이진화 → 연결 성분
                                                          → 가장 큰 덩어리

**정규화는 운영 임계값으로 한다.** 격자값을 임계값으로 나누면 1.0 이
"임계값과 같은 거리"라는 뜻이 되고, `binarize_threshold: 0.5` 는 "임계값의
절반"이 된다. 실측에서 정상 이미지는 최대 0.05, 결함 이미지는 1.0 근처라
이 컷오프가 둘을 가른다.

이미지마다 min-max 로 정규화하지 **않는다.** 그러면 멀쩡한 정상 이미지도
자기 안에서 가장 밝은 곳이 1.0 이 되어 없는 결함이 생긴다.

새 모델이 필요 없다. 이상맵은 PatchCore 추론이 이미 내놓는 값이다.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

#: 면적을 어떻게 합칠 것인가. `criteria.yaml` 의 `measurement.aggregate`.
LARGEST_BLOB = "largest_blob"
TOTAL_AREA = "total_area"


def anomaly_mask(
    patch_distances: list[list[float]],
    *,
    crop: int,
    threshold: float,
    binarize_threshold: float = 0.5,
) -> np.ndarray:
    """격자 이상맵을 이미지 해상도의 이진 마스크로.

    crop
        추론에 쓴 입력 한 변의 픽셀 수. **추론에 쓴 설정과 같아야 한다** —
        합성은 `DEMO_CONFIG.crop`(64), VisA 는 `VISA_CONFIG.crop`(448) 이다.

        한 번 합성 값을 VisA 경로에 그대로 써서 **면적이 49배 작게** 나온
        적이 있다. (448/64)² = 49 다. 칸 면적이 64px² 대신 1.31px² 로 나와
        기준 150px² 를 **어떤 결함으로도 못 넘었고, 실데이터에서 판별 7번이
        무조건 "양품" 이었다.**
    threshold
        운영 임계값. 정규화 기준이다.
    binarize_threshold
        정규화한 값이 이 이상이면 결함으로 본다.
    """
    grid = np.asarray(patch_distances, dtype=np.float64)
    if grid.size == 0 or threshold <= 0:
        return np.zeros((crop, crop), dtype=bool)

    # 격자 → 이미지 해상도. order=1 이 bilinear 다. 계단이 남으면 연결
    # 성분이 실제보다 잘게 쪼개진다.
    zoom = (crop / grid.shape[0], crop / grid.shape[1])
    upsampled = ndimage.zoom(grid, zoom, order=1)
    return (upsampled / threshold) >= binarize_threshold


def defect_area(
    patch_distances: list[list[float]],
    *,
    crop: int,
    threshold: float,
    binarize_threshold: float = 0.5,
    aggregate: str = LARGEST_BLOB,
) -> float:
    """결함 면적(픽셀). 판별 7번이 판정 기준과 대조하는 값이다.

    `largest_blob` 이 기본이다. 흩어진 잡음을 더해 기준을 넘기면 **멀쩡한
    이미지가 불량으로 나가고**, 그것을 "기준 문제"로 오진한다.
    """
    mask = anomaly_mask(
        patch_distances, crop=crop, threshold=threshold,
        binarize_threshold=binarize_threshold,
    )
    if not mask.any():
        return 0.0
    if aggregate == TOTAL_AREA:
        return float(mask.sum())

    labelled, count = ndimage.label(mask)
    if count == 0:
        return 0.0
    sizes = ndimage.sum_labels(mask, labelled, index=range(1, count + 1))
    return float(np.max(sizes))


def blob_count(
    patch_distances: list[list[float]],
    *,
    crop: int,
    threshold: float,
    binarize_threshold: float = 0.5,
) -> int:
    """덩어리가 몇 개인가.

    하나면 결함 하나, 여럿이면 흩어진 잡음일 수 있다. 면적만으로는 그
    구분이 안 되므로 근거에 함께 남긴다.
    """
    mask = anomaly_mask(
        patch_distances, crop=crop, threshold=threshold,
        binarize_threshold=binarize_threshold,
    )
    if not mask.any():
        return 0
    _labelled, count = ndimage.label(mask)
    return int(count)
