"""판별 7번의 면적을 **뱅크마다 다른 기준**으로 잰다.

── 왜 고쳐야 했는가 ────────────────────────────────────────────────────

`inspection/mask.py` 는 이미지 단위 임계값으로 나눠 컷오프를 잡았다.

    (거리 / 임계값) >= 컷오프          2.2 x 0.5 = 1.1

**정상부 거리가 0 근처라는 가정**이다. 실측은 그렇지 않다.

    pcb1      정상 중앙 1.708   결함 중앙 3.128
    capsules  정상 중앙 2.251   결함 중앙 2.460

둘 다 정상부가 이미 1.1 을 넘는다. 그래서 4090 실측에서 면적이 261,901px²
로 나왔다 — 512x512 가 262,144px² 이므로 **화면의 99.9%** 다. 무엇을 넣어도
"불량"이 되는 상수였다. 그 전에는 합성용 입력 크기를 실데이터에 써서 반대로
"무조건 양품" 이었다. 방향만 두 번 바뀐 셈이다.

**전역 상수로는 못 고친다.** capsules 는 정상과 결함 중앙값 차이가 0.21 뿐
이라 어떤 고정 컷오프를 잡아도 둘을 나누지 못한다.

── 어떻게 하는가 ───────────────────────────────────────────────────────

**자리마다 정상일 때의 거리를 빼고, 그 자리의 변동 폭으로 나눈다.**

    잔차[r,c] = (거리[r,c] - 중앙값[r,c]) / (중앙편차[r,c] + eps)

남는 값의 단위는 **중앙절대편차(MAD) 배수**다. "이 자리 평소 변동의 몇
배인가"이므로 뱅크가 달라도 뜻이 같고, 컷오프를 품목마다 새로 잡지 않아도
된다. `criteria.yaml` 의 컷오프가 이제 그 배수를 뜻한다.

기준선은 `inspection/baseline_map.py` 가 만들고 **뱅크와 함께 저장된다**
(`bank_meta.json` 의 `baseline`). 뱅크를 다시 만들면 기준선도 같이 갱신되므로
둘이 어긋날 수 없다.

── 기준선을 만들 이미지 ────────────────────────────────────────────────

**뱅크에 안 들어간 정상 이미지로 만든다.** 뱅크에 든 이미지는 자기 패치가
뱅크에 있어 거리가 낮게 나오고, 그 낮은 값을 기준으로 삼으면 잔차가 부풀어
면적이 다시 커진다.

기준선이 없는 뱅크도 있다. 옛 뱅크이거나 여분 정상 이미지가 모자란 경우다.
그때는 예전 방식으로 재고 **화면에 그렇게 표시한다** — 어느 기준으로 잰
값인지 모르면 그 숫자로 판정을 논할 수 없다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from scipy import ndimage

from .baseline_map import ROBUST, apply_baseline, build_baseline
from .mask import LARGEST_BLOB, TOTAL_AREA, defect_area

if TYPE_CHECKING:  # 순환 수입을 피한다 — trace 가 bank 를 쓴다
    from .bank import MemoryBank
    from .features import PatchEmbedder

#: 기준선이 있을 때의 컷오프. 자리별 변동 폭(MAD)의 몇 배부터 결함으로 볼 것인가.
#:
#: 3.0 은 "평소 변동의 세 배"이며 이상치 판정에서 흔히 쓰는 값이다. **실측으로
#: 확인된 값이 아니다** — `criteria.yaml` 에서 덮어쓰고, 4090 에서 정상·결함
#: 이미지의 면적 분포를 재서 정한다.
DEFAULT_ROBUST_CUTOFF = 3.0

#: 기준선 없이 잰 값. 품목마다 뜻이 달라 판정 기준과 대조할 수 없다.
RAW_BASIS = "raw"


def measured_area(
    patch_distances: Sequence[Sequence[float]],
    *,
    crop: int,
    threshold: float,
    baseline: dict[str, Any] | None = None,
    robust_cutoff: float = DEFAULT_ROBUST_CUTOFF,
    binarize_threshold: float = 0.5,
    aggregate: str = LARGEST_BLOB,
) -> tuple[float, str]:
    """결함 면적(픽셀)과 **무엇을 기준으로 쟀는지**.

    두 번째 값이 중요하다. 기준선이 있는 값과 없는 값은 단위가 달라 나란히
    놓을 수 없는데, 숫자만 돌려주면 그 사실이 사라진다.

    기준선이 없으면 예전 방식(`inspection.mask.defect_area`)으로 잰다.
    **떨어졌다는 사실을 감추지 않는다** — 그 값은 품목마다 뜻이 달라 판정
    기준과 대조할 수 없다.

    반환
        (면적, "robust" 또는 "raw")
    """
    grid = np.asarray(patch_distances, dtype=np.float64)
    if grid.size == 0:
        return 0.0, RAW_BASIS

    if baseline is None:
        area = defect_area(
            patch_distances, crop=crop, threshold=threshold,
            binarize_threshold=binarize_threshold, aggregate=aggregate,
        )
        return area, RAW_BASIS

    scored = apply_baseline(grid, baseline, ROBUST)
    zoom = (crop / scored.shape[0], crop / scored.shape[1])
    upsampled = ndimage.zoom(scored, zoom, order=1)
    mask = upsampled >= robust_cutoff

    if not mask.any():
        return 0.0, ROBUST
    if aggregate == TOTAL_AREA:
        return float(mask.sum()), ROBUST

    labelled, count = ndimage.label(mask)
    if count == 0:
        return 0.0, ROBUST
    sizes = ndimage.sum_labels(mask, labelled, index=range(1, count + 1))
    return float(np.max(sizes)), ROBUST


def attach_baseline(
    bank: "MemoryBank",
    normal_images: Sequence[str | Path],
    embedder: "PatchEmbedder | None" = None,
    root: str | Path | None = None,
    minimum: int = 8,
) -> bool:
    """뱅크에 자리별 기준선을 붙인다. 붙였으면 True.

    **뱅크에 안 들어간 정상 이미지를 넣어야 한다.** 뱅크에 든 이미지는 자기
    패치가 뱅크에 있어 거리가 낮게 나오고, 그 낮은 값을 기준으로 삼으면
    잔차가 부풀어 면적이 다시 커진다. 부르는 쪽이 그것을 지킨다 — 여기서는
    확인할 방법이 없다.

    minimum
        이보다 적으면 붙이지 않는다. 중앙값과 중앙절대편차는 몇 장으로
        만들면 그 몇 장의 우연을 기준으로 삼게 된다. **모자란 기준선보다
        없는 편이 낫다** — 없으면 예전 방식으로 재고 화면이 그렇다고 적지만,
        엉성한 기준선은 그럴듯한 숫자를 내면서 틀린다.
    """
    from .trace import score_images

    paths = list(normal_images)
    if len(paths) < minimum:
        return False

    results = score_images(paths, bank, embedder, root=root)
    maps = [r.patch_distances for r in results if r.patch_distances]
    if len(maps) < minimum:
        return False

    bank.meta["baseline"] = build_baseline(maps)
    return True
