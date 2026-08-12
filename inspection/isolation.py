"""뱅크 안에서 고립된 패치 찾기 — 오염 후보 선별.

정상 패치들은 서로 비슷해서 뭉쳐 있고, 잘못 섞여 들어간 결함 패치는 이웃이
멀다. 이 차이를 재면 시각 언어 모델 없이도 오염 후보를 좁힐 수 있다.

**이것은 판정이 아니라 선별이다.** 고립된 패치가 오염원일 수도 있지만,
드물지만 진짜인 정상품일 수도 있다(조명이 다른 로트, 흔치 않은 형상).
둘을 가르는 것은 여전히 판별 항목 5번의 몫이다. 여기서 하는 일은 5번에
넘길 후보를 몇 개로 줄이는 것이며, 그 자체로 두 가지 값어치가 있다.

  - 시각 언어 모델 호출 수가 줄어든다
  - 역추적이 지목한 패치가 고립도까지 높으면 근거가 겹쳐 단단해진다

측정으로 확인한 한계 두 가지를 코드에 반영해 두었다.

1. coreset 비율이 낮을수록 신호가 나빠진다
   greedy coreset 은 서로 먼 점을 우선 남기므로 결함 패치가 우선 선택된다.
   합성 데이터에서 원본 이미지의 12.5% 였던 오염원이 뱅크의 50~70% 를
   차지했다. 뱅크가 이미 오염 패치로 채워지면 "고립"의 기준선 자체가 올라가
   대비가 흐려진다.

2. 기여 패치가 적은 이미지는 순위가 요동친다
   패치 한두 개만 남은 이미지는 평균이 우연에 좌우된다. 실제로 그런 정상
   이미지가 오염원보다 높은 순위를 차지하는 경우를 확인했다. 그래서
   min_patches 미만은 순위에서 제외한다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch

from .bank import MemoryBank

#: 이미지 단위 순위에 넣기 위한 최소 기여 패치 수.
#: 이보다 적으면 평균이 우연에 좌우되어 순위가 뒤집힌다.
DEFAULT_MIN_PATCHES = 3


@dataclass
class IsolationScore:
    """이미지 하나의 고립도.

    z_mean
        그 이미지가 남긴 패치들의 고립도 z 점수 평균. 클수록 뱅크의 다른
        패치들과 멀다.
    patch_count
        그 이미지가 뱅크에 남긴 패치 수. 적으면 z_mean 을 믿기 어렵다.
    ranked
        순위에 포함됐는가. patch_count 가 기준 미만이면 False.
    """

    image: str
    z_mean: float
    z_max: float
    patch_count: int
    ranked: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def patch_isolation(
    bank: MemoryBank,
    k: int = 8,
    device: torch.device | None = None,
    chunk: int = 4096,
) -> np.ndarray:
    """뱅크 각 행의 고립도를 z 점수로 돌려준다.

    고립도는 그 행에서 가장 가까운 다른 k 개 행까지의 평균 거리다.
    절대값은 뱅크마다 척도가 달라 의미가 없으므로 z 점수로 정규화한다.

    반환  (M,) 배열. 뱅크 행 순서와 같다.
    """
    total = len(bank)
    if total <= k + 1:
        raise ValueError(f"뱅크가 너무 작다 ({total}행). k={k} 보다 충분히 커야 한다.")

    embeddings = torch.from_numpy(bank.embeddings)
    if device is not None:
        embeddings = embeddings.to(device)

    means = torch.empty(total, dtype=torch.float32, device=embeddings.device)
    for start in range(0, total, chunk):
        block = embeddings[start : start + chunk]
        dist = torch.cdist(block, embeddings)
        # 자기 자신까지의 거리 0 을 제외한다.
        rows = torch.arange(block.shape[0], device=dist.device)
        dist[rows, rows + start] = float("inf")
        means[start : start + block.shape[0]] = dist.topk(k=k, dim=1, largest=False).values.mean(dim=1)

    values = means.cpu().numpy()
    std = values.std()
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def image_isolation(
    bank: MemoryBank,
    k: int = 8,
    min_patches: int = DEFAULT_MIN_PATCHES,
    device: torch.device | None = None,
) -> list[IsolationScore]:
    """이미지 단위로 고립도를 모아 순위를 매긴다.

    패치 하나하나보다 이미지 단위가 쓸모 있다. 큐레이션에서 빼고 넣는 단위가
    이미지이고, 여러 패치의 평균이라 우연에 덜 흔들리기 때문이다.

    반환은 z_mean 내림차순이며, 기여 패치가 min_patches 미만인 이미지는
    ranked=False 로 표시해 뒤로 보낸다. 지우지 않고 남기는 이유는 "왜 이
    이미지는 순위에 없나"에 답할 수 있어야 하기 때문이다.
    """
    z = patch_isolation(bank, k=k, device=device)

    grouped: dict[str, list[float]] = {}
    for row_index in range(len(bank)):
        name = bank.origin_of(row_index).source_image
        grouped.setdefault(name, []).append(float(z[row_index]))

    scores = [
        IsolationScore(
            image=name,
            z_mean=float(np.mean(values)),
            z_max=float(np.max(values)),
            patch_count=len(values),
            ranked=len(values) >= min_patches,
        )
        for name, values in grouped.items()
    ]

    # 순위에 드는 것부터, 그 안에서 z_mean 내림차순.
    scores.sort(key=lambda s: (s.ranked, s.z_mean), reverse=True)
    return scores


def suspect_images(
    bank: MemoryBank,
    k: int = 8,
    min_patches: int = DEFAULT_MIN_PATCHES,
    z_threshold: float = 1.0,
    top_n: int | None = None,
) -> list[IsolationScore]:
    """오염이 의심되는 이미지 후보를 추린다.

    z_threshold
        이 값을 넘는 이미지만 후보로 본다. 1.0 은 자리표시이며, 실제 값은
        시나리오로 측정해 정해야 한다.
    top_n
        주면 상위 이 개수까지만 돌려준다. 시각 언어 모델 호출 수를 묶는 용도다.

    후보가 비어 있을 수 있다. 그때는 "고립도로는 의심 대상을 찾지 못했다"가
    결론이며, 억지로 상위 몇 개를 뽑아 오염으로 몰지 않는다.
    """
    candidates = [s for s in image_isolation(bank, k=k, min_patches=min_patches) if s.ranked]
    candidates = [s for s in candidates if s.z_mean >= z_threshold]
    return candidates[:top_n] if top_n else candidates


def contamination_amplification(bank: MemoryBank) -> dict[str, Any]:
    """coreset 이 특정 이미지의 기여를 얼마나 부풀렸는지 본다.

    greedy coreset 은 서로 먼 점을 남기므로, 튀는 이미지의 패치가 우선
    선택된다. 그 결과 원본에서 소수였던 이미지가 뱅크에서는 다수가 될 수 있다.
    오염이 섞였을 때 이 증폭이 그대로 피해를 키우므로, 뱅크를 만든 뒤
    한 번 확인할 값이다.

    반환의 amplification 은 (뱅크 내 비중) / (원본 이미지 비중) 이다.
    1.0 이면 고르게 남았다는 뜻이고, 크면 그 이미지가 과대 대표됐다는 뜻이다.
    """
    counts = bank.contributing_images()
    total_rows = sum(counts.values())
    image_count = len(bank.images)
    if total_rows == 0 or image_count == 0:
        return {"images": [], "max_amplification": 0.0}

    even_share = 1.0 / image_count
    rows = []
    for name, count in counts.items():
        share = count / total_rows
        rows.append(
            {
                "image": name,
                "patch_count": count,
                "bank_share": share,
                "amplification": share / even_share,
            }
        )
    rows.sort(key=lambda r: -r["amplification"])
    return {
        "images": rows,
        "max_amplification": rows[0]["amplification"] if rows else 0.0,
        "note": (
            "amplification 이 크면 그 이미지가 뱅크를 과대 대표한다. "
            "오염된 이미지라면 피해가 그만큼 커진다."
        ),
    }
