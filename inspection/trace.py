"""추론과 최근접 패치 역추적.

이 파일이 하는 일은 두 가지다.

  1. 이미지의 이상 점수를 낸다 — 판별 항목 3번의 재료
  2. 점수를 만든 자리가 뱅크의 **어느 정상 패치**와 가까웠는지 되짚는다
     — 판별 항목 4번

2번이 이 과제의 기술적 토대다. 최근접 패치를 원본 이미지와 좌표까지
되돌려 놓으면, 그 패치가 잘못 섞인 결함인지 진짜 정상품인지를 판독해
뱅크 오염과 정상 분포 중첩을 가를 수 있다. 조치가 정반대인 두 원인이
여기서 갈린다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .bank import MemoryBank
from .features import PatchEmbedder
from .types import InferenceResult, NearestMatch, PatchRef


def _nearest_in_bank(
    queries: torch.Tensor,
    bank: torch.Tensor,
    chunk: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """질의 패치마다 가장 가까운 뱅크 행과 그 거리를 찾는다.

    뱅크가 커지면 거리 행렬을 통째로 만들 수 없으므로 뱅크를 잘라가며
    최소값만 갱신한다. 결과는 전체를 한 번에 계산한 것과 같다.

    반환  (거리 (P,), 뱅크 행 번호 (P,))
    """
    best_dist: torch.Tensor | None = None
    best_index: torch.Tensor | None = None

    for start in range(0, bank.shape[0], chunk):
        block = bank[start : start + chunk]
        dist = torch.cdist(queries, block)  # (P, chunk)
        dist_min, arg_min = dist.min(dim=1)
        arg_min = arg_min + start

        if best_dist is None:
            best_dist, best_index = dist_min, arg_min
        else:
            improved = dist_min < best_dist
            best_dist = torch.where(improved, dist_min, best_dist)
            best_index = torch.where(improved, arg_min, best_index)

    assert best_dist is not None and best_index is not None
    return best_dist, best_index


def _reweight(
    query_patch: torch.Tensor,
    bank: torch.Tensor,
    bank_index: int,
    raw_score: float,
    neighbors: int,
) -> float:
    """PatchCore 의 이미지 점수 보정.

    가장 이상한 패치가 뱅크에서 외딴 지점과 짝지어졌다면, 그 지점 자체가
    뱅크에서 드문 패치라는 뜻이므로 거리를 곧이곧대로 믿기 어렵다.
    최근접 뱅크 패치 주변이 얼마나 촘촘한지를 반영해 점수를 깎는다.

    수식은 논문과 같고, 지수 계산은 softmax 로 안정화했다.
    """
    if neighbors <= 1 or bank.shape[0] <= 1:
        return raw_score

    anchor = bank[bank_index : bank_index + 1]
    to_anchor = torch.cdist(anchor, bank).squeeze(0)  # (M,)
    k = min(neighbors, bank.shape[0])
    neighbor_idx = torch.topk(to_anchor, k=k, largest=False).indices

    # 질의 패치에서 그 이웃들까지의 거리. 첫 항이 최근접(=raw_score)이 되도록 정렬한다.
    dist_to_neighbors = torch.cdist(query_patch.unsqueeze(0), bank[neighbor_idx]).squeeze(0)
    weights = torch.softmax(dist_to_neighbors, dim=0)
    nearest_pos = int(torch.argmin(dist_to_neighbors).item())
    w = float(1.0 - weights[nearest_pos].item())
    return w * raw_score


def score_image(
    image_path: str | Path,
    bank: MemoryBank,
    embedder: PatchEmbedder | None = None,
    top_k: int = 5,
    reweight: bool = True,
    reweight_neighbors: int = 9,
    root: str | Path | None = None,
) -> InferenceResult:
    """이미지 한 장을 추론하고 최근접 패치까지 되짚는다.

    top_k
        결과에 담을 상위 패치 수. matches[0] 이 판정을 좌우한 자리다.
    reweight
        이미지 점수 보정 여부. 임계값 스윕처럼 척도를 고정해야 하는
        곳에서는 꺼서 원 거리를 쓴다.
    """
    embedder = embedder or PatchEmbedder()
    bank.assert_compatible(embedder.config)

    patches, grid = embedder.embed_paths([image_path], batch_size=1)
    queries = patches[0].to(embedder.device)  # (P, D)
    bank_tensor = torch.from_numpy(bank.embeddings).to(embedder.device)

    distances, bank_indices = _nearest_in_bank(queries, bank_tensor)

    worst = int(torch.argmax(distances).item())
    raw_score = float(distances[worst].item())
    score = (
        _reweight(queries[worst], bank_tensor, int(bank_indices[worst].item()), raw_score, reweight_neighbors)
        if reweight
        else raw_score
    )

    grid_h, grid_w = grid
    image_name = Path(image_path)
    if root:
        try:
            image_name = image_name.relative_to(Path(root))
        except ValueError:
            pass
    image_name = image_name.as_posix()

    # 거리가 큰 순서로 상위 패치를 뽑아 각각의 뱅크 출처를 붙인다.
    order = torch.argsort(distances, descending=True)[: max(0, top_k)]
    matches: list[NearestMatch] = []
    for pos in order.tolist():
        row, col = divmod(pos, grid_w)
        bank_row = int(bank_indices[pos].item())
        matches.append(
            NearestMatch(
                query=PatchRef(source_image=image_name, row=row, col=col, patch_index=pos),
                bank=bank.origin_of(bank_row),
                distance=float(distances[pos].item()),
                bank_row_index=bank_row,
            )
        )

    return InferenceResult(
        image=image_name,
        score=score,
        max_patch_distance=raw_score,
        grid_h=grid_h,
        grid_w=grid_w,
        matches=matches,
        patch_distances=distances.reshape(grid_h, grid_w).cpu().tolist(),
        bank_version=bank.version,
    )


def score_images(
    image_paths: Sequence[str | Path],
    bank: MemoryBank,
    embedder: PatchEmbedder | None = None,
    **kwargs,
) -> list[InferenceResult]:
    """여러 장을 순서대로 추론한다. 순서는 입력과 같다."""
    embedder = embedder or PatchEmbedder()
    return [score_image(p, bank, embedder=embedder, **kwargs) for p in image_paths]


def anomaly_map(result: InferenceResult) -> np.ndarray:
    """격자 거리 맵을 배열로. 마스크 면적 판정과 시각화에 쓴다."""
    return np.asarray(result.patch_distances, dtype=np.float32)


def bank_contribution(results: Sequence[InferenceResult]) -> dict[str, int]:
    """미검출 건들이 어느 뱅크 이미지와 짝지어졌는지 센다.

    같은 정상 이미지가 여러 미검출 건의 최근접으로 반복해서 나오면 그
    이미지를 의심할 근거가 된다. 뱅크 오염을 좁혀 들어가는 첫 신호다.
    한 장의 우연이 아니라 반복이 근거이므로 건수로 센다.
    """
    counts: dict[str, int] = {}
    for result in results:
        top = result.top_match
        if top is None:
            continue
        counts[top.bank.source_image] = counts.get(top.bank.source_image, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
