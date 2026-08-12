"""메모리 뱅크 구축과 역추적 대장.

PatchCore 의 뱅크는 정상 이미지에서 뽑은 패치 벡터 모음이다. 보통 여기까지만
저장하는데, 그러면 "몇 번 패치"라는 번호만 남고 그게 어느 이미지 어느 자리에서
왔는지는 사라진다.

이 파일은 벡터와 **함께 출처(provenance)를 저장한다.** 뱅크 행 번호 하나로
원본 이미지와 격자 좌표를 되짚을 수 있어야, 미검출이 났을 때 "이 결함이
뱅크의 이 정상 패치와 가까웠고, 그 패치는 사실 이 이미지의 여기였다"까지
말할 수 있다. 이 대장이 없으면 진단이 성립하지 않는다.

저장물은 두 개다.
  bank.npz        벡터와 출처 배열. 용량이 커서 커밋하지 않는다
  bank_meta.json  구성 이력. 사람이 읽고 학습 이력 인덱서가 스캔한다
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .features import FeatureConfig, PatchEmbedder
from .types import PatchRef

_BANK_ARRAYS = "bank.npz"
_BANK_META = "bank_meta.json"


# ── coreset 선별 ────────────────────────────────────────────────────────


def greedy_coreset(
    embeddings: torch.Tensor,
    size: int,
    seed: int = 0,
    projection_dim: int | None = 128,
    device: torch.device | None = None,
) -> np.ndarray:
    """가장 멀리 떨어진 점부터 골라 나가는 방식으로 부분집합을 뽑는다.

    전체 패치를 다 담으면 뱅크가 수십만 행이 되어 조회가 느리고, 무작위로
    솎으면 드문 정상 패턴이 통째로 빠져 커버리지 부족을 스스로 만든다.
    가장 먼 점을 반복해 고르면 분포의 가장자리가 보존된다.

    projection_dim
        거리 계산 전에 무작위 사영으로 차원을 줄인다. 거리 관계는 근사적으로
        보존되면서 속도가 크게 붙는다. None 이면 원 차원 그대로 쓴다.

    반환값은 오름차순 정렬된 인덱스 배열이다. 정렬해 두어야 같은 입력에서
    항상 같은 뱅크가 나온다. 재현성 목표가 여기에 걸려 있다.
    """
    total = embeddings.shape[0]
    if size >= total:
        return np.arange(total, dtype=np.int64)
    if size < 1:
        raise ValueError(f"coreset 크기는 1 이상이어야 한다: {size}")

    device = device or embeddings.device
    generator = torch.Generator().manual_seed(seed)

    work = embeddings
    if projection_dim and projection_dim < work.shape[1]:
        projection = torch.randn(
            work.shape[1], projection_dim, generator=generator, dtype=work.dtype
        ) / np.sqrt(projection_dim)
        work = work @ projection
    work = work.to(device)

    first = int(torch.randint(total, (1,), generator=generator).item())
    selected = [first]
    min_dist = torch.linalg.vector_norm(work - work[first], dim=1)

    for _ in range(size - 1):
        nxt = int(torch.argmax(min_dist).item())
        selected.append(nxt)
        dist = torch.linalg.vector_norm(work - work[nxt], dim=1)
        min_dist = torch.minimum(min_dist, dist)

    return np.sort(np.asarray(selected, dtype=np.int64))


# ── 뱅크 ────────────────────────────────────────────────────────────────


@dataclass
class MemoryBank:
    """정상 패치 벡터 + 출처 대장.

    embeddings  (M, D) 패치 벡터
    origins     (M, 3) 각 행의 출처. 열은 (image_index, row, col)
    images      image_index 가 가리키는 이미지 경로 목록
    meta        구성 이력. 버전, 백본 설정, 시드, 원본 장수 등
    """

    embeddings: np.ndarray
    origins: np.ndarray
    images: list[str]
    meta: dict

    # ── 역추적 ──────────────────────────────────────────────────────────

    def origin_of(self, row_index: int) -> PatchRef:
        """뱅크 행 번호 → 원본 이미지와 격자 좌표.

        이 함수가 이 과제의 출발점이다. 최근접 패치를 찾은 뒤 여기를 거쳐야
        "그 패치가 결함인지 진짜 정상품인지"를 사람이나 시각 언어 모델이
        판독할 수 있다.
        """
        image_index, row, col = (int(v) for v in self.origins[row_index])
        grid_w = int(self.meta["grid"][1])
        return PatchRef(
            source_image=self.images[image_index],
            row=row,
            col=col,
            patch_index=row * grid_w + col,
        )

    def contributing_images(self) -> dict[str, int]:
        """이미지별로 뱅크에 몇 개의 패치를 남겼는지.

        커버리지 부족을 가릴 때 쓴다. 특정 조건의 이미지가 뱅크에 아예
        없거나 기여가 미미하면 그 조건은 뱅크가 모르는 상태다.
        """
        counts: dict[str, int] = {name: 0 for name in self.images}
        for image_index in self.origins[:, 0]:
            counts[self.images[int(image_index)]] += 1
        return counts

    @property
    def grid(self) -> tuple[int, int]:
        h, w = self.meta["grid"]
        return int(h), int(w)

    @property
    def version(self) -> str:
        return str(self.meta.get("bank_version", "unknown"))

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    # ── 저장과 불러오기 ─────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        """뱅크를 폴더에 저장한다.

        벡터는 npz(커밋 제외), 구성 이력은 json(사람이 읽음)으로 나눈다.
        학습 이력 인덱서는 json 만 스캔해도 뱅크 구성을 복원할 수 있다.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            directory / _BANK_ARRAYS,
            embeddings=self.embeddings,
            origins=self.origins,
        )
        meta = dict(self.meta)
        meta["images"] = self.images
        meta["patch_count"] = len(self)
        (directory / _BANK_META).write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "MemoryBank":
        directory = Path(directory)
        arrays_path = directory / _BANK_ARRAYS
        meta_path = directory / _BANK_META
        if not arrays_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"뱅크를 찾지 못했다: {directory}. 뱅크 파일은 커밋되지 않으므로 "
                f"스크립트로 다시 구성해야 한다."
            )

        arrays = np.load(arrays_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return cls(
            embeddings=arrays["embeddings"],
            origins=arrays["origins"],
            images=list(meta["images"]),
            meta=meta,
        )

    def assert_compatible(self, config: FeatureConfig) -> None:
        """추론 설정이 뱅크를 만들 때와 같은지 확인한다.

        백본이나 입력 크기가 다르면 거리 자체가 다른 척도가 되어 점수가
        조용히 틀린다. 조용히 틀리느니 여기서 멈추는 편이 낫다.
        """
        stored = self.meta.get("feature_config")
        if stored is None:
            return
        current = config.fingerprint()
        mismatched = {k: (stored.get(k), current.get(k)) for k in current if stored.get(k) != current.get(k)}
        if mismatched:
            detail = ", ".join(f"{k}: 뱅크={a} 추론={b}" for k, (a, b) in mismatched.items())
            raise ValueError(f"뱅크와 추론 설정이 다르다 ({detail}). 같은 설정으로 맞춰라.")


# ── 구축 ────────────────────────────────────────────────────────────────


#: coreset 선별은 뽑을 개수에 비례해 느려진다(O(k·N)). 이 값을 넘어가면
#: 실행 시간이 시연에 쓸 수 없는 수준이 되므로 기본 상한을 둔다.
#: 실측: 이미지 600장(패치 47만개)에 ratio 0.25 를 주면 11.8만 번을 반복한다.
DEFAULT_MAX_BANK_SIZE = 20_000


def build_bank(
    image_paths: Sequence[str | Path],
    embedder: PatchEmbedder | None = None,
    coreset_ratio: float = 0.01,
    seed: int = 0,
    bank_version: str = "v1",
    projection_dim: int | None = 128,
    batch_size: int = 8,
    root: str | Path | None = None,
    extra_meta: dict | None = None,
    max_bank_size: int | None = DEFAULT_MAX_BANK_SIZE,
) -> MemoryBank:
    """정상 이미지 목록으로 메모리 뱅크를 만든다.

    coreset_ratio
        전체 패치 중 남길 비율. 논문 기본값은 1% 다. 합성 데이터처럼 패치가
        적을 때만 이보다 크게 잡는다. 실데이터에서 0.25 같은 값을 주면
        뱅크가 십만 행대가 되어 선별이 끝나지 않는다.
    max_bank_size
        coreset 결과 행 수의 상한. 비율만 보고 크게 잡았다가 실행이 멈추는
        것을 막는다. 상한이 걸리면 meta["coreset_capped"] 에 남기므로
        조용히 잘리지 않는다. None 이면 상한을 두지 않는다.
    root
        주면 이미지 경로를 이 기준의 상대 경로로 기록한다. 사람마다 다른
        절대 경로가 뱅크에 박히는 것을 막는다.
    seed
        coreset 선별의 무작위 요소를 고정한다. 같은 입력이면 같은 뱅크가
        나와야 게이트 판정 재현성이 성립한다.
    """
    if not image_paths:
        raise ValueError("뱅크를 만들 정상 이미지가 없다.")
    if not 0 < coreset_ratio <= 1:
        raise ValueError(f"coreset_ratio 는 0 초과 1 이하여야 한다: {coreset_ratio}")

    embedder = embedder or PatchEmbedder()
    patches, grid = embedder.embed_paths(image_paths, batch_size=batch_size)
    n_images, per_image, dim = patches.shape

    flat = patches.reshape(n_images * per_image, dim)

    # 평탄화 순서를 그대로 좌표로 되돌린다. features.py 가 행 우선을 보장한다.
    grid_h, grid_w = grid
    image_index = np.repeat(np.arange(n_images, dtype=np.int64), per_image)
    rows = np.tile(np.repeat(np.arange(grid_h, dtype=np.int64), grid_w), n_images)
    cols = np.tile(np.tile(np.arange(grid_w, dtype=np.int64), grid_h), n_images)
    origins_all = np.stack([image_index, rows, cols], axis=1)

    requested = max(1, int(round(flat.shape[0] * coreset_ratio)))
    target = requested
    capped = False
    if max_bank_size is not None and target > max_bank_size:
        target = max_bank_size
        capped = True

    keep = greedy_coreset(
        flat, size=target, seed=seed, projection_dim=projection_dim, device=embedder.device
    )

    root_path = Path(root) if root else None

    def to_name(p: str | Path) -> str:
        p = Path(p)
        if root_path:
            try:
                return p.relative_to(root_path).as_posix()
            except ValueError:
                pass
        return p.as_posix()

    meta = {
        "bank_version": bank_version,
        "grid": [grid_h, grid_w],
        "embedding_dim": int(dim),
        "source_image_count": int(n_images),
        "patches_per_image": int(per_image),
        "total_patches_before_coreset": int(flat.shape[0]),
        "coreset_ratio": coreset_ratio,
        "coreset_requested": int(requested),
        "coreset_capped": capped,
        "max_bank_size": max_bank_size,
        "seed": seed,
        "projection_dim": projection_dim,
        "feature_config": embedder.config.fingerprint(),
    }
    if capped:
        # 조용히 잘리면 "비율대로 만들어졌다"고 오해하게 된다.
        meta["coreset_note"] = (
            f"coreset_ratio {coreset_ratio} 는 {requested:,}행을 요구했으나 "
            f"max_bank_size {max_bank_size:,} 로 제한했다. 비율을 낮추거나 "
            f"max_bank_size 를 올려라."
        )
    if extra_meta:
        meta.update(extra_meta)

    return MemoryBank(
        embeddings=flat[keep].numpy().astype(np.float32),
        origins=origins_all[keep],
        images=[to_name(p) for p in image_paths],
        meta=meta,
    )
