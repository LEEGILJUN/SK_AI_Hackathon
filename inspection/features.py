"""패치 임베딩 추출.

PatchCore는 이미지를 격자로 잘라 각 칸의 특징 벡터를 만들고, 그 벡터가
정상 패치 뱅크에서 얼마나 떨어져 있는지로 이상 여부를 판정한다.
이 파일은 "이미지 → 격자 패치 벡터"까지를 담당한다.

중간층 두 개(layer2, layer3)를 쓰는 이유는 원 논문과 같다. 너무 얕으면
ImageNet 특유의 저수준 무늬에 휘둘리고, 너무 깊으면 분류에 치우쳐
위치 정보가 뭉개진다.

격자 좌표(row, col)를 끝까지 유지하는 것이 이 파일의 핵심이다.
이 좌표가 있어야 나중에 "뱅크의 몇 번 패치가 원본 이미지 어디였는지"를
되짚을 수 있고, 그게 이 과제 전체의 토대다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import get_model

# ImageNet 정규화 상수. 사전학습 백본을 쓰므로 학습 시점과 맞춰야 한다.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class FeatureConfig:
    """패치 임베딩 설정.

    이 값이 바뀌면 뱅크와 추론 결과가 호환되지 않는다. 그래서 뱅크를 저장할 때
    설정을 함께 기록하고, 불러올 때 대조한다(bank.py 참조).

    backbone
        torchvision 모델 이름. 기준은 wide_resnet50_2 이며 현행 검사 모델과
        맞추기 위한 선택이다. 가중치 내려받기가 막힌 환경에서는 resnet18 처럼
        이미 캐시된 백본으로 바꿔 끼울 수 있다.
    crop
        **원본을 crop×crop 정사각으로 리사이즈해서 쓴다.** 비율은 깨지지만
        잘리는 곳이 없고, 카테고리마다 다르게 맞출 것도 없다.

        전에는 짧은 변을 맞춘 뒤 중앙을 잘랐다(MVTec 관행). VisA 는 정사각이
        아니라 가로가 길어서 **양옆이 잘려 나갔다.** 정답 마스크로 세니 pcb3
        은 40건 중 12건이 일부 잘리고 capsules 는 1건이 통째로 사라졌다.
        사라진 건 어떤 모델로도 검출할 수 없다.

        anomalib 의 VisA 기본값도 이 형태다(`center_crop: null`).

        448 은 VisA 실측으로 정했다. 흔히 쓰는 224 로는 결함을 거의 못 잡는다.

            224, 정상 60장   AUROC 0.526   ← 무작위 수준
            448, 정상 60장   AUROC 0.931
            224, 정상 150장  AUROC 0.502   ← 뱅크만 키워도 소용없다
            448, 정상 150장  AUROC 0.998

        이유는 결함 크기다. VisA capsules 의 결함은 원본(1500x1000)에서
        40x40px, 전체 면적의 0.1% 다. 224 입력이면 격자 한 칸이 원본 31px 라
        결함이 딱 한 칸이고, 아래 neighborhood 평균에 주변 정상 8칸과 섞여
        희석된다. 448 이면 결함이 여러 칸에 걸친다.

        **뱅크 크기로는 해결되지 않는다.** 데이터를 더 넣어도 해상도가
        모자라면 신호 자체가 없다.

        위 표는 중앙 크롭으로 잰 값이다. 정사각 리사이즈로 다시 재야 한다.
    neighborhood
        각 격자 칸의 특징을 주변 이웃과 평균낼 때의 커널 크기. 논문의
        local neighborhood aggregation 이며, 미세한 위치 흔들림을 흡수한다.
        결함이 격자 한 칸보다 작으면 여기서 희석된다.
    """

    backbone: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    weights: str | None = "IMAGENET1K_V1"
    crop: int = 448
    neighborhood: int = 3

    def fingerprint(self) -> dict:
        """뱅크에 함께 저장할 설정 지문."""
        return asdict(self) | {"layers": list(self.layers)}


class PatchEmbedder:
    """이미지를 격자 패치 벡터로 바꾼다.

    백본은 고정이고 학습하지 않는다. PatchCore 는 역전파가 없으므로
    이 클래스는 순전히 특징 추출기다.
    """

    def __init__(self, config: FeatureConfig | None = None, device: torch.device | None = None):
        from .device import pick_device

        self.config = config or FeatureConfig()
        self.device = device or pick_device()

        try:
            model = get_model(self.config.backbone, weights=self.config.weights)
        except Exception as exc:  # 네트워크 차단 환경에서 가중치 캐시가 없을 때
            raise RuntimeError(
                f"백본 '{self.config.backbone}' 을 준비하지 못했다. 사전학습 가중치가 "
                f"캐시(~/.cache/torch/hub/checkpoints)에 없고 네트워크가 막혀 있으면 "
                f"이 오류가 난다. 온라인 상태에서 한 번 받아두거나, 이미 받아둔 "
                f"백본으로 FeatureConfig(backbone=...) 를 바꿔라. 원인: {exc}"
            ) from exc

        self.model = model.eval().to(self.device)
        for param in self.model.parameters():
            param.requires_grad_(False)

        # 지정한 중간층의 출력을 가로챈다.
        self._captured: dict[str, torch.Tensor] = {}
        for name in self.config.layers:
            module = getattr(self.model, name, None)
            if module is None:
                raise ValueError(
                    f"백본 '{self.config.backbone}' 에 '{name}' 층이 없다. "
                    f"FeatureConfig(layers=...) 를 확인하라."
                )
            module.register_forward_hook(self._make_hook(name))

        # **정사각으로 리사이즈한다. 자르지 않는다.** 튜플을 주면
        # 비율을 무시하고 정확히 그 크기로 맞춘다.
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.crop, self.config.crop)),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._captured[name] = output

        return hook

    # ── 공개 API ────────────────────────────────────────────────────────

    @torch.no_grad()
    def embed_batch(self, batch: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """정규화까지 끝난 배치 텐서를 패치 벡터로 바꾼다.

        입력  (B, 3, H, W)
        출력  (B, P, D) 와 격자 크기 (Hg, Wg). P = Hg * Wg 이며
              패치 순서는 행 우선(row-major)이다. 이 순서가 격자 좌표를
              복원하는 근거이므로 어디서도 뒤섞지 않는다.
        """
        self._captured.clear()
        self.model(batch.to(self.device))

        feats = [self._captured[name] for name in self.config.layers]

        # 각 층에서 지역 이웃 평균. 미세한 위치 흔들림을 흡수한다.
        pad = self.config.neighborhood // 2
        feats = [
            F.avg_pool2d(f, kernel_size=self.config.neighborhood, stride=1, padding=pad)
            for f in feats
        ]

        # 가장 해상도가 높은 층(첫 층)의 격자에 나머지를 맞춘다.
        target_hw = feats[0].shape[-2:]
        feats = [
            f if f.shape[-2:] == target_hw else F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
            for f in feats
        ]

        merged = torch.cat(feats, dim=1)  # (B, D, Hg, Wg)
        b, d, hg, wg = merged.shape
        patches = merged.permute(0, 2, 3, 1).reshape(b, hg * wg, d)  # 행 우선
        return patches.contiguous(), (hg, wg)

    def embed_images(self, images: Iterable[Image.Image]) -> tuple[torch.Tensor, tuple[int, int]]:
        """PIL 이미지 여러 장을 한 배치로 임베딩한다."""
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images])
        return self.embed_batch(batch)

    def embed_paths(
        self, paths: Sequence[str | Path], batch_size: int = 8
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """이미지 경로 목록을 임베딩한다.

        출력 (N, P, D) 의 첫 축 순서는 입력 paths 순서와 같다.
        이 대응이 곧 provenance 의 image_index 가 된다.
        """
        if not paths:
            raise ValueError("임베딩할 이미지 경로가 비어 있다.")

        chunks: list[torch.Tensor] = []
        grid: tuple[int, int] | None = None

        for start in range(0, len(paths), batch_size):
            window = paths[start : start + batch_size]
            images = [Image.open(p) for p in window]
            try:
                patches, this_grid = self.embed_images(images)
            finally:
                for img in images:
                    img.close()

            if grid is None:
                grid = this_grid
            elif grid != this_grid:
                raise RuntimeError(
                    f"격자 크기가 배치마다 다르다: {grid} vs {this_grid}. "
                    f"입력 이미지 크기가 제각각인지 확인하라."
                )
            chunks.append(patches.cpu())

        assert grid is not None
        return torch.cat(chunks, dim=0), grid

    @property
    def embedding_dim(self) -> int:
        """패치 벡터 차원. 백본과 층 조합에 따라 정해진다."""
        dummy = torch.zeros(1, 3, self.config.crop, self.config.crop)
        patches, _ = self.embed_batch(dummy)
        return int(patches.shape[-1])
