"""추론과 역추적 결과의 구조.

진단 에이전트는 이 구조를 그대로 받아 판단한다. 자유 서술이 아니라 필드로
주고받아야 판정 근거가 검증 가능해지고, 채점도 가능해진다.

특히 NearestMatch 가 이 과제의 핵심 자료형이다. "이 이미지의 이 자리가
뱅크의 어느 정상 패치와 가까웠는가"를 담으며, 그 정상 패치가 진짜
정상품인지 잘못 섞인 결함인지를 뒤에서 판별하면 원인이 갈린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class PatchRef:
    """패치 한 칸을 가리키는 좌표.

    source_image
        이미지 경로. 뱅크 쪽 패치면 뱅크를 구성한 정상 이미지이고,
        질의 쪽 패치면 지금 검사 중인 이미지다.
    row, col
        격자 좌표. 행 우선 순서를 전제한다.
    patch_index
        평탄화한 인덱스. row * grid_w + col 과 같다.
    """

    source_image: str
    row: int
    col: int
    patch_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NearestMatch:
    """질의 패치 하나와 그에 가장 가까운 뱅크 패치의 짝.

    distance 가 클수록 정상에서 멀다. 이상 점수는 이 거리에서 나온다.
    """

    query: PatchRef
    bank: PatchRef
    distance: float
    bank_row_index: int  # 뱅크 배열에서의 행 번호. 뱅크 구성 이력 조회에 쓴다

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "bank": self.bank.to_dict(),
            "distance": self.distance,
            "bank_row_index": self.bank_row_index,
        }


@dataclass
class InferenceResult:
    """이미지 한 장의 추론 결과.

    score
        이미지 단위 이상 점수. 임계값과 비교하는 값이다.
    max_patch_distance
        가중 보정을 하기 전의 원 거리. 임계값 스윕에서 기준을 통일하려면
        보정 전 값이 필요할 때가 있어 함께 남긴다.
    matches
        이상 점수가 높은 순서로 뽑은 질의 패치와 그 최근접 뱅크 패치.
        matches[0] 이 판정을 좌우한 자리이며, 진단은 여기서 시작한다.
    patch_distances
        격자 모양(grid_h × grid_w)의 거리 맵. 마스크 면적 기준 판정과
        시각화에 쓴다. 리스트의 리스트로 보관해 직렬화가 되게 한다.
    """

    image: str
    score: float
    max_patch_distance: float
    grid_h: int
    grid_w: int
    matches: list[NearestMatch] = field(default_factory=list)
    patch_distances: list[list[float]] = field(default_factory=list)
    bank_version: str | None = None

    @property
    def top_match(self) -> NearestMatch | None:
        """판정을 좌우한 자리. 진단의 출발점이다."""
        return self.matches[0] if self.matches else None

    def verdict(self, threshold: float) -> str:
        """임계값 대비 판정. 임계값은 뱅크가 아니라 운영 설정에서 온다."""
        return "defect" if self.score >= threshold else "pass"

    def score_position(self, threshold: float, near_ratio: float = 0.9) -> str:
        """이상 점수가 임계값 대비 어디에 있는가 — 판별 항목 3번.

        above  임계값 이상. 검출됨
        near   임계값 바로 아래. 임계값 조정만으로 해결될 여지가 있다
        below  임계값에 크게 못 미침. 임계값 문제로 보기 어렵다
        """
        if self.score >= threshold:
            return "above"
        if self.score >= threshold * near_ratio:
            return "near"
        return "below"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": self.image,
            "score": self.score,
            "max_patch_distance": self.max_patch_distance,
            "grid": [self.grid_h, self.grid_w],
            "bank_version": self.bank_version,
            "matches": [m.to_dict() for m in self.matches],
        }
