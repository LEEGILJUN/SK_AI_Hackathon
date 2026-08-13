"""섀도 비교 — 신·구 뱅크를 같은 이미지에 병렬로 돌린다 (작업 18).

양산 데이터에는 정답 라벨이 없다. 그리고 신규 뱅크를 배포해 버리면 기존
뱅크는 더 이상 동작하지 않아 동일 조건 비교가 불가능하다. 그래서 신규 뱅크를
**실제 판정에 쓰지 않고** 같은 이미지에 병렬로만 추론시켜, 판정이 갈리는
케이스만 뽑는다.

사람은 불일치 건만 확인하면 되므로 검증 공수가 줄고, 그 확인 결과가 곧
정답 라벨이 되어 다음 평가의 재료가 된다.

홀드아웃 평가와 혼동하지 말 것.
  홀드아웃  별도 데이터로 성능을 잰다. 정답이 있어야 한다
  섀도      같은 데이터에 두 모델을 돌려 차이를 본다. 정답이 없어도 된다
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

from .bank import MemoryBank
from .features import PatchEmbedder
from .trace import score_image
from .types import InferenceResult


@dataclass
class Disagreement:
    """두 뱅크의 판정이 갈린 이미지 한 건.

    kind
        newly_detected  구 뱅크는 놓쳤는데 신 뱅크가 잡았다
        newly_missed    구 뱅크는 잡았는데 신 뱅크가 놓쳤다

    newly_missed 가 중요하다. 고치려던 문제는 해결됐는데 다른 것을 잃는
    경우이며, 이걸 못 보면 개선인 줄 알고 배포한다.
    """

    image: str
    kind: str
    current_score: float
    candidate_score: float
    current_verdict: str
    candidate_verdict: str
    candidate_nearest_image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowCase:
    """섀도로 돌린 이미지 한 장의 전·후 판정.

    갈린 것만이 아니라 **전부** 남긴다. 사람이 확인할 것은 갈린 건뿐이지만,
    "몇 장을 어떻게 통과시켰는가"를 보여주려면 통과한 것도 있어야 한다.
    라인 시뮬레이터가 이것을 한 장씩 흘려보낸다.
    """

    image: str
    current_score: float
    candidate_score: float
    current_verdict: str
    candidate_verdict: str
    agreed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShadowReport:
    """섀도 비교 결과.

    agreement_rate 가 높다고 좋은 것이 아니다. 아무것도 안 바뀌었다는 뜻일
    수도 있다. 무엇이 어떻게 갈렸는지를 봐야 한다.
    """

    total: int
    agreed: int
    disagreements: list[Disagreement] = field(default_factory=list)
    #: 갈린 것 포함 전부. 순서는 입력 순서와 같다.
    cases: list[ShadowCase] = field(default_factory=list)
    current_version: str = ""
    candidate_version: str = ""
    current_threshold: float = 0.0
    candidate_threshold: float = 0.0

    @property
    def agreement_rate(self) -> float:
        return self.agreed / self.total if self.total else 0.0

    @property
    def newly_detected(self) -> list[Disagreement]:
        return [d for d in self.disagreements if d.kind == "newly_detected"]

    @property
    def newly_missed(self) -> list[Disagreement]:
        return [d for d in self.disagreements if d.kind == "newly_missed"]

    @property
    def review_count(self) -> int:
        """사람이 확인해야 할 건수. 전수 확인 대비 이만큼으로 줄어든다."""
        return len(self.disagreements)

    def summary(self) -> str:
        return (
            f"{self.total}장 중 {self.review_count}장에서 판정이 갈렸다 "
            f"(새로 검출 {len(self.newly_detected)}, 새로 놓침 {len(self.newly_missed)}). "
            f"사람은 이 {self.review_count}장만 확인하면 된다."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "agreed": self.agreed,
            "agreement_rate": self.agreement_rate,
            "review_count": self.review_count,
            "newly_detected": len(self.newly_detected),
            "newly_missed": len(self.newly_missed),
            "current_version": self.current_version,
            "candidate_version": self.candidate_version,
            "summary": self.summary(),
            "disagreements": [d.to_dict() for d in self.disagreements],
            "cases": [c.to_dict() for c in self.cases],
        }


def shadow_compare(
    image_paths: Sequence[str | Path],
    current_bank: MemoryBank,
    candidate_bank: MemoryBank,
    current_threshold: float,
    candidate_threshold: float | None = None,
    embedder: PatchEmbedder | None = None,
    root: str | Path | None = None,
) -> ShadowReport:
    """같은 이미지에 두 뱅크를 돌려 판정이 갈리는 건만 뽑는다.

    candidate_threshold
        신규 뱅크의 임계값. 뱅크가 바뀌면 점수 척도도 바뀌므로 같은 임계값을
        그대로 쓰면 안 되는 경우가 많다. 주지 않으면 현행 임계값을 쓰되,
        그때는 척도 차이가 결과에 섞인다는 점을 감안해야 한다.
    """
    if not image_paths:
        raise ValueError("섀도 비교할 이미지가 없다.")

    embedder = embedder or PatchEmbedder()
    candidate_threshold = current_threshold if candidate_threshold is None else candidate_threshold

    agreed = 0
    disagreements: list[Disagreement] = []
    cases: list[ShadowCase] = []

    for path in image_paths:
        current: InferenceResult = score_image(path, current_bank, embedder, root=root, top_k=1)
        candidate: InferenceResult = score_image(path, candidate_bank, embedder, root=root, top_k=1)

        current_verdict = current.verdict(current_threshold)
        candidate_verdict = candidate.verdict(candidate_threshold)
        same = current_verdict == candidate_verdict

        cases.append(
            ShadowCase(
                image=current.image,
                current_score=current.score,
                candidate_score=candidate.score,
                current_verdict=current_verdict,
                candidate_verdict=candidate_verdict,
                agreed=same,
            )
        )

        if same:
            agreed += 1
            continue

        kind = (
            "newly_detected"
            if current_verdict == "pass" and candidate_verdict == "defect"
            else "newly_missed"
        )
        top = candidate.top_match
        disagreements.append(
            Disagreement(
                image=current.image,
                kind=kind,
                current_score=current.score,
                candidate_score=candidate.score,
                current_verdict=current_verdict,
                candidate_verdict=candidate_verdict,
                candidate_nearest_image=top.bank.source_image if top else None,
            )
        )

    return ShadowReport(
        total=len(image_paths),
        agreed=agreed,
        disagreements=disagreements,
        cases=cases,
        current_version=current_bank.version,
        candidate_version=candidate_bank.version,
        current_threshold=current_threshold,
        candidate_threshold=candidate_threshold,
    )
