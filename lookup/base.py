"""결정론적 조회 계층 — 인터페이스 (작업 5).

진단의 신뢰도는 벡터 검색이 아니라 여기서 나온다. 뱅크 구성 프로파일, 화질
기준 분포, 임계값, 판정 기준은 조인과 집계로 정확한 값을 얻어야 하는 대상이며
임베딩하면 오히려 정확도가 떨어진다.

**이 파일은 명세다.** 진단 에이전트는 LookupLayer 만 보고 동작하고, 실제
구현이 무엇인지 알지 못한다. 지금은 목 구현(mock.py)으로 루프를 닫아 두고,
이동현이 가상 공장 데이터에 붙은 구현으로 갈아끼운다. 그때 진단 코드는
고치지 않는다.

각 함수가 어느 판별 항목에 대응하는지 적어 둔다. 이 대응이 흐려지면
"이 조회가 왜 필요한가"를 나중에 아무도 답하지 못한다.

    판별 2번  화질 기준 분포     get_quality_baseline
    판별 3번  임계값             get_threshold
    판별 6번  뱅크 구성 이력     get_bank_profile
    판별 7번  판정 기준          get_criteria

판별 1·5번은 시각 언어 모델(agents/vision.py), 4번은 뱅크 역추적
(inspection/trace.py)이 맡는다. 여기서 다루지 않는다.

── MES 쪽 세 개는 판별 항목이 아니라 그 앞 단계다 ──────────────────────

    resolve_bank         이 품목은 어느 뱅크로 판정하는가
    find_images          이 제품·로트의 이미지가 무엇인가
    defect_distribution  결함이 특정 라인·로트에 몰렸는가

접수된 이슈는 보통 이미지가 아니라 **제품명이나 로트로** 온다("A-217 로트
캡슐이 계속 빠집니다"). 그것을 이미지로 바꾸고, 그 품목의 뱅크를 찾아 추론해야
비로소 판별 항목을 잴 수 있다. 그 앞단이 여기다.

**이 셋은 벡터 검색이 아니다.** "3라인 A-217 로트 캡슐 이미지 목록"은 조인으로
정확히 답할 문제이고, 임베딩하면 비슷한 로트를 섞어 온다. 언어 모델이 하는 일은
"MES 를 조회해야겠다"고 판단해 도구를 부르는 데까지고, 조회 자체는 결정론적이다.
그래프 검색은 find_similar_issues 하나뿐이며 역할은 중복 차단이다.

뱅크는 **품목마다 따로 있다.** 캡슐의 정상 패치로 PCB 를 판정할 수 없다.
resolve_bank 가 없으면 뱅크가 하나뿐인 전제가 코드 곳곳에 박힌다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Protocol, runtime_checkable


#: 각 조회가 **실제로 어떤 방식으로** 답을 찾는가.
#:
#: 화면에 이것을 그대로 띄운다. "다 RAG 로 찾습니다"가 보기에는 좋지만
#: 사실이 아니고, 사실이 아닌 것을 띄우면 심사에서 한 번만 파고들어도 무너진다.
#: 그리고 **이 구분 자체가 이 과제의 논거다** — 진단의 신뢰도는 벡터 검색이
#: 아니라 결정론적 조회에서 나온다. 여덟 중 일곱이 조인이고 그래프 검색은
#: 하나뿐이며, 그 하나의 역할도 진단 근거가 아니라 중복 차단이다.
#:
#: 새 조회 함수를 추가하면 여기에도 등록한다. 빠지면 화면이 "미분류"로 뜬다.
def bank_item_key(line: str, object_name: str) -> str:
    """뱅크를 라인·품목 단위로 묶는 열쇠. **저장소 폴더 이름이기도 하다.**

    `bank_version_for` 와 같은 규칙을 쓴다. 버전만 떼어낸 것이며, 뱅크
    저장소(`inspection/store.py`)가 이 값으로 폴더를 나눈다.

    저장소가 이 규칙을 따로 갖지 않는다. 두 벌이 되면 저장한 자리와 찾는
    자리가 갈린다.
    """
    return f"{object_name}-{line.split('_')[-1]}"


def bank_version_for(line: str, object_name: str) -> str:
    """이 라인·품목을 판정하는 뱅크의 이름.

    **이름 규칙이 두 벌이면 안 된다.** 뱅크를 만드는 쪽(`app/pipeline.py` 의
    `DemoFactory`)과 조회하는 쪽(`lookup/factory.py`)이 다른 이름을 쓰면
    `get_bank_profile(version)` 이 조용히 `None` 을 돌려주고, 판별 6번
    커버리지가 통째로 비게 된다. 실제로 `pcb1-v3` 대 `pcb1-01-v1` 로
    갈려 있었다.

    라인을 이름에 넣는 이유는 **뱅크가 라인마다 따로**이기 때문이다. 지금은
    라인↔품목이 1:1 이라 품목만으로도 유일하지만, 같은 품목이 두 라인에서
    돌면 이름이 겹쳐 서로의 뱅크를 물게 된다.
    """
    return f"{bank_item_key(line, object_name)}-v1"


RETRIEVAL_KIND: dict[str, str] = {
    "get_threshold": "join",
    "get_quality_baseline": "join",
    "get_criteria": "join",
    "get_bank_profile": "join",
    "resolve_bank": "join",
    "find_images": "join",
    "defect_distribution": "aggregate",
    "find_similar_issues": "graph",
}

#: 사람이 읽을 이름과 한 줄 설명.
RETRIEVAL_LABEL: dict[str, tuple[str, str]] = {
    "join": ("결정론적 조회", "조인으로 정확한 값을 얻는다. 임베딩하면 오히려 틀린다"),
    "aggregate": ("집계", "세어서 답한다. 어디에 몰렸는지는 계산 문제다"),
    "graph": ("그래프 검색", "유사 사례 탐색. 중복 차단 전용이며 진단 근거가 아니다"),
    "llm": ("언어 모델", "자연어에서 항목을 뽑는다. 판정하지 않는다"),
    "vlm": ("시각 언어 모델", "이미지를 읽는다. 판별 1·5번뿐"),
    "compute": ("계산", "추론·역추적·통계"),
}


# ── 조회 결과 ───────────────────────────────────────────────────────────


@dataclass
class ThresholdRecord:
    """판별 3번 — 현재 운영 중인 이상 점수 임계값.

    임계값은 뱅크에 들어 있지 않다. 운영 설정이며 뱅크와 별개로 바뀐다.
    그래서 어느 뱅크 버전에 대해 언제부터 쓰인 값인지를 함께 남긴다.
    """

    line: str
    object_name: str
    bank_version: str
    value: float
    effective_from: date | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "effective_from": self.effective_from.isoformat() if self.effective_from else None
        }


@dataclass
class QualityBaselineRecord:
    """판별 2번 — 라인·객체별 화질 기준 분포.

    stats 는 지표별 {"mean", "std", ...}. inspection.quality.assess_quality 에
    그대로 넘길 수 있는 형태여야 한다.
    """

    line: str
    object_name: str
    stats: dict[str, dict[str, float]]
    computed_from: dict[str, Any] = field(default_factory=dict)
    tolerance_sigma: float = 3.0
    outlier_ratio_threshold: float = 0.30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriteriaRule:
    """판별 7번 — 마스크 면적 기반 판정 기준.

    기준은 덮어쓰지 않고 쌓인다. 과거 이슈를 다시 볼 때 그 시점의 기준으로
    판정해야 하므로 유효 기간이 붙는다.
    """

    rule_id: str
    line: str
    object_name: str
    defect_type: str | None
    defect_area: float
    review_area: float | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    #: 면적을 **어떻게 재는가**. `criteria.yaml` 의 `measurement` 절.
    #:
    #: 같은 마스크라도 세는 방법으로 판정이 갈린다. 흩어진 잡음을 다 더하면
    #: 멀쩡한 이미지가 불량으로 나가고, 그것을 "기준 문제"로 오진한다.
    #:
    #:     aggregate            largest_blob | total_area
    #:     binarize_threshold   이상맵을 마스크로 만들 때의 컷오프.
    #:                          운영 임계값으로 정규화한 값 기준이다
    measurement: dict[str, Any] = field(default_factory=dict)

    def verdict_for(self, area: float) -> str:
        """면적을 판정으로 옮긴다. defect | review | pass."""
        if area >= self.defect_area:
            return "defect"
        if self.review_area is not None and area >= self.review_area:
            return "review"
        return "pass"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "effective_from": self.effective_from.isoformat() if self.effective_from else None,
            "effective_to": self.effective_to.isoformat() if self.effective_to else None,
        }


@dataclass
class BankProfile:
    """판별 6번 — 뱅크가 무엇으로 만들어졌는가.

    커버리지 부족을 가리는 근거다. 지금 문제가 난 조건(로트, 일자, 설비)의
    정상 이미지가 뱅크 구성에 들어갔는지를 여기서 확인한다.

    conditions
        뱅크를 구성한 이미지들이 가진 조건의 집합. 예를 들어
        {"date": ["2026-06-01", ...], "lot": [...], "equipment": [...]}
        진단은 "지금 조건이 이 집합에 있는가"만 물으면 된다.
    """

    bank_version: str
    line: str
    object_name: str
    source_image_count: int
    patch_count: int
    conditions: dict[str, list[str]] = field(default_factory=dict)
    built_at: date | None = None
    is_estimated: bool = False  # 폴더 스캔으로 역추정한 이력인가

    def covers(self, key: str, value: str) -> bool | None:
        """그 조건이 뱅크 구성에 포함됐는가.

        셋을 구분한다. **"기록하지 않은 축"과 "값이 없는 축"은 다르다.**

            True   그 축을 기록했고 값이 있다
            False  그 축을 기록했는데 값이 없다 → 커버리지 부족
            None   그 축을 아예 기록하지 않았다 → 모른다

        전에는 `None` 자리도 `False` 를 돌려줬다. 그러면 **뱅크 프로파일이
        설비를 안 담고 있다는 이유만으로 "설비 조건이 뱅크에 없다"가 되어**
        커버리지 부족으로 오진한다. 모르는 것을 없다고 답하면 안 된다.
        """
        values = self.conditions.get(key)
        if values is None:
            return None
        return value in values

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "built_at": self.built_at.isoformat() if self.built_at else None
        }


@dataclass
class ImageRecord:
    """MES 가 아는 이미지 한 장.

    이슈는 이미지가 아니라 제품명이나 로트로 온다. 그것을 실제 파일로 바꾸는
    것이 이 자료형이며, **판정에 필요한 맥락을 함께 들고 온다.** 어느 라인
    어느 로트에서 언제 찍혔는지가 있어야 커버리지 부족도 로트 집중도도 잴 수
    있다.

    path
        저장소 기준 상대 경로. 절대 경로를 넣으면 장비가 바뀔 때 깨진다.
    verdict
        그때 검사 설비가 낸 판정. 사람이 나중에 확인한 값이 있으면
        ground_truth 에 들어가고, 둘이 다르면 미검 또는 과검이다.
    """

    product_id: str
    path: str
    line: str
    object_name: str
    lot: str | None = None
    captured_at: date | None = None
    verdict: str | None = None          # 설비 판정 defect | pass
    ground_truth: str | None = None     # 사람 확인 defect | pass. 없으면 미확인
    #: 이 이미지가 어느 구간의 것인가. bank | operation | holdout | pending.
    #:
    #: **`pending` 은 아직 검사하지 않은 생산분이다.** 예약 스케줄러가 지정
    #: 시각에 이것만 집어 돌리고, 섀도 평가가 신·구 뱅크를 나란히 적용한다.
    #: 모르면 None 이다 — 구간 개념이 없는 조회 구현도 있을 수 있다.
    split: str | None = None
    equipment: str | None = None

    @property
    def is_missed(self) -> bool:
        """미검 — 실제 불량인데 설비가 양품이라 했다."""
        return self.ground_truth == "defect" and self.verdict == "pass"

    @property
    def is_overkill(self) -> bool:
        """과검 — 실제 양품인데 설비가 불량이라 했다."""
        return self.ground_truth == "pass" and self.verdict == "defect"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
            "is_missed": self.is_missed,
            "is_overkill": self.is_overkill,
        }


@dataclass
class DefectDistribution:
    """결함이 어디에 몰렸는가 — 승인 문서에 싣는 집계.

    "이 결함이 특정 라인이나 로트에 몰려 있는가"는 조치를 가르는 질문이다.
    한 로트에만 몰려 있으면 그 로트의 자재나 설비를 의심해야 하고, 라인 전반에
    고르게 퍼져 있으면 모델 쪽 문제로 본다. **뱅크를 다시 만들기 전에 봐야 할
    값**이며, 몰려 있는데 재구성부터 하면 원인을 놔둔 채 증상만 덮는다.

    by_lot / by_line
        키별 {값: 결함 건수}.
    total
        집계에 들어간 전체 건수. 비중을 내려면 필요하다.
    """

    total: int
    by_lot: dict[str, int] = field(default_factory=dict)
    by_line: dict[str, int] = field(default_factory=dict)
    by_equipment: dict[str, int] = field(default_factory=dict)

    #: 한 키가 이 비중을 넘게 차지하면 "몰려 있다"고 본다. 자리표시 값이며
    #: 시나리오로 측정해 정해야 한다.
    CONCENTRATION_THRESHOLD = 0.60

    def _top(self, counts: dict[str, int]) -> tuple[str, float] | None:
        if not counts or not self.total:
            return None
        key = max(counts, key=lambda k: counts[k])
        return key, counts[key] / self.total

    def concentrated_in(self) -> dict[str, tuple[str, float]]:
        """기준을 넘게 몰린 축만 돌려준다. 비어 있으면 고르게 퍼진 것이다."""
        found = {}
        for name, counts in (("lot", self.by_lot), ("line", self.by_line),
                             ("equipment", self.by_equipment)):
            top = self._top(counts)
            if top and top[1] >= self.CONCENTRATION_THRESHOLD:
                found[name] = top
        return found

    def describe(self) -> str:
        if not self.total:
            return "집계할 결함 건수가 없다."
        hits = self.concentrated_in()
        if not hits:
            return f"결함 {self.total}건이 특정 라인·로트에 몰려 있지 않다. 고르게 퍼져 있다."
        parts = [f"{name} {key} 에 {share:.0%}" for name, (key, share) in hits.items()]
        return (
            f"결함 {self.total}건 중 {', '.join(parts)} 가 몰려 있다. "
            f"뱅크를 다시 만들기 전에 그쪽 원인을 먼저 확인해야 한다."
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "concentrated_in": {k: {"value": v[0], "share": v[1]}
                                for k, v in self.concentrated_in().items()},
            "note": self.describe(),
        }


@dataclass
class IssueEdge:
    """이슈 이력 그래프의 간선 하나.

    이 프로젝트에서 온톨로지가 실제로 사는 곳이다. MES 도 이미지 메타데이터도
    조인으로 답하고, **그래프로 표현할 값어치가 있는 것은 이것 하나**다 —
    운영 이력은 개체 사이의 관계 자체가 답이기 때문이다.

        이슈 ─[발생_라인]→ 라인
        이슈 ─[대상_품목]→ 품목
        이슈 ─[결함_유형]→ 결함유형
        이슈 ─[진단_원인]→ 원인
        원인 ─[조치]→ 조치
        조치 ─[결과]→ 해결/미해결

    유사 사례를 "왜 비슷하다고 봤는가"가 이 간선들에 남는다. 점수 하나만
    돌려주면 사람이 검증할 수 없다.
    """

    source: str
    relation: str
    target: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PastIssue:
    """그래프 검색 결과 — 유사 사례 한 건.

    역할은 중복 작업 차단 하나다. 진단 근거로 쓰지 않는다. 진단의 근거는
    결정론적 조회에서 나오고, 이것은 "이미 해결된 건 아닌가"를 묻는 용도다.

    **이 좁은 역할이 설계다.** 과거 사례가 비슷하다고 이번 원인을 그것으로
    정하면 진단이 유사도 맞히기가 된다. 그래프는 "이미 답이 나온 일인가"만
    묻고, 원인은 매번 판별 7항목으로 새로 규명한다.

    path
        이 사례에 도달한 그래프 경로. 어떤 간선을 밟았는지가 남아 있어야
        "왜 비슷하다고 봤는가"를 사람이 검증할 수 있다.
    """

    issue_id: str
    line: str
    object_name: str
    cause: str
    action: str
    resolved: bool
    similarity: float
    summary: str = ""
    defect_type: str | None = None
    path: list[IssueEdge] = field(default_factory=list)
    matched_on: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── 인터페이스 ──────────────────────────────────────────────────────────


@runtime_checkable
class LookupLayer(Protocol):
    """진단 에이전트가 부르는 조회 함수 전체.

    구현이 값을 찾지 못하면 예외를 내지 말고 None 을 돌려준다. 진단은
    "근거를 얻지 못했다"를 하나의 상태로 다뤄야 하며, 조회 실패로 멈추면
    안 된다. 근거가 비어 있는 것과 진단이 죽는 것은 다르다.
    """

    def get_threshold(
        self, line: str, object_name: str, bank_version: str
    ) -> ThresholdRecord | None:
        """판별 3번 — 현재 임계값."""
        ...

    def get_quality_baseline(
        self, line: str, object_name: str
    ) -> QualityBaselineRecord | None:
        """판별 2번 — 화질 기준 분포."""
        ...

    def get_criteria(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        at: date | None = None,
    ) -> CriteriaRule | None:
        """판별 7번 — 그 시점에 유효한 판정 기준.

        at 을 주면 그 날짜에 유효했던 기준을 돌려준다. 과거 이슈를 다시 볼 때
        지금 기준으로 판정하면 "기준 문제"를 영영 찾지 못한다.
        """
        ...

    def get_bank_profile(self, bank_version: str) -> BankProfile | None:
        """판별 6번 — 뱅크 구성 이력."""
        ...

    def find_similar_issues(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        limit: int = 5,
    ) -> list[PastIssue]:
        """유사 사례. 중복 차단 전용이며 진단 근거가 아니다."""
        ...

    # ── MES 쪽. 판별 항목 앞 단계 ───────────────────────────────────────

    def resolve_bank(self, line: str, object_name: str) -> BankProfile | None:
        """이 품목을 판정하는 뱅크는 무엇인가.

        뱅크는 품목마다 따로 있다. 캡슐의 정상 패치로 PCB 를 판정할 수 없다.
        운영 중인 뱅크가 없으면 None 이고, 그때는 "아직 배포된 모델이 없다"가
        답이며 진단으로 넘어가지 않는다.
        """
        ...

    def find_images(
        self,
        line: str | None = None,
        object_name: str | None = None,
        lot: str | None = None,
        product_id: str | None = None,
        limit: int = 50,
    ) -> list[ImageRecord]:
        """제품명·로트·라인으로 이미지를 찾는다.

        **조인으로 답한다. 임베딩하지 않는다.** 조건을 하나도 주지 않으면 빈
        목록을 돌려준다 — 전체를 훑어 오는 것은 실수일 가능성이 높다.

        찾지 못하면 빈 목록이다. 예외를 내지 않는다.
        """
        ...

    def defect_distribution(
        self,
        line: str | None = None,
        object_name: str | None = None,
        defect_type: str | None = None,
        since: date | None = None,
    ) -> DefectDistribution:
        """결함이 어느 라인·로트에 몰렸는가.

        승인 문서에 싣는 값이다. 한 로트에 몰려 있으면 자재나 설비를 먼저
        의심해야 하고, 그때 뱅크부터 다시 만들면 증상만 덮는다.

        집계할 것이 없으면 total=0 인 빈 집계를 돌려준다. None 이 아니다 —
        "몰린 곳이 없다"와 "못 셌다"를 호출하는 쪽이 구분할 수 있어야 한다.
        """
        ...
