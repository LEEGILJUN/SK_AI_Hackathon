"""조회 계층 목 구현 — 실제 데이터가 붙기 전까지의 임시 대체물.

가상 공장 데이터(작업 1·4)와 조회 계층 실구현(작업 5)이 데이터 담당 쪽에서
나오기 전에 진단 루프를 닫아 두기 위한 것이다. **시연에 쓰지 않는다.**

값을 파이썬 안에 둔 이유가 있다. data/ 에 비슷하게 생긴 yaml 을 두면
도메인 담당의 정답 파일·기준 파일과 섞여 나중에 어느 것이 진짜인지 헷갈린다.
여기 있는 숫자는 전부 지어낸 것이고, 지어냈다는 사실이 눈에 보여야 한다.

실구현으로 갈아끼울 때 진단 코드는 고치지 않는다. LookupLayer 만 지키면 된다.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import date

from .base import (
    BankProfile,
    CriteriaRule,
    DefectDistribution,
    ImageRecord,
    IssueEdge,
    PastIssue,
    QualityBaselineRecord,
    ThresholdRecord,
)

#: 이 값들은 전부 임시로 지어낸 것이다. 실데이터가 아니다.
IS_MOCK = True


#: 이슈 이력 그래프 — 온톨로지가 실제로 사는 곳.
#:
#: 조회 계층의 나머지는 전부 조인으로 답한다. 여기만 그래프인 이유는 운영
#: 이력이 **개체 사이의 관계 자체가 답**이기 때문이다. "이 증상이 다른 라인에서
#: 어떤 원인으로 규명돼 어떤 조치로 해결됐나"는 이슈→원인→조치→결과를 따라가야
#: 나오고, 표로 만들면 매번 새 조인을 짜야 한다.
#:
#: 데이터 담당의 실제 그래프가 오면 이 상수가 빠진다. 스키마는 같다.
ISSUE_GRAPH: list[dict] = [
    # 라인↔품목은 `data/build_factory.py` 의 VALID_LINES 를 따른다
    # (line_01=pcb1 · line_02=pcb2 · line_03=pcb3 · line_04=pcb4).
    #
    # **ISS-0042 만 예외다.** pcb1 이 예전에 line_02 에서도 검사되던 때의
    # 이력이다. 라인 재배치는 현장에서 흔하고, **바로 이것 때문에 중복 차단이
    # 라인을 봐야 한다** — "line_02 pcb1 뱅크 오염 이력이 있으니 이번 것도 중복"은
    # 틀린 추론이다. 라인마다 뱅크가 따로이기 때문이다.
    {
        "issue_id": "ISS-0042", "line": "line_02", "object_name": "pcb1",
        "defect_type": "scratch", "cause": "bank_contamination",
        "action": "혼입 이미지 2장 제거 후 뱅크 재구성", "resolved": True,
        "summary": "정상 학습셋에 불량이 섞여 같은 유형을 정상으로 끌어당기고 있었다.",
    },
    {
        "issue_id": "ISS-0031", "line": "line_03", "object_name": "pcb3",
        "defect_type": "scratch", "cause": "threshold",
        "action": "임계값 2.35 → 2.10 재조정", "resolved": True,
        "summary": "이상 점수는 충분히 높았으나 임계값 바로 아래에 몰려 있었다.",
    },
    {
        "issue_id": "ISS-0055", "line": "line_04", "object_name": "pcb4",
        "defect_type": "missing", "cause": "normal_overlap",
        "action": "촬영 각도 변경 · 전용 판별 로직 추가", "resolved": True,
        "summary": "최근접 패치가 진짜 정상품이었다. 재구성으로는 해결되지 않는 유형.",
    },
    {
        "issue_id": "ISS-0067", "line": "line_02", "object_name": "pcb2",
        "defect_type": "bent", "cause": "coverage_gap",
        "action": "야간 로트 정상 이미지 40장 보충", "resolved": True,
        "summary": "야간 조명 조건의 정상 패치가 뱅크 구성에 없었다.",
    },
    {
        "issue_id": "ISS-0071", "line": "line_01", "object_name": "pcb1",
        "defect_type": "melt", "cause": "equipment_optics",
        "action": "설비 점검 요청: 조명 열화 확인", "resolved": False,
        "summary": "화질 지표가 기준 분포를 벗어났다. 모델 문제가 아니다.",
    },
]

#: 어느 속성이 겹치면 얼마나 가까운 것으로 보는가.
#: 자리표시 값이며, 실제 가중치는 시나리오로 측정해 정해야 한다.
_MATCH_WEIGHT = {"object_name": 0.45, "defect_type": 0.40, "line": 0.15}


class MockLookup:
    """지어낸 값으로 채운 조회 계층.

    생성자 인자로 개별 값을 덮어쓸 수 있다. 진단 에이전트 테스트에서
    "임계값만 다르면 판정이 어떻게 갈리는가" 같은 것을 재는 데 쓴다.
    """

    is_mock = True

    def __init__(
        self,
        threshold: float = 2.20,
        quality_stats: dict[str, dict[str, float]] | None = None,
        quality_provider=None,
        criteria_defect_area: float = 150.0,
        criteria_review_area: float | None = 90.0,
        bank_conditions: dict[str, list[str]] | None = None,
        similar_issues: list[PastIssue] | None = None,
        bank_version: str = "v3",
        line: str = "line_01",
        object_name: str = "pcb1",
        catalog: list[ImageRecord] | None = None,
        banks: dict[tuple[str, str], str] | None = None,
        bank_profiles: dict[str, BankProfile] | None = None,
        thresholds: dict[tuple[str, str], float] | None = None,
    ):
        """
        catalog
            MES 가 안다고 치는 이미지 목록. 가상 공장(작업 1·4)이 오기 전까지의
            대체물이다. 비워 두면 find_images 가 빈 목록을 돌려주고, 그때는
            "MES 에서 못 찾았다"가 화면에 그대로 뜬다. 지어낸 이미지를
            돌려주는 것보다 낫다.
        banks
            {(라인, 품목): 뱅크 버전}. 품목마다 뱅크가 다르다는 것을 목에서도
            지킨다. 여기 없는 품목은 resolve_bank 가 None 을 돌려주고,
            "그 품목에는 배포된 모델이 없다"가 답이 된다.
        bank_profiles
            {뱅크 버전: 구성 이력}. 주면 **뱅크마다 다른 구성**을 돌려준다.
            안 주면 아래 `bank_conditions` 하나를 모든 뱅크에 돌려주는데,
            그러면 판별 6번의 답이 뱅크와 무관해진다 — 가상 공장에 없는
            로트가 적혀 있어도 알 길이 없었다. 시연은 `DemoFactory.
            bank_profiles()` 로 실제 구성을 넘긴다.
        thresholds
            {(라인, 품목): 임계값}. **품목마다 다르다.** 4090 실측에서 과검
            1% 지점이 candle 1.925 · capsules 2.560 으로 갈렸다. 여기 없는
            품목은 `threshold` 하나를 그대로 쓴다.
        """
        self.threshold = threshold
        #: (라인, 품목) → 화질 기준을 돌려주는 함수. 있으면 이쪽이 먼저다.
        #:
        #: **기준 분포는 원래 품목·라인마다 다르다.** 목이라고 상수 하나를
        #: 돌려주면 그 사실이 코드에서 사라지고, 실제로 그래서 한 번 틀렸다 —
        #: 아래 자리표시가 캡슐 합성 무늬에 맞춰져 있어서 데모 품목을 pcb 로
        #: 바꾸자 멀쩡한 이미지가 전부 화질 이탈로 잡히고 진단이
        #: `equipment_optics` 로 나왔다. 원인 하나가 다른 것을 가린 것이다.
        self.quality_provider = quality_provider
        self.quality_stats = quality_stats or {
            # 합성 이미지의 실제 분포에 대략 맞춘 자리표시 값이다.
            "brightness": {"mean": 137.0, "std": 4.0},
            "contrast": {"mean": 27.5, "std": 2.0},
            "sharpness": {"mean": 900.0, "std": 120.0},
            "noise": {"mean": 2.2, "std": 0.4},
        }
        self.criteria_defect_area = criteria_defect_area
        self.criteria_review_area = criteria_review_area
        self.bank_conditions = bank_conditions or {
            "date": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"],
            "lot": ["LOT-20260601-001", "LOT-20260602-004", "LOT-20260603-009"],
        }
        self.similar_issues = similar_issues or []
        self.bank_version = bank_version
        self.line = line
        self.object_name = object_name
        self.catalog = list(catalog or [])
        self.banks = dict(banks or {(line, object_name): bank_version})
        self.bank_profiles = dict(bank_profiles or {})
        self.thresholds = dict(thresholds or {})

        #: 호출 기록. 진단이 실제로 어떤 조회를 했는지 검증할 때 쓴다.
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))

    # ── 판별 3번 ────────────────────────────────────────────────────────

    def get_threshold(
        self, line: str, object_name: str, bank_version: str
    ) -> ThresholdRecord | None:
        self._record("get_threshold", line=line, object_name=object_name, bank_version=bank_version)
        value = self.thresholds.get((line, object_name), self.threshold)
        return ThresholdRecord(
            line=line,
            object_name=object_name,
            bank_version=bank_version,
            value=value,
            effective_from=date(2026, 6, 1),
            note=("목 구현이 돌려준 임시값. 실데이터 아님."
                  if (line, object_name) not in self.thresholds else
                  "목 구현. 이 품목에 따로 걸린 값이다."),
        )

    # ── 판별 2번 ────────────────────────────────────────────────────────

    def get_quality_baseline(
        self, line: str, object_name: str
    ) -> QualityBaselineRecord | None:
        self._record("get_quality_baseline", line=line, object_name=object_name)
        stats, note = self.quality_stats, "목 구현. 실제 산출 구간 아님."
        if self.quality_provider is not None:
            measured = self.quality_provider(line, object_name)
            if measured:
                stats = measured
                note = "목 구현. 그 품목 뱅크 정상 이미지에서 산출."
        return QualityBaselineRecord(
            line=line,
            object_name=object_name,
            stats=stats,
            computed_from={"note": note},
        )

    # ── 판별 7번 ────────────────────────────────────────────────────────

    def get_criteria(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        at: date | None = None,
    ) -> CriteriaRule | None:
        self._record(
            "get_criteria", line=line, object_name=object_name, defect_type=defect_type, at=at
        )
        return CriteriaRule(
            rule_id="MOCK-CR-001",
            line=line,
            object_name=object_name,
            defect_type=defect_type,
            defect_area=self.criteria_defect_area,
            review_area=self.criteria_review_area,
            # 목이라도 형식은 맞춘다. 비워 두면 파이프라인이 기본값으로
            # 떨어지는데, 그 기본값이 기준 파일과 다르면 목과 실구현의
            # 판정이 갈린다.
            measurement={"aggregate": "largest_blob", "binarize_threshold": 0.5},
            effective_from=date(2026, 6, 1),
            effective_to=None,
        )

    # ── 판별 6번 ────────────────────────────────────────────────────────

    def get_bank_profile(self, bank_version: str) -> BankProfile | None:
        self._record("get_bank_profile", bank_version=bank_version)
        known = self.bank_profiles.get(bank_version)
        if known is not None:
            # 실제 구성이 있으면 그것을 돌려준다. **복사해서 준다** —
            # resolve_bank 가 line·object_name 을 물어본 품목으로 고쳐 쓰는데,
            # 원본을 고치면 공장이 들고 있는 구성이 함께 바뀐다.
            return replace(known)
        return BankProfile(
            bank_version=bank_version,
            line=self.line,
            object_name=self.object_name,
            source_image_count=len(self.bank_conditions.get("lot", [])) * 4,
            patch_count=0,
            conditions=self.bank_conditions,
            built_at=date(2026, 6, 5),
            is_estimated=True,  # 목이므로 추정으로 표시한다
        )

    # ── 그래프 검색 ─────────────────────────────────────────────────────

    def find_similar_issues(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        limit: int = 5,
    ) -> list[PastIssue]:
        self._record(
            "find_similar_issues", line=line, object_name=object_name, defect_type=defect_type
        )
        if self.similar_issues:
            # 생성자로 직접 넣어 준 것이 있으면 그것을 쓴다. 테스트가 특정
            # 시나리오를 만들 때 그래프를 통째로 바꾸지 않아도 되게 한다.
            return self.similar_issues[:limit]
        return self._walk_graph(line, object_name, defect_type)[:limit]

    def _walk_graph(
        self, line: str, object_name: str, defect_type: str | None
    ) -> list[PastIssue]:
        """이슈 이력 그래프를 걸어 유사 사례를 찾는다.

        **점수만 돌려주지 않는다.** 어떤 간선을 밟아 도달했는지를 함께 남긴다.
        "왜 비슷하다고 봤는가"를 사람이 검증할 수 없으면 그래프 검색은
        블랙박스가 되고, 그러면 중복 차단이라는 역할도 못 맡긴다.
        """
        query = {"line": line, "object_name": object_name, "defect_type": defect_type}
        found: list[PastIssue] = []

        for node in ISSUE_GRAPH:
            matched = [k for k, w in _MATCH_WEIGHT.items()
                       if query.get(k) and query[k] == node.get(k)]
            if not matched:
                continue
            score = sum(_MATCH_WEIGHT[k] for k in matched)

            issue = node["issue_id"]
            path = [IssueEdge(issue, "발생_라인", node["line"]),
                    IssueEdge(issue, "대상_품목", node["object_name"])]
            if node.get("defect_type"):
                path.append(IssueEdge(issue, "결함_유형", node["defect_type"]))
            path.append(IssueEdge(issue, "진단_원인", node["cause"]))
            path.append(IssueEdge(node["cause"], "조치", node["action"]))
            path.append(IssueEdge(node["action"], "결과",
                                  "해결" if node["resolved"] else "미해결"))

            found.append(
                PastIssue(
                    issue_id=issue,
                    line=node["line"],
                    object_name=node["object_name"],
                    cause=node["cause"],
                    action=node["action"],
                    resolved=node["resolved"],
                    similarity=round(score, 2),
                    summary=node["summary"],
                    defect_type=node.get("defect_type"),
                    path=path,
                    matched_on=matched,
                )
            )

        found.sort(key=lambda i: -i.similarity)
        return found

    # ── MES 쪽 ──────────────────────────────────────────────────────────

    def resolve_bank(self, line: str, object_name: str) -> BankProfile | None:
        """품목에 걸린 뱅크를 찾는다. 없으면 None — 배포된 모델이 없다는 뜻이다."""
        self._record("resolve_bank", line=line, object_name=object_name)
        version = self.banks.get((line, object_name))
        if version is None:
            return None
        profile = self.get_bank_profile(version)
        if profile is not None:
            # get_bank_profile 은 목이라 생성자 값을 그대로 쓴다. 실제로 물어본
            # 품목으로 맞춰 돌려줘야 호출한 쪽이 헷갈리지 않는다.
            profile.line = line
            profile.object_name = object_name
        return profile

    def find_images(
        self,
        line: str | None = None,
        object_name: str | None = None,
        lot: str | None = None,
        product_id: str | None = None,
        limit: int = 50,
    ) -> list[ImageRecord]:
        """조건에 맞는 이미지를 목록에서 걸러 낸다. 조인 흉내다."""
        self._record("find_images", line=line, object_name=object_name,
                     lot=lot, product_id=product_id)
        if not any((line, object_name, lot, product_id)):
            # 조건 없이 전체를 훑는 것은 실수일 가능성이 높다.
            return []
        found = [
            r for r in self.catalog
            if (line is None or r.line == line)
            and (object_name is None or r.object_name == object_name)
            and (lot is None or r.lot == lot)
            and (product_id is None or r.product_id == product_id)
        ]
        return found[:limit]

    def defect_distribution(
        self,
        line: str | None = None,
        object_name: str | None = None,
        defect_type: str | None = None,
        since: date | None = None,
    ) -> DefectDistribution:
        """결함으로 확인된 건이 어느 라인·로트에 몰렸는지 센다."""
        self._record("defect_distribution", line=line, object_name=object_name,
                     defect_type=defect_type, since=since)
        rows = [
            r for r in self.catalog
            if r.ground_truth == "defect"
            and (line is None or r.line == line)
            and (object_name is None or r.object_name == object_name)
            and (since is None or (r.captured_at is not None and r.captured_at >= since))
        ]
        return DefectDistribution(
            total=len(rows),
            by_lot=dict(Counter(r.lot for r in rows if r.lot)),
            by_line=dict(Counter(r.line for r in rows)),
            by_equipment=dict(Counter(r.equipment for r in rows if r.equipment)),
        )


def resolved_duplicate(
    cause: str = "bank_contamination",
    similarity: float = 0.92,
    line: str = "line_01",
) -> PastIssue:
    """중복 차단 시나리오용 과거 사례 하나를 만든다."""
    return PastIssue(
        issue_id="MOCK-ISS-0042",
        line=line,
        object_name="pcb1",
        cause=cause,
        action="혼입 이미지 제거 후 뱅크 재구성",
        resolved=True,
        similarity=similarity,
        summary="타 라인에서 동일 증상이 접수되어 뱅크 오염으로 규명, 조치 완료됨.",
    )
