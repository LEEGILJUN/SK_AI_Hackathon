"""조회 계층 목 구현 — 실제 데이터가 붙기 전까지의 임시 대체물.

가상 공장 데이터(작업 1·4)와 조회 계층 실구현(작업 5)이 이동현 쪽에서
나오기 전에 진단 루프를 닫아 두기 위한 것이다. **시연에 쓰지 않는다.**

값을 파이썬 안에 둔 이유가 있다. data/ 에 비슷하게 생긴 yaml 을 두면
장영진의 정답 파일·기준 파일과 섞여 나중에 어느 것이 진짜인지 헷갈린다.
여기 있는 숫자는 전부 지어낸 것이고, 지어냈다는 사실이 눈에 보여야 한다.

실구현으로 갈아끼울 때 진단 코드는 고치지 않는다. LookupLayer 만 지키면 된다.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

from .base import (
    BankProfile,
    CriteriaRule,
    DefectDistribution,
    ImageRecord,
    PastIssue,
    QualityBaselineRecord,
    ThresholdRecord,
)

#: 이 값들은 전부 임시로 지어낸 것이다. 실데이터가 아니다.
IS_MOCK = True


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
        criteria_defect_area: float = 150.0,
        criteria_review_area: float | None = 90.0,
        bank_conditions: dict[str, list[str]] | None = None,
        similar_issues: list[PastIssue] | None = None,
        bank_version: str = "v3",
        line: str = "line_02",
        object_name: str = "capsules",
        catalog: list[ImageRecord] | None = None,
        banks: dict[tuple[str, str], str] | None = None,
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
        """
        self.threshold = threshold
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

        #: 호출 기록. 진단이 실제로 어떤 조회를 했는지 검증할 때 쓴다.
        self.calls: list[tuple[str, dict]] = []

    def _record(self, name: str, **kwargs) -> None:
        self.calls.append((name, kwargs))

    # ── 판별 3번 ────────────────────────────────────────────────────────

    def get_threshold(
        self, line: str, object_name: str, bank_version: str
    ) -> ThresholdRecord | None:
        self._record("get_threshold", line=line, object_name=object_name, bank_version=bank_version)
        return ThresholdRecord(
            line=line,
            object_name=object_name,
            bank_version=bank_version,
            value=self.threshold,
            effective_from=date(2026, 6, 1),
            note="목 구현이 돌려준 임시값. 실데이터 아님.",
        )

    # ── 판별 2번 ────────────────────────────────────────────────────────

    def get_quality_baseline(
        self, line: str, object_name: str
    ) -> QualityBaselineRecord | None:
        self._record("get_quality_baseline", line=line, object_name=object_name)
        return QualityBaselineRecord(
            line=line,
            object_name=object_name,
            stats=self.quality_stats,
            computed_from={"note": "목 구현. 실제 산출 구간 아님."},
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
            effective_from=date(2026, 6, 1),
            effective_to=None,
        )

    # ── 판별 6번 ────────────────────────────────────────────────────────

    def get_bank_profile(self, bank_version: str) -> BankProfile | None:
        self._record("get_bank_profile", bank_version=bank_version)
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
        return self.similar_issues[:limit]

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
        object_name="capsules",
        cause=cause,
        action="오염 샘플 제거 후 뱅크 재구성",
        resolved=True,
        similarity=similarity,
        summary="타 라인에서 동일 증상이 접수되어 뱅크 오염으로 규명, 조치 완료됨.",
    )
