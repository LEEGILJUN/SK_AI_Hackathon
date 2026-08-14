"""가상 공장 데이터를 읽는 조회 계층 — `lookup/mock.py` 를 대신한다.

`data/build_factory.py` 가 만든 `manifest.csv` · `mes.csv` 와 장영진의 기준
파일들을 읽어 `lookup/base.py` 의 함수 8개를 채운다.

── 무엇이 목과 다른가 ───────────────────────────────────────────────────

목은 라인·품목을 보지 않고 상수를 돌려줬다. 그래서 **없는 조합을 물어도
그럴듯한 값이 나왔고**, 라인↔품목 매핑이 네 군데서 달랐는데도 아무것도
안 터졌다. 여기서는 조인으로 답하므로 없는 것은 없다고 말한다.

── 못 찾으면 지어내지 않는다 ────────────────────────────────────────────

**목 값으로 떨어지지 않는다.** 기준 파일이 아직 안 채워졌으면 `None` 을
돌려주고, 진단은 그 항목을 "근거를 얻지 못했다"로 다룬다. 여기서 상수를
돌려주면 채워지지 않은 것이 채워진 것처럼 보이고, 그게 목의 문제였다.

지금 채워진 것과 아닌 것:

    resolve_bank          manifest      ✓
    find_images           manifest      ✓
    defect_distribution   manifest      ✓
    get_bank_profile      manifest      ✓
    get_threshold         thresholds.yaml     자리표시 (장영진 확정 필요)
    get_criteria          criteria.yaml       규칙 1건뿐 — pcb 는 아직 없음
    get_quality_baseline  quality_baseline.yaml / 이미지에서 산출
    find_similar_issues   issue_history.jsonl  ✓ 24건 (중복 차단 전용)

── 조인으로 답한다. 임베딩하지 않는다 ───────────────────────────────────

"3라인 A-217 로트 이미지 목록"은 정확히 답할 문제다. 벡터 검색을 쓰면
비슷한 로트를 섞어 온다. 그래프로 표현할 값어치가 있는 것은 이슈 이력
하나뿐이고, 그것도 역할은 중복 차단이다.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lookup.base import (
    BankProfile,
    CriteriaRule,
    DefectDistribution,
    ImageRecord,
    IssueEdge,
    PastIssue,
    QualityBaselineRecord,
    ThresholdRecord,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 유사도 가중치. 어느 간선이 겹쳤는지를 함께 남기므로 합이 1.0 이어야
#: 화면의 백분율이 말이 된다. `lookup/mock.py` 와 같은 값을 쓴다 —
#: 목을 빼고 끼웠을 때 중복 차단 판정이 달라지면 안 된다.
MATCH_WEIGHT = {"object_name": 0.45, "defect_type": 0.40, "line": 0.15}


def _as_date(value: Any) -> date | None:
    if value in (None, "") or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    """비어 있는 CSV 칸을 `None` 으로. pandas 는 그것을 NaN 으로 읽는다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


class FactoryLookup:
    """`data/` 의 공장 데이터를 읽는다. 인자 없이 만들 수 있어야 한다."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        *,
        compute_missing_baselines: bool = False,
    ):
        """
        compute_missing_baselines
            `quality_baseline.yaml` 에 없는 품목의 기준을 **뱅크 구간
            이미지에서 직접 산출**한다. 기본은 끈다 — 이미지 수만 장을 읽는
            일이라 조회 한 번에 낄 비용이 아니고, `--no-images` 로 만든
            데이터에는 이미지가 아예 없다. 4090 에서 켠다.
        """
        self.data_dir = Path(data_dir) if data_dir else REPO_ROOT / "data"
        self.compute_missing_baselines = compute_missing_baselines
        #: 어떤 조회가 실제로 불렸는가. 화면이 방식별로 표시한다.
        self.calls: list[dict[str, Any]] = []

    # ── 원본 읽기. 처음 물어볼 때 한 번만 ───────────────────────────────

    @cached_property
    def manifest(self) -> pd.DataFrame:
        path = self.data_dir / "manifest.csv"
        if not path.exists():
            return pd.DataFrame(
                columns=["image_path", "line", "object", "date", "lot_id",
                         "equipment_id", "split", "label", "mask_path", "visa_source"]
            )
        return pd.read_csv(path, dtype=str).fillna("")

    @cached_property
    def mes(self) -> pd.DataFrame:
        path = self.data_dir / "mes.csv"
        if not path.exists():
            return pd.DataFrame(columns=["lot_id", "line", "object", "date"])
        return pd.read_csv(path, dtype=str).fillna("")

    def _yaml(self, name: str) -> dict:
        path = self.data_dir / name
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def _record(self, name: str, **arguments: Any) -> None:
        from lookup.base import RETRIEVAL_KIND

        self.calls.append(
            {"name": name, "kind": RETRIEVAL_KIND.get(name, "join"), "arguments": arguments}
        )

    # ── 뱅크 ────────────────────────────────────────────────────────────

    def _bank_version(self, line: str, object_name: str) -> str:
        """이 품목을 판정하는 뱅크 이름.

        manifest 에 뱅크 버전 열이 없다. 뱅크는 라인·품목마다 하나이고
        초기 구성 구간에서 만들어지므로 그 둘로 이름을 짓는다. 규칙이
        결정론적이어야 같은 데이터에서 같은 이름이 나온다.
        """
        return f"{object_name}-{line.split('_')[-1]}-v1"

    def _bank_rows(self, line: str, object_name: str) -> pd.DataFrame:
        m = self.manifest
        return m[(m["line"] == line) & (m["object"] == object_name) & (m["split"] == "bank")]

    def _profile_from(self, line: str, object_name: str) -> BankProfile | None:
        rows = self._bank_rows(line, object_name)
        if rows.empty:
            return None
        conditions: dict[str, list[str]] = {}
        for column, key in (("date", "date"), ("equipment_id", "equipment")):
            values = sorted({v for v in rows[column].unique() if v})
            if values:
                conditions[key] = values
        built = sorted(rows["date"].unique())
        return BankProfile(
            bank_version=self._bank_version(line, object_name),
            line=line,
            object_name=object_name,
            source_image_count=int(len(rows)),
            # 패치 수는 뱅크 파일에 있는 값이라 manifest 만으로는 모른다.
            # 0 을 넣고 추정으로 표시한다 — 지어낸 수를 넣으면 재현성
            # 검사가 그 값을 근거로 통과해 버린다.
            patch_count=0,
            conditions=conditions,
            built_at=_as_date(built[-1]) if built else None,
            is_estimated=True,
        )

    def resolve_bank(self, line: str, object_name: str) -> BankProfile | None:
        self._record("resolve_bank", line=line, object_name=object_name)
        return self._profile_from(line, object_name)

    def get_bank_profile(self, bank_version: str) -> BankProfile | None:
        self._record("get_bank_profile", bank_version=bank_version)
        m = self.manifest
        for line, object_name in {
            (r["line"], r["object"]) for _, r in m[["line", "object"]].drop_duplicates().iterrows()
        }:
            if self._bank_version(line, object_name) == bank_version:
                return self._profile_from(line, object_name)
        return None

    # ── 판별 3번 · 임계값 ───────────────────────────────────────────────

    def get_threshold(
        self, line: str, object_name: str, bank_version: str
    ) -> ThresholdRecord | None:
        self._record(
            "get_threshold", line=line, object_name=object_name, bank_version=bank_version
        )
        for row in self._yaml("thresholds.yaml").get("thresholds", []) or []:
            if row.get("line") != line or row.get("object") != object_name:
                continue
            # bank_version 이 비어 있으면 그 품목의 현행 값으로 본다.
            wanted = row.get("bank_version")
            if wanted and bank_version and wanted != bank_version:
                continue
            return ThresholdRecord(
                line=line,
                object_name=object_name,
                bank_version=wanted or bank_version,
                value=float(row["value"]),
                effective_from=_as_date(row.get("effective_from")),
                note=str(row.get("note", "")),
            )
        return None

    # ── 판별 2번 · 화질 기준 분포 ───────────────────────────────────────

    def get_quality_baseline(
        self, line: str, object_name: str
    ) -> QualityBaselineRecord | None:
        self._record("get_quality_baseline", line=line, object_name=object_name)
        payload = self._yaml("quality_baseline.yaml")
        rule = payload.get("outlier_rule", {}) or {}
        for row in payload.get("baselines", []) or []:
            if row.get("line") == line and row.get("object") == object_name:
                override = row.get("outlier_rule_override", {}) or {}
                return QualityBaselineRecord(
                    line=line,
                    object_name=object_name,
                    stats=row.get("stats", {}) or {},
                    computed_from=row.get("computed_from", {}) or {},
                    tolerance_sigma=float(
                        override.get("tolerance_sigma", rule.get("tolerance_sigma", 3.0))
                    ),
                    outlier_ratio_threshold=float(
                        rule.get("outlier_ratio_threshold", 0.30)
                    ),
                )
        if self.compute_missing_baselines:
            return self._baseline_from_images(line, object_name, rule)
        return None

    def _baseline_from_images(
        self, line: str, object_name: str, rule: dict
    ) -> QualityBaselineRecord | None:
        """뱅크 구성 구간 이미지에서 직접 산출한다.

        **초기 뱅크 구간에서만 뽑는다.** 열화가 주입된 운영 구간이 섞이면
        기준 자체가 오염되어 설비 문제를 영영 못 잡는다
        (`data/quality_baseline.yaml` 의 `outlier_rule` 이 그렇게 정의한다).
        """
        from inspection.quality import compute_baseline

        rows = self._bank_rows(line, object_name)
        if rows.empty:
            return None
        paths = [self.data_dir / "factory" / p for p in rows["image_path"]]
        present = [p for p in paths if p.exists()]
        if not present:
            return None
        return QualityBaselineRecord(
            line=line,
            object_name=object_name,
            stats=compute_baseline(present),
            computed_from={
                "split": "bank",
                "n_images": len(present),
                "note": "manifest 의 뱅크 구간 이미지에서 산출",
            },
            tolerance_sigma=float(rule.get("tolerance_sigma", 3.0)),
            outlier_ratio_threshold=float(rule.get("outlier_ratio_threshold", 0.30)),
        )

    # ── 판별 7번 · 판정 기준 ────────────────────────────────────────────

    def get_criteria(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        at: date | None = None,
    ) -> CriteriaRule | None:
        self._record(
            "get_criteria", line=line, object_name=object_name,
            defect_type=defect_type, at=at,
        )
        payload = self._yaml("criteria.yaml")
        matched: list[tuple[int, dict]] = []
        for row in payload.get("rules", []) or []:
            matcher = row.get("matcher", {}) or {}
            if matcher.get("line") not in (None, line):
                continue
            if matcher.get("object") not in (None, object_name):
                continue
            wanted_defect = matcher.get("defect_type")
            if wanted_defect not in (None, defect_type):
                continue
            starts = _as_date(row.get("effective_from"))
            ends = _as_date(row.get("effective_to"))
            when = at or date.today()
            if starts and when < starts:
                continue
            if ends and when > ends:
                continue
            # 좁은 규칙이 우선. defaults.match_order 가 그렇게 정한다.
            specificity = sum(
                1 for key in ("line", "object", "zone", "defect_type") if matcher.get(key)
            )
            matched.append((specificity, row))

        if not matched:
            return None
        matched.sort(key=lambda pair: pair[0], reverse=True)
        row = matched[0][1]
        thresholds = row.get("thresholds", {}) or {}
        return CriteriaRule(
            rule_id=str(row.get("id", "")),
            line=line,
            object_name=object_name,
            defect_type=(row.get("matcher", {}) or {}).get("defect_type"),
            defect_area=float(thresholds["defect_area"]),
            review_area=(
                float(thresholds["review_area"])
                if thresholds.get("review_area") is not None
                else None
            ),
            effective_from=_as_date(row.get("effective_from")),
            effective_to=_as_date(row.get("effective_to")),
        )

    # ── 이슈 이력 그래프 ────────────────────────────────────────────────

    def find_similar_issues(
        self,
        line: str,
        object_name: str,
        defect_type: str | None = None,
        limit: int = 5,
    ) -> list[PastIssue]:
        """**중복 차단 전용이다. 진단 근거가 아니다.**

        과거가 비슷하다고 이번 원인을 그것으로 정하면 진단이 유사도 맞히기가
        된다. 어느 간선이 겹쳐서 비슷하다고 봤는지를 `matched_on` 과 `path`
        로 함께 남긴다 — 유사도 숫자만 돌려주면 검증할 수 없다.
        """
        self._record(
            "find_similar_issues", line=line, object_name=object_name, defect_type=defect_type
        )
        path = self.data_dir / "issue_history.jsonl"
        if not path.exists():
            return []

        query = {"line": line, "object_name": object_name, "defect_type": defect_type}
        found: list[PastIssue] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            node = json.loads(raw)
            # 주석 줄. JSONL 에는 주석 문법이 없어서 이렇게 둔다 — 데이터
            # 파일에 규칙을 적어 두지 않으면 다음 사람이 왜 이렇게 생겼는지
            # 모른 채 고친다.
            if "_comment" in node:
                continue
            matched = [
                key for key, _weight in MATCH_WEIGHT.items()
                if query.get(key) and query[key] == node.get(key)
            ]
            if not matched:
                continue
            issue = str(node["issue_id"])
            edges = [
                IssueEdge(issue, "발생_라인", str(node["line"])),
                IssueEdge(issue, "대상_품목", str(node["object_name"])),
            ]
            if node.get("defect_type"):
                edges.append(IssueEdge(issue, "결함_유형", str(node["defect_type"])))
            edges.append(IssueEdge(issue, "진단_원인", str(node["cause"])))
            edges.append(IssueEdge(issue, "조치", str(node["action"])))
            edges.append(
                IssueEdge(issue, "결과", "해결됨" if node.get("resolved") else "미해결")
            )
            found.append(
                PastIssue(
                    issue_id=issue,
                    line=str(node["line"]),
                    object_name=str(node["object_name"]),
                    cause=str(node["cause"]),
                    action=str(node["action"]),
                    resolved=bool(node.get("resolved", False)),
                    similarity=round(sum(MATCH_WEIGHT[k] for k in matched), 4),
                    summary=str(node.get("summary", "")),
                    defect_type=node.get("defect_type"),
                    path=edges,
                    matched_on=matched,
                )
            )
        found.sort(key=lambda i: (-i.similarity, i.issue_id))
        return found[:limit]

    # ── MES 쪽 ──────────────────────────────────────────────────────────

    def _product_id(self, row: pd.Series) -> str:
        """제품명. manifest 에 열이 없어 경로에서 만든다.

        이슈는 이미지가 아니라 제품명으로 오므로 되짚을 수 있어야 한다.
        `data/build_factory.py` 가 파일명을 `img_NNNN.png` 로 짓는다.
        """
        stem = Path(str(row["image_path"])).stem
        return f"{str(row['object']).upper()}-{row['lot_id']}-{stem}"

    def _to_record(self, row: pd.Series) -> ImageRecord:
        label = _text(row.get("label"))
        return ImageRecord(
            product_id=self._product_id(row),
            path=str(row["image_path"]),
            line=str(row["line"]),
            object_name=str(row["object"]),
            lot=_text(row.get("lot_id")),
            captured_at=_as_date(row.get("date")),
            # manifest 의 label 은 **사실**이다. 설비가 그때 뭐라고 판정했는지는
            # 별개이며 여기에는 없다. 지어내지 않고 비워 둔다 — 채우면
            # 미검·과검 집계가 사실이 아닌 값 위에서 돌게 된다.
            verdict=None,
            ground_truth=label,
            equipment=_text(row.get("equipment_id")),
        )

    def find_images(
        self,
        line: str | None = None,
        object_name: str | None = None,
        lot: str | None = None,
        product_id: str | None = None,
        limit: int = 50,
    ) -> list[ImageRecord]:
        self._record(
            "find_images", line=line, object_name=object_name, lot=lot, product_id=product_id
        )
        if not any((line, object_name, lot, product_id)):
            # 조건 없는 호출은 실수일 가능성이 높다. 공장 전체를 돌려주면
            # 그다음 단계가 통째로 막힌다.
            return []

        rows = self.manifest
        if line:
            rows = rows[rows["line"] == line]
        if object_name:
            rows = rows[rows["object"] == object_name]
        if lot:
            rows = rows[rows["lot_id"] == lot]
        if rows.empty:
            return []

        records = [self._to_record(row) for _, row in rows.iterrows()]
        if product_id:
            # 제품 하나가 지목되면 **그 제품이 속한 로트를 함께 가져온다.**
            # 한 장만 보고는 제품 하나의 문제인지 로트 전체의 문제인지
            # 갈리지 않는다.
            hit = next((r for r in records if r.product_id == product_id), None)
            if hit is None:
                return []
            records = [r for r in records if r.lot == hit.lot]
            records.sort(key=lambda r: (r.product_id != product_id, r.product_id))
        return records[:limit]

    def defect_distribution(
        self,
        line: str | None = None,
        object_name: str | None = None,
        defect_type: str | None = None,
        since: date | None = None,
    ) -> DefectDistribution:
        self._record(
            "defect_distribution", line=line, object_name=object_name,
            defect_type=defect_type, since=since,
        )
        rows = self.manifest
        rows = rows[rows["label"] == "defect"]
        if line:
            rows = rows[rows["line"] == line]
        if object_name:
            rows = rows[rows["object"] == object_name]
        if since is not None:
            rows = rows[rows["date"] >= since.isoformat()]
        if rows.empty:
            return DefectDistribution(total=0)
        return DefectDistribution(
            total=int(len(rows)),
            by_lot=dict(Counter(rows["lot_id"])),
            by_line=dict(Counter(rows["line"])),
            by_equipment=dict(Counter(rows["equipment_id"])),
        )
