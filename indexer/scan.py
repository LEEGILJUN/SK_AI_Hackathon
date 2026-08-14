"""폴더 스캔으로 뱅크 구성 이력을 복원한다 — 작업 10.

── 무엇을 푸는가 ───────────────────────────────────────────────────────

배포된 뱅크가 **무엇으로 만들어졌는지 아무도 안 적어 두는 것**이 현장의
기본값이다. 그런데 판별 6번("현재 조건의 정상 패치가 뱅크에 있는가")은
구성 이력이 있어야 답할 수 있고, 그것 없이는 **커버리지 부족과 정상 분포
중첩이 갈리지 않는다.** 둘은 조치가 정반대다.

`lookup/factory.py` 와 다르다. 그쪽은 `manifest.csv` 를 읽는다 — 누군가
대장을 써 뒀다는 전제다. 여기는 **대장이 없을 때** 파일만 보고 되짚는다.

── 사내 폴더 구조를 전제하지 않는다 ────────────────────────────────────

특정 폴더 이름이나 깊이를 가정하지 않는다. 루트 아래를 전부 걸어
`bank_meta.json`(`inspection/bank.py` 가 쓰는 이름)을 찾을 뿐이다. 폴더가
어떻게 생겼든, 몇 단계 깊이든 상관없다.

── 복원한 것은 추정으로 표시한다 ───────────────────────────────────────

세 등급으로 가른다. **확정으로 올리는 것은 사람이 한다.**

    확정(recorded)   bank_meta.json 이 있다. 구성 이력 그 자체이며 추정이 아니다
    추정(inferred)   벡터 파일만 있다. 패치 수는 알지만 어느 이미지에서 왔는지는
                     모른다. 이웃 폴더의 이미지를 후보로 제시하되 **후보일 뿐**이다
    미상(unknown)    벡터도 이력도 없다. 뱅크라고 부르지 않는다

근거를 `evidence` 에 남긴다. "왜 이 이미지가 이 뱅크에 들어갔다고 보는가"에
답할 수 없는 복원은 담당자가 확인할 방법이 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

from lookup.base import BankProfile

#: `inspection/bank.py` 의 `MemoryBank.save` 가 쓰는 이름. 두 벌이 되면
#: 인덱서가 뱅크를 못 찾으므로 여기서 import 하지 않고 값을 맞춰 둔 뒤
#: `tests/test_indexer.py` 가 어긋나면 잡는다(순환 import 를 피한다).
BANK_META = "bank_meta.json"
BANK_ARRAYS = "bank.npz"

#: 이미지로 볼 확장자. 대소문자를 가리지 않는다 — VisA 원본이 `.JPG` 다.
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

#: 추정 등급에서 이웃 이미지를 몇 단계까지 올라가 찾을 것인가.
NEIGHBOUR_DEPTH = 2


@dataclass
class BankRecord:
    """복원한 뱅크 하나."""

    bank_version: str
    directory: str
    #: recorded | inferred | unknown
    confidence: str
    images: list[str] = field(default_factory=list)
    patch_count: int = 0
    source_image_count: int = 0
    built_at: date | None = None
    line: str | None = None
    object_name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    #: 무엇을 보고 이렇게 판단했는가. 담당자 확인의 재료다.
    evidence: list[str] = field(default_factory=list)

    @property
    def is_estimated(self) -> bool:
        """확정으로 승격하려면 사람이 확인해야 한다."""
        return self.confidence != "recorded"

    def to_profile(self) -> BankProfile:
        """조회 계층이 쓰는 형태로. 판별 6번이 이것을 본다."""
        conditions: dict[str, list[str]] = {}
        # 조건은 이미지 경로에서 읽을 수 있는 것만 넣는다. 지어내지 않는다.
        folders = sorted({str(Path(p).parent) for p in self.images})
        if folders:
            conditions["source_folder"] = folders
        return BankProfile(
            bank_version=self.bank_version,
            line=self.line or "",
            object_name=self.object_name or "",
            source_image_count=self.source_image_count or len(self.images),
            patch_count=self.patch_count,
            conditions=conditions,
            built_at=self.built_at,
            is_estimated=self.is_estimated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_version": self.bank_version,
            "directory": self.directory,
            "confidence": self.confidence,
            "is_estimated": self.is_estimated,
            "patch_count": self.patch_count,
            "source_image_count": self.source_image_count,
            "built_at": self.built_at.isoformat() if self.built_at else None,
            "line": self.line,
            "object_name": self.object_name,
            "image_count": len(self.images),
            "evidence": self.evidence,
        }


@dataclass
class VersionDiff:
    """두 뱅크 버전 사이에 무엇이 들고 났는가."""

    before: str
    after: str
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    kept: int = 0
    #: 한쪽이라도 추정이면 이 비교도 추정이다.
    is_estimated: bool = False

    def describe(self) -> str:
        if not self.added and not self.removed:
            return f"{self.before} → {self.after}: 구성이 같다 (이미지 {self.kept}장)"
        parts = []
        if self.added:
            parts.append(f"{len(self.added)}장 추가")
        if self.removed:
            parts.append(f"{len(self.removed)}장 제거")
        tail = " (추정이라 담당자 확인 필요)" if self.is_estimated else ""
        return f"{self.before} → {self.after}: {', '.join(parts)}, {self.kept}장 유지{tail}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before, "after": self.after,
            "added": self.added, "removed": self.removed,
            "kept": self.kept, "is_estimated": self.is_estimated,
            "note": self.describe(),
        }


@dataclass
class ScanResult:
    """스캔 한 번의 결과."""

    root: str
    records: list[BankRecord] = field(default_factory=list)
    #: 뱅크처럼 생겼는데 이력이 없어 뱅크로 세지 않은 자리.
    skipped: list[str] = field(default_factory=list)

    def by_version(self, bank_version: str) -> BankRecord | None:
        for record in self.records:
            if record.bank_version == bank_version:
                return record
        return None

    def diff(self, before: str, after: str) -> VersionDiff | None:
        """두 버전의 구성 차이. 한쪽이라도 없으면 None.

        **재구성 전후에 무엇이 빠졌는지가 승인의 근거다.** "정상 이미지
        몇 장을 함께 버렸는가"를 답할 수 있어야 사람이 판단한다.
        """
        first, second = self.by_version(before), self.by_version(after)
        if first is None or second is None:
            return None
        old, new = set(first.images), set(second.images)
        return VersionDiff(
            before=before,
            after=after,
            added=sorted(new - old),
            removed=sorted(old - new),
            kept=len(old & new),
            is_estimated=first.is_estimated or second.is_estimated,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "records": [r.to_dict() for r in self.records],
            "skipped": self.skipped,
            "recorded": sum(1 for r in self.records if not r.is_estimated),
            "estimated": sum(1 for r in self.records if r.is_estimated),
        }


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _neighbour_images(directory: Path, depth: int = NEIGHBOUR_DEPTH) -> list[str]:
    """이웃에서 이미지를 찾는다. **후보일 뿐 구성 이력이 아니다.**

    벡터 파일만 있고 이력이 없을 때, 같은 폴더와 위로 몇 단계까지 훑어
    "이 근처 이미지들로 만들었을 것"을 제시한다. 담당자가 확인할 재료이지
    확정이 아니다.
    """
    seen: list[str] = []
    current = directory
    for _ in range(depth + 1):
        for path in sorted(current.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
                text = str(path)
                if text not in seen:
                    seen.append(text)
        if current.parent == current:
            break
        current = current.parent
    return seen


def _from_meta(directory: Path, meta_path: Path) -> BankRecord:
    """이력 파일이 있으면 그대로 읽는다. 추정이 아니다."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    images = [str(p) for p in meta.get("images", [])]
    built = _as_date(meta.get("built_at")) or date.fromtimestamp(
        meta_path.stat().st_mtime
    )
    version = str(meta.get("bank_version") or directory.name)

    evidence = [
        f"{meta_path.name} 을 읽었다 — 구성 이력 그 자체이며 추정이 아니다",
        f"이미지 {len(images)}장 · 패치 {int(meta.get('patch_count', 0)):,}개",
    ]
    if meta.get("feature_config"):
        evidence.append(f"특징 설정 {meta['feature_config']}")
    if meta.get("coreset_capped"):
        # 조용히 잘린 것을 "비율대로 만들어졌다"로 읽으면 안 된다.
        evidence.append(
            f"coreset 상한에 걸렸다 — 요청 {meta.get('coreset_ratio')} 가 "
            f"실제로는 {meta.get('max_bank_size'):,}개로 잘렸다"
        )

    return BankRecord(
        bank_version=version,
        directory=str(directory),
        confidence="recorded",
        images=images,
        patch_count=int(meta.get("patch_count", 0)),
        source_image_count=int(meta.get("source_image_count", len(images))),
        built_at=built,
        line=meta.get("line"),
        object_name=meta.get("object_name"),
        meta=meta,
        evidence=evidence,
    )


def _from_arrays_only(directory: Path, arrays_path: Path) -> BankRecord:
    """벡터만 있고 이력이 없다. **여기서 나온 이미지 목록은 후보다.**"""
    candidates = _neighbour_images(directory)
    return BankRecord(
        bank_version=directory.name,
        directory=str(directory),
        confidence="inferred",
        images=candidates,
        patch_count=0,
        source_image_count=0,
        built_at=date.fromtimestamp(arrays_path.stat().st_mtime),
        evidence=[
            f"{arrays_path.name} 은 있는데 {BANK_META} 가 없다 — 구성 이력이 없다",
            f"이웃 {NEIGHBOUR_DEPTH}단계에서 이미지 {len(candidates)}장을 후보로 잡았다",
            "**후보일 뿐 확정이 아니다.** 담당자 확인 후에만 확정으로 올린다",
        ],
    )


def scan_history(root: str | Path) -> ScanResult:
    """루트 아래를 걸어 뱅크 구성 이력을 복원한다.

    **폴더 구조를 전제하지 않는다.** 이름도 깊이도 가정하지 않고 이력
    파일과 벡터 파일이 있는 자리를 찾을 뿐이다.

    같은 `bank_version` 이 여러 자리에서 나오면 **확정을 우선**하고, 둘 다
    확정이면 나중에 만들어진 것을 남긴다. 조용히 하나를 고르지 않고 근거에
    적는다.
    """
    root = Path(root)
    result = ScanResult(root=str(root))
    if not root.exists():
        return result

    found: dict[str, BankRecord] = {}
    for meta_path in sorted(root.rglob(BANK_META)):
        record = _from_meta(meta_path.parent, meta_path)
        _keep_better(found, record, result)

    for arrays_path in sorted(root.rglob(BANK_ARRAYS)):
        directory = arrays_path.parent
        if (directory / BANK_META).exists():
            continue  # 위에서 확정으로 읽었다
        _keep_better(found, _from_arrays_only(directory, arrays_path), result)

    result.records = sorted(found.values(), key=lambda r: r.bank_version)
    return result


def _keep_better(found: dict[str, BankRecord], record: BankRecord,
                 result: ScanResult) -> None:
    """같은 버전이 두 자리에서 나왔을 때 무엇을 남길지 정한다."""
    previous = found.get(record.bank_version)
    if previous is None:
        found[record.bank_version] = record
        return

    # 확정이 추정을 이긴다. 둘 다 확정이면 나중 것.
    keep, drop = record, previous
    if previous.confidence == "recorded" and record.confidence != "recorded":
        keep, drop = previous, record
    elif previous.confidence == record.confidence:
        if (previous.built_at or date.min) >= (record.built_at or date.min):
            keep, drop = previous, record

    keep.evidence.append(
        f"같은 버전이 {drop.directory} 에도 있어 {drop.confidence} 쪽을 버렸다"
    )
    found[record.bank_version] = keep
    result.skipped.append(drop.directory)


def summarise(result: ScanResult) -> str:
    """사람이 읽는 한 문단. 승인 문서에 붙일 수 있게."""
    if not result.records:
        return f"{result.root} 아래에서 뱅크를 찾지 못했다."

    recorded = [r for r in result.records if not r.is_estimated]
    estimated = [r for r in result.records if r.is_estimated]

    lines = [
        f"{result.root} 에서 뱅크 {len(result.records)}개를 복원했다 "
        f"(확정 {len(recorded)} · 추정 {len(estimated)})."
    ]
    for record in result.records:
        mark = "확정" if not record.is_estimated else "추정"
        lines.append(
            f"  [{mark}] {record.bank_version} — 이미지 {len(record.images)}장 · "
            f"패치 {record.patch_count:,}개 · {record.built_at or '날짜 미상'}"
        )
    if estimated:
        lines.append(
            "추정으로 표시된 것은 구성 이력 파일이 없어 이웃 이미지를 후보로 "
            "잡은 것이다. **담당자 확인 후에만 확정으로 올린다.**"
        )
    return "\n".join(lines)
