"""뱅크 재구성 실행 (작업 16).

큐레이션이 세운 계획을 실제 뱅크로 옮긴다. PatchCore 는 역전파가 없어
재구성이 수 분이면 끝나므로, 프롬프트 한 줄에서 시작해 새 뱅크가 나오기까지가
한 번의 대화 안에서 닫힌다. 이 과제가 노리는 지점이다.

**새 뱅크를 배포하지 않는다.** 여기서 만드는 것은 후보이고, 실제 판정에
쓰려면 평가 게이트를 통과하고 사람이 승인해야 한다. 품질 검사 설비의 특성상
의도적으로 그은 경계다.

기록을 남기는 것이 실행만큼 중요하다. "왜 이 시점에 뱅크를 바꿨나"에
사후에 답할 수 있어야 조직 자산이 된다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from inspection.bank import MemoryBank, build_bank
from inspection.features import PatchEmbedder

from .curate import CurationPlan


@runtime_checkable
class ImageSource(Protocol):
    """재구성에 쓸 이미지를 찾아 주는 계층.

    조회 계층(lookup)이 이 역할을 겸할 수도 있고, 별도로 둘 수도 있다.
    큐레이션 계획이 이미지 목록이 아니라 **조건**을 담고 있으므로,
    조건을 이미지로 바꾸는 일이 여기서 일어난다.
    """

    def images_for(self, condition_key: str, condition_value: str) -> list[str]:
        """그 조건에 해당하는 정상 이미지들의 상대 경로."""
        ...

    def resolve(self, relative_path: str) -> Path:
        """상대 경로를 실제 파일 경로로."""
        ...


@dataclass
class RebuildRecord:
    """무엇을 왜 바꿨는가 — 거버넌스 기록.

    새 뱅크 파일과 함께 저장되어, 나중에 "이 뱅크는 어떻게 만들어졌나"에
    답한다. 사람이 읽을 수 있어야 하므로 근거를 문장으로 남긴다.
    """

    from_version: str
    to_version: str
    cause: str | None
    removed: list[str] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    kept_count: int = 0
    reason: str = ""
    plan_summary: str = ""
    triggered_by: str = ""          # 누가·무엇이 시작했는가 (프롬프트 원문 등)
    approved_for_deploy: bool = False   # 항상 False. 승인은 사람이 별도로 한다

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RebuildResult:
    """재구성 결과.

    executed 가 False 면 실행하지 않은 것이다. 계획이 뱅크를 건드리지 않기로
    했거나, 실행할 수 없는 상태였다는 뜻이며 reason 에 이유가 있다.
    """

    executed: bool
    bank: MemoryBank | None = None
    record: RebuildRecord | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed": self.executed,
            "reason": self.reason,
            "record": self.record.to_dict() if self.record else None,
            "bank_size": len(self.bank) if self.bank else 0,
        }


def _next_version(current: str) -> str:
    """판 번호를 하나 올린다. `pcb1-01-v1` → `pcb1-01-v2`, `v3` → `v4`.

    **끝의 `v<숫자>` 를 본다.** 전에는 이름 전체가 `v3` 형태라고 보고
    `startswith("v")` 로 걸렀는데, 실제 값은 `lookup.base.bank_version_for`
    가 만드는 `pcb1-01-v1` 이라 한 번도 걸린 적이 없다. 그래서 재구성할 때마다
    `-rebuilt` 가 뒤에 붙어 `pcb1-01-v1-rebuilt-rebuilt` 가 됐고, **판 번호로
    앞뒤를 가릴 수 없어 원복 대상이 정해지지 않았다.**

    규칙에 안 맞는 이름이면 그대로 `-rebuilt` 를 붙인다. 사람이 손으로 지은
    이름일 수 있고, 조용히 번호를 붙이면 남의 판을 덮는다.
    """
    match = re.search(r"v(\d+)$", current)
    if match:
        return f"{current[:match.start()]}v{int(match.group(1)) + 1}"
    return f"{current}-rebuilt"


def execute_rebuild(
    plan: CurationPlan,
    current_bank: MemoryBank,
    source: ImageSource,
    embedder: PatchEmbedder | None = None,
    triggered_by: str = "",
    new_version: str | None = None,
    **build_kwargs: Any,
) -> RebuildResult:
    """큐레이션 계획대로 새 뱅크를 만든다.

    현재 뱅크의 구성 이미지에서 계획한 것을 빼고 더한 뒤 다시 만든다.
    시드와 설정은 현재 뱅크와 같게 유지한다. 그래야 달라진 것이 구성뿐이고,
    성능 차이를 구성 변화로 해석할 수 있다.
    """
    if not plan.touches_bank:
        return RebuildResult(
            executed=False,
            reason=f"계획이 뱅크를 건드리지 않기로 했다. {plan.reason}",
        )

    if plan.is_empty:
        return RebuildResult(
            executed=False,
            reason="계획에 제거·보충 대상이 없어 실행할 것이 없다.",
        )

    # ── 구성 이미지 계산 ────────────────────────────────────────────
    remove_set = {c.image for c in plan.remove}
    kept = [image for image in current_bank.images if image not in remove_set]

    not_found = remove_set - set(current_bank.images)
    added: list[str] = []
    #: 보충하겠다고 해 놓고 한 장도 못 찾은 조건. **조용히 넘어가면 안 된다.**
    #:
    #: 실측에서 `conditions` 를 안 넘겨 `images_for` 가 언제나 빈 목록이었다.
    #: 그런데도 `composition = kept` 라 재구성이 성공으로 끝났다 — 화면은
    #: "보충하겠다"고 말하고 0장을 넣은 뒤 성공이라 적었다. 뱅크가 안 바뀌는데
    #: 성공으로 보이는 것이 이 계열에서 가장 위험한 실패다.
    empty: list[str] = []
    for request in plan.add:
        found = source.images_for(request.condition_key, request.condition_value)
        fresh = [image for image in found if image not in kept]
        if not fresh:
            empty.append(f"{request.condition_key}={request.condition_value}")
        added.extend(fresh)

    if plan.add and not added and not remove_set:
        # 뺄 것도 없고 넣은 것도 없으면 **뱅크가 그대로다.** 성공이라 적으면
        # 화면이 "재구성했다"고 말하는데 실제로는 아무 일도 안 일어난 것이다.
        return RebuildResult(
            executed=False,
            reason=(f"보충하려던 조건({', '.join(empty)})의 이미지를 하나도 찾지 "
                    f"못해 뱅크가 그대로다. 조회 계층에 그 조건의 정상 이미지가 "
                    f"있는지 확인해야 한다."),
        )

    composition = kept + added
    if not composition:
        return RebuildResult(
            executed=False,
            reason="제거하고 나니 남는 이미지가 없다. 계획이 잘못됐다.",
        )

    # ── 재구성 ──────────────────────────────────────────────────────
    embedder = embedder or PatchEmbedder()
    version = new_version or _next_version(current_bank.version)
    meta = current_bank.meta

    kwargs: dict[str, Any] = {
        "coreset_ratio": meta.get("coreset_ratio", 0.01),
        "seed": meta.get("seed", 0),
        "projection_dim": meta.get("projection_dim", 128),
        "max_bank_size": meta.get("max_bank_size"),
    }
    kwargs.update(build_kwargs)

    paths = [source.resolve(image) for image in composition]
    new_bank = build_bank(
        paths,
        embedder,
        bank_version=version,
        extra_meta={
            "rebuilt_from": current_bank.version,
            "cause": plan.cause,
            "removed_images": sorted(remove_set - not_found),
            "added_images": added,
            "triggered_by": triggered_by,
            "approved_for_deploy": False,
        },
        **kwargs,
    )
    # 상대 경로 표기를 현재 뱅크와 맞춘다.
    new_bank.images = composition

    reason_parts = []
    if remove_set:
        reason_parts.append(f"{len(remove_set - not_found)}장 제거")
    if added:
        reason_parts.append(f"{len(added)}장 보충")
    if not_found:
        reason_parts.append(f"(제거 대상 {len(not_found)}장은 현재 뱅크에 없어 건너뜀)")
    if empty:
        # **보충이 0장이면 그렇다고 적는다.** 계획이 "보충하겠다" 인데 결과가
        # 침묵이면, 받는 쪽은 보충이 된 줄 안다.
        reason_parts.append(
            f"보충할 이미지를 못 찾은 조건 {len(empty)}개({', '.join(empty)}) — "
            f"이 조건의 정상 이미지가 조회 계층에 없다"
        )

    record = RebuildRecord(
        from_version=current_bank.version,
        to_version=version,
        cause=plan.cause,
        removed=sorted(remove_set - not_found),
        added=added,
        kept_count=len(kept),
        reason=f"{plan.reason} 실행 결과: {', '.join(reason_parts)}.",
        plan_summary=plan.summary(),
        triggered_by=triggered_by,
        approved_for_deploy=False,
    )

    return RebuildResult(executed=True, bank=new_bank, record=record, reason=record.reason)


def compare_banks(before: MemoryBank, after: MemoryBank) -> dict[str, Any]:
    """두 뱅크의 구성 차이. 재구성이 의도대로 됐는지 확인하는 용도다."""
    before_images = set(before.images)
    after_images = set(after.images)
    return {
        "from_version": before.version,
        "to_version": after.version,
        "images_before": len(before_images),
        "images_after": len(after_images),
        "removed": sorted(before_images - after_images),
        "added": sorted(after_images - before_images),
        "patches_before": len(before),
        "patches_after": len(after),
    }


class DirectoryImageSource:
    """폴더 기반 이미지 소스 — 조회 계층 실구현이 오기 전까지의 단순 구현.

    manifest.csv 가 준비되면 그것을 읽는 구현으로 대체한다. 인터페이스가
    같으므로 재구성 코드는 고치지 않는다.
    """

    def __init__(self, root: str | Path, conditions: dict[str, dict[str, list[str]]] | None = None):
        """
        conditions
            {"date": {"2026-06-05": ["line_02/.../img_1.png", ...]}} 형태.
            조건 → 이미지 목록. 없으면 images_for 가 빈 목록을 돌려준다.
        """
        self.root = Path(root)
        self.conditions = conditions or {}

    def images_for(self, condition_key: str, condition_value: str) -> list[str]:
        return list(self.conditions.get(condition_key, {}).get(condition_value, []))

    def resolve(self, relative_path: str) -> Path:
        return self.root / relative_path
