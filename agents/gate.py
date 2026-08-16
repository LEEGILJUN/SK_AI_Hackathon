"""평가 게이트 — 배포 후보로 넘길 것인가 (작업 17).

새 뱅크가 나왔다고 쓸 수 있는 것이 아니다. 세 가지를 통과해야 한다.

  1. 홀드아웃 성능   고치려던 문제가 실제로 나아졌는가
  2. 기준셋          이미 잡고 있던 것을 계속 잡는가 (회귀 방지)
  3. 재현성          같은 입력에 같은 판정이 나오는가

2번이 특히 중요하다. 혼입 이미지를 제거해 목표 결함은 잡게 됐는데 다른 것을 놓치기
시작하면 개선이 아니다. 섀도 비교의 newly_missed 가 같은 것을 본다.

**통과 기준은 도메인이 정하고 `data/gate.yaml` 에 있다.** 값마다 근거가 함께
적혀 있고 라인별로 덮어쓸 수 있다. 코드에 박아 두면 바꿀 때마다 커밋이
필요하고, 나중에 왜 그 값인지 아무도 답하지 못한다.

재현성이 목표에 들어간 이유가 있다. 같은 입력에 판정이 흔들리면 게이트를
몇 번 돌려 통과할 때까지 재시도하는 일이 생긴다. 그러면 게이트가 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from inspection.shadow import ShadowReport
from inspection.sweep import ThresholdCurve, sweep_thresholds


#: 통과 기준 파일. 값과 그 근거가 함께 있다.
CRITERIA_PATH = Path(__file__).resolve().parent.parent / "data" / "gate.yaml"


@dataclass
class GateCriteria:
    """통과 기준.

    **여기 있는 값은 `data/gate.yaml` 이 없을 때의 대비책이다.** 실제 값은
    파일에서 오고, 근거도 거기 함께 적혀 있다. 코드에 박아 두면 바꿀 때마다
    커밋이 필요하고 나중에 왜 그 값인지 아무도 답하지 못한다.

    라인마다 다를 수 있다. 과검 한 건의 무게가 라인마다 다르기 때문이다.
    `load(line)` 이 그것을 푼다.
    """

    min_detection_rate: float = 0.90      # 홀드아웃 검출률 하한
    max_false_positive_rate: float = 0.05  # 홀드아웃 과검률 상한
    min_auroc: float = 0.85                # 분리도 하한
    max_newly_missed: int = 0              # 섀도에서 새로 놓치는 건수 상한
    require_improvement: bool = True       # 이전보다 나아져야 하는가
    reproducibility_runs: int = 10         # 재현성 확인 반복 횟수

    @classmethod
    def load(cls, line: str | None = None, path: str | Path | None = None) -> "GateCriteria":
        """설정 파일에서 읽는다. 라인 설정이 있으면 기본값 위에 덮어쓴다.

        **파일이 없거나 깨져도 예외를 던지지 않는다.** 게이트가 못 서는 것보다
        기본값으로라도 도는 편이 낫고, 어느 값으로 판정했는지는 결과에 남는다.

        모르는 항목은 조용히 버린다. 오타 하나로 게이트가 안 서면 시연 중에
        고칠 수 없다.
        """
        source = Path(path) if path is not None else CRITERIA_PATH
        try:
            loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return cls()

        known = {f.name for f in fields(cls)}
        values: dict[str, Any] = {
            k: v for k, v in (loaded.get("defaults") or {}).items() if k in known
        }
        if line:
            per_line = (loaded.get("lines") or {}).get(line) or {}
            values.update({k: v for k, v in per_line.items() if k in known})
        return cls(**values)


@dataclass
class CheckResult:
    """검사 한 항목의 결과."""

    name: str
    passed: bool
    value: Any
    threshold: Any
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateResult:
    """게이트 판정.

    passed 가 False 여도 왜 떨어졌는지가 남아야 다음 조치를 정할 수 있다.
    "못 통과했다"만으로는 데이터를 더 넣을지 계획을 바꿀지 알 수 없다.
    """

    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    reason: str = ""
    candidate_version: str = ""

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "candidate_version": self.candidate_version,
            "reason": self.reason,
            "checks": [c.to_dict() for c in self.checks],
        }


def evaluate_gate(
    normal_scores: Sequence[float],
    defect_scores: Sequence[float],
    threshold: float,
    criteria: GateCriteria | None = None,
    shadow: ShadowReport | None = None,
    baseline_curve: ThresholdCurve | None = None,
    candidate_version: str = "",
) -> GateResult:
    """홀드아웃 성능과 섀도 결과로 게이트를 판정한다.

    baseline_curve
        이전 뱅크의 곡선. 주면 나아졌는지를 함께 본다.
    """
    criteria = criteria or GateCriteria()
    checks: list[CheckResult] = []

    curve = sweep_thresholds(normal_scores, defect_scores, current_threshold=threshold)
    point = curve.at_threshold(threshold)
    auroc = curve.auroc()

    # ── 1. 홀드아웃 성능 ────────────────────────────────────────────
    checks.append(
        CheckResult(
            name="detection_rate",
            passed=point.detection_rate >= criteria.min_detection_rate,
            value=round(point.detection_rate, 4),
            threshold=criteria.min_detection_rate,
            detail=f"임계값 {threshold:.4f} 에서 불량 {point.detected}/{point.detected + point.missed} 검출",
        )
    )
    checks.append(
        CheckResult(
            name="false_positive_rate",
            passed=point.false_positive_rate <= criteria.max_false_positive_rate,
            value=round(point.false_positive_rate, 4),
            threshold=criteria.max_false_positive_rate,
            detail=f"양품 {point.false_positives}/{point.false_positives + point.true_negatives} 과검",
        )
    )
    checks.append(
        CheckResult(
            name="auroc",
            passed=auroc >= criteria.min_auroc,
            value=round(auroc, 4),
            threshold=criteria.min_auroc,
            detail="임계값과 무관한 분리도. 낮으면 어디에 두어도 안 된다",
        )
    )

    # ── 2. 회귀 방지 ────────────────────────────────────────────────
    if shadow is not None:
        missed = len(shadow.newly_missed)
        checks.append(
            CheckResult(
                name="newly_missed",
                passed=missed <= criteria.max_newly_missed,
                value=missed,
                threshold=criteria.max_newly_missed,
                detail=(
                    f"섀도 비교에서 새로 놓친 건 {missed}건, 새로 잡은 건 "
                    f"{len(shadow.newly_detected)}건. 고치려던 문제가 나아져도 "
                    f"다른 것을 잃으면 개선이 아니다"
                ),
            )
        )

    if criteria.require_improvement and baseline_curve is not None:
        before = baseline_curve.auroc()
        checks.append(
            CheckResult(
                name="improvement",
                passed=auroc >= before,
                value=round(auroc, 4),
                threshold=round(before, 4),
                detail=f"이전 뱅크 AUROC {before:.4f} → 후보 {auroc:.4f}",
            )
        )

    passed = all(c.passed for c in checks)
    if passed:
        reason = (
            f"모든 항목을 통과했다. 검출률 {point.detection_rate:.0%}, "
            f"과검률 {point.false_positive_rate:.1%}, AUROC {auroc:.3f}. "
            f"배포 후보로 넘길 수 있으나 승인은 사람이 한다."
        )
    else:
        failed = ", ".join(f"{c.name}({c.value} vs 기준 {c.threshold})" for c in checks if not c.passed)
        reason = f"통과하지 못했다: {failed}."

    return GateResult(passed=passed, checks=checks, reason=reason, candidate_version=candidate_version)


# ── 재현성 ──────────────────────────────────────────────────────────────


@dataclass
class ReproducibilityResult:
    """같은 입력을 여러 번 돌렸을 때 판정이 같은가.

    정량 목표가 100% 다. 흔들리면 게이트를 통과할 때까지 재시도하는 일이
    생기고, 그러면 게이트로서 의미가 없다.
    """

    runs: int
    identical: bool
    distinct_outcomes: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_reproducibility(
    run_once: Callable[[], Any],
    runs: int = 10,
    key: Callable[[Any], str] | None = None,
) -> ReproducibilityResult:
    """같은 절차를 여러 번 돌려 결과가 같은지 본다.

    run_once
        한 번 실행하고 결과를 돌려주는 함수. 게이트 판정이든 진단이든
        같은 방식으로 잴 수 있다.
    key
        결과를 비교 가능한 문자열로 바꾸는 함수. 기본은 repr.
    """
    if runs < 2:
        raise ValueError(f"재현성은 2회 이상 돌려야 잰다: {runs}")

    to_key = key or (lambda value: repr(value))
    seen: list[str] = []
    for _ in range(runs):
        seen.append(to_key(run_once()))

    distinct = sorted(set(seen))
    identical = len(distinct) == 1

    return ReproducibilityResult(
        runs=runs,
        identical=identical,
        distinct_outcomes=distinct if not identical else distinct[:1],
        detail=(
            f"{runs}회 모두 같은 결과"
            if identical
            else f"{runs}회 중 서로 다른 결과가 {len(distinct)}종류 나왔다. 랜덤 요소를 시드로 묶어야 한다"
        ),
    )
