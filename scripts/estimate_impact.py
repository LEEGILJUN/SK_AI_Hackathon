"""운영 임팩트를 산정한다. 측정에서 온 값과 가정에서 온 값을 갈라서.

    .venv/bin/python scripts/estimate_impact.py
    .venv/bin/python scripts/estimate_impact.py --issues-per-year 120 --diag-hours 6
    .venv/bin/python scripts/estimate_impact.py --markdown        # 표만

── 왜 스크립트로 만드는가 ──────────────────────────────────────────────

기대 효과를 수치로 적으라는 요구는 흔하고, 그때 나오는 숫자가 대개
**어디서 왔는지 알 수 없는 값**이다. 몇 % 절감이라고 적혀 있는데 무엇을
재서 나온 것인지 되짚을 수 없으면 그 숫자는 근거가 아니다.

여기서는 둘을 갈라 놓는다.

    측정   저장소에서 계산한다. 시나리오 파일과 실측 기록에서 온다
    가정   사람이 넣는다. 현장 값이라 우리가 모른다

**가정을 측정처럼 쓰지 않는다.** 화면에도 문서에도 어느 쪽인지 함께
찍는다. 가정이 틀리면 결론도 틀리는데, 어느 가정이 틀렸는지 짚을 수
있어야 고칠 수 있다.

── 무엇을 계산하는가 ───────────────────────────────────────────────────

미검출 이슈 한 건에 드는 사람 시간을 지금 방식과 우리 방식으로 비교한다.

    지금    원인 파악 → (짐작으로) 재구성 → 검증
    우리    자동 진단 → 재구성이 답일 때만 재구성 → 갈린 것만 검증 → 승인

**우리 쪽에도 사람 시간이 남는다.** 승인은 사람이 하고, 신구 비교에서
갈린 건도 사람이 본다. 그 둘을 빼지 않는다. 무인 운영을 만들지 않는
것이 이 과제의 설계이므로 계산에서도 빼면 안 된다.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SCENARIOS = REPO_ROOT / "data" / "scenarios.yaml"

#: 신구 비교 실측. 실모델 완주에서 홀드아웃 14장 중 판정이 갈린 것이 1장이었다.
#: 사람은 그 1장만 본다. `docs/실험_*` 와 CLAUDE.md 에 기록이 있다.
SHADOW_TOTAL = 14
SHADOW_REVIEW = 1

#: 전 구간 자동 실행 시간(초). 4090 실측, 실모델로 도구 10개 완주.
PIPELINE_SECONDS = 151

#: 진단 규칙 채점. `scripts/measure_rules.py` 가 내는 값이다.
DIAGNOSIS_CORRECT = 22
DIAGNOSIS_TOTAL = 24


@dataclass
class Measured:
    """저장소에서 계산한 값. 사람이 넣지 않는다."""

    scenarios: int
    rebuild_needed: int

    @property
    def rebuild_rate(self) -> float:
        return self.rebuild_needed / self.scenarios if self.scenarios else 0.0

    @property
    def no_rebuild(self) -> int:
        return self.scenarios - self.rebuild_needed

    @property
    def shadow_reduction(self) -> float:
        return 1 - (SHADOW_REVIEW / SHADOW_TOTAL)


def measure() -> Measured:
    """채점 기준 파일에서 재구성이 답인 비율을 센다.

    **정답 라벨에서 온다.** 우리 판정이 아니라 도메인 담당이 걸어 둔 값이라,
    "재구성이 답인 경우가 얼마나 되는가"의 근거로 쓸 수 있다.
    """
    import yaml

    data = yaml.safe_load(SCENARIOS.read_text(encoding="utf-8"))
    real = [
        s for s in data.get("scenarios", [])
        if not str(s.get("id", "")).startswith("SC-TEMPLATE")
    ]
    needed = sum(
        1 for s in real if (s.get("ground_truth") or {}).get("requires_bank_rebuild")
    )
    return Measured(scenarios=len(real), rebuild_needed=needed)


def main() -> int:
    parser = argparse.ArgumentParser(description="운영 임팩트 산정")
    # ── 가정. 전부 현장 값이라 우리가 모른다 ──────────────────────────
    parser.add_argument("--issues-per-year", type=int, default=100,
                        help="가정: 라인 하나에서 한 해에 접수되는 미검출 이슈 건수")
    parser.add_argument("--diag-hours", type=float, default=4.0,
                        help="가정: 한 건의 원인을 사람이 파악하는 데 드는 시간")
    parser.add_argument("--rebuild-hours", type=float, default=6.0,
                        help="가정: 뱅크를 다시 세우고 배포 후보를 만드는 데 드는 시간")
    parser.add_argument("--verify-hours", type=float, default=8.0,
                        help="가정: 좋아졌는지 사람이 확인하는 데 드는 시간")
    parser.add_argument("--approve-hours", type=float, default=1.0,
                        help="가정: 승인 문서를 읽고 판단하는 데 드는 시간")
    parser.add_argument("--as-is-rebuild-rate", type=float, default=1.0,
                        help="가정: 지금은 미검출이 오면 재학습을 거는 비율")
    parser.add_argument("--markdown", action="store_true", help="표만 출력")
    args = parser.parse_args()

    m = measure()
    auto_hours = PIPELINE_SECONDS / 3600

    # ── 지금 방식 ───────────────────────────────────────────────────
    as_is = args.diag_hours + args.as_is_rebuild_rate * (
        args.rebuild_hours + args.verify_hours
    )

    # ── 우리 방식 ───────────────────────────────────────────────────
    #
    # 재구성은 실제로 답일 때만 돈다. 검증은 신구 비교가 갈린 것만 남기므로
    # 확인 대상이 줄고, 승인은 사람이 그대로 한다.
    verify_after = args.verify_hours * (SHADOW_REVIEW / SHADOW_TOTAL)
    to_be = auto_hours + m.rebuild_rate * (args.rebuild_hours + verify_after) + args.approve_hours

    saved_each = as_is - to_be
    saved_year = saved_each * args.issues_per_year
    ratio = saved_each / as_is if as_is else 0.0

    rows = [
        ("측정", "채점 시나리오", f"{m.scenarios}건"),
        ("측정", "재구성이 답인 것", f"{m.rebuild_needed}건 ({m.rebuild_rate:.1%})"),
        ("측정", "재구성이 답이 아닌 것", f"{m.no_rebuild}건 ({1 - m.rebuild_rate:.1%})"),
        ("측정", "진단 원인 일치", f"{DIAGNOSIS_CORRECT}/{DIAGNOSIS_TOTAL}"),
        ("측정", "신구 비교 확인 대상", f"{SHADOW_TOTAL}장 중 {SHADOW_REVIEW}장 ({m.shadow_reduction:.1%} 감소)"),
        ("측정", "전 구간 자동 실행", f"{PIPELINE_SECONDS}초 ({auto_hours:.3f}시간)"),
        ("가정", "연간 이슈", f"{args.issues_per_year}건"),
        ("가정", "원인 파악", f"{args.diag_hours}시간"),
        ("가정", "재구성", f"{args.rebuild_hours}시간"),
        ("가정", "성능 확인", f"{args.verify_hours}시간"),
        ("가정", "승인 검토", f"{args.approve_hours}시간"),
        ("가정", "지금의 재구성 비율", f"{args.as_is_rebuild_rate:.0%}"),
    ]

    if args.markdown:
        print("| 구분 | 항목 | 값 |")
        print("|---|---|---|")
        for kind, name, value in rows:
            print(f"| {kind} | {name} | {value} |")
        print()
        print("| 항목 | 지금 | 우리 방식 |")
        print("|---|---|---|")
        print(f"| 한 건에 드는 사람 시간 | {as_is:.1f}시간 | {to_be:.1f}시간 |")
        print(f"| 연간 ({args.issues_per_year}건) | {as_is * args.issues_per_year:,.0f}시간 "
              f"| {to_be * args.issues_per_year:,.0f}시간 |")
        return 0

    print("운영 임팩트 산정\n")
    print("  구분  항목                       값")
    for kind, name, value in rows:
        print(f"  {kind}  {name:24} {value}")

    print(f"\n  한 건에 드는 사람 시간   지금 {as_is:.1f}시간 → 우리 방식 {to_be:.1f}시간")
    print(f"  건당 절감               {saved_each:.1f}시간 ({ratio:.0%})")
    print(f"  연간 절감               {saved_year:,.0f}시간 "
          f"({args.issues_per_year}건 기준)")

    print(
        "\n**아래 두 줄만 측정에서 왔습니다.**\n"
        f"  재구성이 답인 경우는 {m.rebuild_needed}/{m.scenarios} 이고, 나머지 "
        f"{m.no_rebuild}건에서 뱅크를 건드리지 않습니다.\n"
        f"  신구 비교가 사람 확인 대상을 {SHADOW_TOTAL}장에서 {SHADOW_REVIEW}장으로 줄입니다.\n"
        "\n**시간 값은 전부 가정입니다.** 현장 값이라 우리가 모릅니다. "
        "도메인 담당이 확정하면\n그때 이 명령의 인자만 바꾸면 됩니다. "
        "가정을 바꿔도 위 두 줄은 안 변합니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
