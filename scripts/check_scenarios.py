"""시나리오 파일 검사 — 장영진용.

`data/scenarios.yaml` 을 채우다가 형식이 맞는지 확인할 때 씁니다.
코드를 읽지 않아도 무엇이 잘못됐는지 한국어로 알려줍니다.

실행:
    .venv/bin/python scripts/check_scenarios.py
    .venv/bin/python scripts/check_scenarios.py examples/scenarios_예시.yaml

검사하는 것
    - 필수 항목이 빠지지 않았는가
    - 원인 코드가 정해진 6종 안에 있는가
    - requires_bank_rebuild 가 원인과 맞는가 (이게 정량 목표의 채점 기준입니다)
    - 판별 항목 값이 허용된 값인가
    - 시나리오 20건과 원인별 배분이 채워졌는가

이 검사를 통과한다고 정답이 맞다는 뜻은 아닙니다. 형식만 봅니다.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PATH = REPO_ROOT / "data" / "scenarios.yaml"

VALID_CAUSES = {
    "threshold": False,
    "bank_contamination": True,
    "coverage_gap": True,
    "normal_overlap": False,
    "equipment_optics": False,
    "criteria": False,
}
CAUSE_KO = {
    "threshold": "임계값 문제",
    "bank_contamination": "뱅크 오염",
    "coverage_gap": "커버리지 부족",
    "normal_overlap": "정상 분포 중첩",
    "equipment_optics": "설비·광학",
    "criteria": "기준 문제",
}
VALID_INTAKE = {"proceed", "need_more_info", "duplicate"}
VALID_SCORE_POSITION = {"above", "near", "below"}
VALID_CRITERIA_VERDICT = {"defect", "pass", "review", "out_of_scope"}

TARGET_COUNT = 20
TARGET_PER_CAUSE = 3

problems: list[str] = []
warnings: list[str] = []


def problem(scenario_id: str, message: str) -> None:
    problems.append(f"  [{scenario_id}] {message}")


def warn(scenario_id: str, message: str) -> None:
    warnings.append(f"  [{scenario_id}] {message}")


def check_scenario(index: int, entry: dict) -> str | None:
    """시나리오 한 건을 검사하고 원인 코드를 돌려준다."""
    sid = entry.get("id") or f"{index + 1}번째 항목(id 없음)"

    if not entry.get("id"):
        problem(sid, "id 가 없습니다. SC-BC-001 처럼 붙여 주세요.")
    if not entry.get("title"):
        problem(sid, "title 이 없습니다.")

    # ── 입력 ────────────────────────────────────────────────────────
    issue_input = entry.get("input") or {}
    text = (issue_input.get("issue_text") or "").strip()
    if not text:
        problem(sid, "input.issue_text 가 비어 있습니다. 접수자가 칠 문장이 필요합니다.")
    elif len(text) < 10:
        warn(sid, f"issue_text 가 너무 짧습니다({len(text)}자). 실제 접수 문장처럼 써 주세요.")

    # ── 정답 ────────────────────────────────────────────────────────
    truth = entry.get("ground_truth")
    if truth is None:
        problem(sid, "ground_truth 가 없습니다. 정답이 없으면 채점할 수 없습니다.")
        return None

    intake = truth.get("intake_verdict")
    if intake not in VALID_INTAKE:
        problem(sid, f"ground_truth.intake_verdict 가 {intake!r} 입니다. {sorted(VALID_INTAKE)} 중 하나여야 합니다.")

    cause = truth.get("cause")
    rebuild = truth.get("requires_bank_rebuild")

    if intake in {"need_more_info", "duplicate"}:
        # 진단으로 넘어가지 않는 케이스. 원인이 없는 것이 정상이다.
        if cause is not None:
            warn(sid, f"intake_verdict 가 {intake} 인데 cause 가 채워져 있습니다. 진단으로 넘어가지 않는 케이스라면 null 이어야 합니다.")
        if intake == "duplicate" and not truth.get("duplicate_of"):
            problem(sid, "intake_verdict 가 duplicate 인데 duplicate_of 가 비어 있습니다.")
        return None

    if cause not in VALID_CAUSES:
        problem(sid, f"ground_truth.cause 가 {cause!r} 입니다. {sorted(VALID_CAUSES)} 중 하나여야 합니다.")
        return None

    expected_rebuild = VALID_CAUSES[cause]
    if rebuild is None:
        problem(sid, f"requires_bank_rebuild 가 없습니다. {CAUSE_KO[cause]} 이면 {expected_rebuild} 입니다.")
    elif rebuild != expected_rebuild:
        problem(
            sid,
            f"requires_bank_rebuild 가 {rebuild} 인데 {CAUSE_KO[cause]} 이면 {expected_rebuild} 여야 합니다. "
            f"이 값은 원인이 정해지면 따라옵니다. 손으로 정하지 마세요.",
        )

    # ── 판별 항목 ───────────────────────────────────────────────────
    evidence = truth.get("evidence")
    if evidence is None:
        warn(sid, "evidence 가 없습니다. 원인만 맞고 근거가 틀린 답을 걸러낼 수 없습니다.")
    else:
        position = evidence.get("score_vs_threshold")
        if position is not None and position not in VALID_SCORE_POSITION:
            problem(sid, f"evidence.score_vs_threshold 가 {position!r} 입니다. {sorted(VALID_SCORE_POSITION)} 중 하나여야 합니다.")

        verdict = evidence.get("criteria_verdict")
        if verdict is not None and verdict not in VALID_CRITERIA_VERDICT:
            problem(sid, f"evidence.criteria_verdict 가 {verdict!r} 입니다. {sorted(VALID_CRITERIA_VERDICT)} 중 하나여야 합니다.")

        for key in ("defect_visible", "quality_within_baseline", "nearest_patch_is_defect", "coverage_present"):
            value = evidence.get(key)
            if value is not None and not isinstance(value, bool):
                problem(sid, f"evidence.{key} 는 true 또는 false 여야 합니다. 지금은 {value!r} 입니다.")

        # 원인별로 반드시 맞아야 하는 값
        if cause == "bank_contamination" and evidence.get("nearest_patch_is_defect") is not True:
            problem(sid, "뱅크 오염이면 evidence.nearest_patch_is_defect 가 true 여야 합니다. 이 값이 원인을 가릅니다.")
        if cause == "normal_overlap" and evidence.get("nearest_patch_is_defect") is not False:
            problem(sid, "정상 분포 중첩이면 evidence.nearest_patch_is_defect 가 false 여야 합니다. 뱅크 오염과 여기서 갈립니다.")
        if cause == "equipment_optics" and evidence.get("quality_within_baseline") is not False:
            problem(sid, "설비·광학이면 evidence.quality_within_baseline 가 false 여야 합니다.")
        if cause == "coverage_gap" and evidence.get("coverage_present") is not False:
            problem(sid, "커버리지 부족이면 evidence.coverage_present 가 false 여야 합니다.")

    # ── 조치 ────────────────────────────────────────────────────────
    actions = truth.get("expected_actions") or []
    if not actions:
        warn(sid, "expected_actions 가 비어 있습니다. 무엇을 해야 하는지가 정답의 일부입니다.")
    if not expected_rebuild and "rebuild_bank" in actions:
        problem(sid, f"{CAUSE_KO[cause]} 인데 expected_actions 에 rebuild_bank 가 있습니다. 재구성이 답이 아닌 원인입니다.")

    # ── 재현 ────────────────────────────────────────────────────────
    injection = entry.get("injection")
    if injection is None:
        warn(sid, "injection 이 없습니다. 이 상황을 어떻게 재현하는지 적어야 데이터를 만들 수 있습니다.")
    elif injection.get("method") not in (None, "none") and injection.get("seed") is None:
        warn(sid, "injection.seed 가 없습니다. 시드를 고정해야 누구나 같은 데이터를 만듭니다.")

    return cause


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.exists():
        print(f"파일이 없습니다: {path}")
        return 1

    print(f"검사 대상: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}\n")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        print("YAML 형식이 깨졌습니다. 들여쓰기나 따옴표를 확인하세요.\n")
        print(f"  {exc}")
        return 1

    scenarios = data.get("scenarios") or []
    if not scenarios:
        print("scenarios 항목이 비어 있습니다. 아직 채우지 않으셨다면 정상입니다.")
        print("작성 예시는 examples/scenarios_예시.yaml 을 보세요.")
        return 0

    causes = [check_scenario(i, s) for i, s in enumerate(scenarios)]
    counts = Counter(c for c in causes if c)

    # ── 결과 ────────────────────────────────────────────────────────
    print(f"시나리오 {len(scenarios)}건")
    for code in VALID_CAUSES:
        n = counts.get(code, 0)
        mark = "OK" if n >= TARGET_PER_CAUSE else f"{TARGET_PER_CAUSE - n}건 부족"
        print(f"  {CAUSE_KO[code]:<12} {n}건   {mark}")
    other = len(scenarios) - sum(counts.values())
    if other:
        print(f"  {'진단 이전 단계':<12} {other}건   (need_more_info · duplicate)")

    if len(scenarios) < TARGET_COUNT:
        print(f"\n목표 {TARGET_COUNT}건 중 {len(scenarios)}건. {TARGET_COUNT - len(scenarios)}건 남았습니다.")

    if problems:
        print(f"\n고쳐야 할 것 {len(problems)}건")
        for line in problems:
            print(line)
    if warnings:
        print(f"\n확인해 보실 것 {len(warnings)}건")
        for line in warnings:
            print(line)

    if not problems and not warnings:
        print("\n형식 문제 없습니다.")

    print(
        "\n이 검사는 형식만 봅니다. 정답이 맞는지는 사람이 판단해야 합니다."
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
