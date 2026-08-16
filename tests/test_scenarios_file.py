"""채점 대상이 몇 건인가 — 이 수가 흔들리면 정확도가 흔들린다.

**한 번 흔들렸다.** 파일 끝의 스키마 예시가 실제 시나리오와 같은 `SC-BC-001`
이라, 채점 스크립트는 **id 중복 검사로** 그것을 걸러 내고 있었다. 겹친 id 를
떼어 내자 거름망이 풀려 채점 대상이 24 → 25 로 늘었고, 정확도가
19/24(79%)에서 20/25(80%)로 **좋아진 것처럼 보였다.**

목표가 80% 라 그대로 뒀으면 기획서에 "달성"으로 들어갔다.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = yaml.safe_load(
    (REPO_ROOT / "data" / "scenarios.yaml").read_text(encoding="utf-8")
)["scenarios"]


def _real():
    return [s for s in SCENARIOS if not s["id"].startswith("SC-TEMPLATE")]


def test_the_schema_example_is_not_a_scenario():
    """예시는 채점 대상이 아니다.

    도메인 담당이 "예시 1건 — 아래 형식을 그대로 복제해 나머지를 채웁니다" 라고
    적어 둔 골격이고, 대상 라인·품목이 공장 구성에 없어 생성기도 걸러 낸다.
    """
    examples = [s for s in SCENARIOS if s["id"].startswith("SC-TEMPLATE")]
    assert len(examples) == 1, f"예시가 {len(examples)}건이다"
    assert len(_real()) == 24, f"채점 대상이 {len(_real())}건이다 — 24건이어야 한다"


def test_every_scenario_id_is_unique():
    """id 가 겹치면 id 로 찾는 코드가 생기는 순간 하나가 조용히 이긴다."""
    ids = [s["id"] for s in SCENARIOS]
    assert len(ids) == len(set(ids)), f"겹친 id: {sorted({i for i in ids if ids.count(i) > 1})}"


def test_the_six_causes_have_four_each():
    """원인 6종 × 4건. 한쪽으로 쏠리면 정확도가 그 원인의 성적이 된다."""
    from collections import Counter

    counts = Counter(s["cause_group"] for s in _real())
    assert len(counts) == 6, f"원인이 {len(counts)}종이다"
    assert set(counts.values()) == {4}, f"건수가 고르지 않다: {dict(counts)}"


def test_the_scoring_script_excludes_the_example():
    """채점 스크립트가 24건을 센다.

    **이 시험이 실패하면 정확도 분모가 바뀐 것이다.** 기획서와 발표 자료의
    수치가 전부 이 분모 위에 있다.
    """
    done = subprocess.run(
        [sys.executable, "scripts/measure_rules.py"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, cwd=REPO_ROOT,
    )
    assert done.returncode == 0, done.stderr[-800:]
    assert "원인 일치        19/24" in done.stdout or "/24" in done.stdout, (
        f"분모가 24 가 아니다:\n{done.stdout[-600:]}"
    )
    assert "/25" not in done.stdout, "예시를 함께 세고 있다"
