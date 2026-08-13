"""진단 규칙 정확도를 잰다 — 근거는 정답을 그대로 먹인다.

이 측정이 보는 것은 decide() 하나다. 이미지도 뱅크도 조회 계층도 쓰지 않고,
scenarios.yaml 의 ground_truth.evidence 를 그대로 Evidence 로 옮겨 넣는다.
그래서 여기서 틀리면 증거 수집을 아무리 잘해도 틀린다.

읽기만 한다. 저장소 파일을 고치지 않는다.

    python scripts/measure_rules.py            판별 3번을 시나리오 라벨에서 가져옴
    python scripts/measure_rules.py --computed 판별 3번을 score_ratio 에서 계산
"""

import sys
from pathlib import Path

import yaml

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else Path.cwd()
sys.path.insert(0, str(REPO))

from agents.diagnose import Evidence, decide  # noqa: E402

# 판별 3번을 라벨에서 가져올지 비율에서 계산할지. 운영에서는 계산이다.
USE_COMPUTED_POSITION = "--computed" in sys.argv
NEAR_RATIO = 0.9  # inspection/types.py score_position() 의 기본값


def _verdict(flag: object) -> str:
    """시나리오는 참/거짓으로 적고 코드는 defect/normal 로 읽는다."""
    if flag is None:
        return "unknown"
    return "defect" if flag else "normal"


def _position(ev: dict) -> str | None:
    if not USE_COMPUTED_POSITION:
        return ev.get("score_vs_threshold")
    ratio = ev.get("score_ratio")
    if ratio is None:
        return None
    if ratio >= 1.0:
        return "above"
    return "near" if ratio >= NEAR_RATIO else "below"


def build_evidence(ev: dict) -> list[Evidence]:
    """정답 근거를 판별 7항목으로 옮긴다. 값이 없으면 usable=False 로 둔다."""
    patch_ref = ev.get("nearest_patch_ref")
    return [
        Evidence(1, "defect_visible", _verdict(ev.get("defect_visible")),
                 "vlm", ev.get("defect_visible") is not None),
        Evidence(2, "quality_within_baseline", ev.get("quality_within_baseline"),
                 "compute", ev.get("quality_within_baseline") is not None),
        Evidence(3, "score_position", _position(ev),
                 "lookup", _position(ev) is not None),
        Evidence(4, "nearest_patch", patch_ref, "trace", patch_ref is not None),
        Evidence(5, "nearest_patch_is_defect", _verdict(ev.get("nearest_patch_is_defect")),
                 "vlm", ev.get("nearest_patch_is_defect") is not None),
        Evidence(6, "coverage_present", ev.get("coverage_present"),
                 "lookup", ev.get("coverage_present") is not None),
        Evidence(7, "criteria_verdict", ev.get("criteria_verdict"),
                 "lookup", ev.get("criteria_verdict") is not None),
    ]


def main() -> None:
    data = yaml.safe_load((REPO / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])

    seen: set[str] = set()
    rows = []
    for s in scenarios:
        sid = s.get("id", "?")
        if sid in seen:
            continue  # 파일 끝의 예시 템플릿. 앞의 실제 시나리오를 남긴다
        seen.add(sid)

        gt = s.get("ground_truth") or {}
        ev = gt.get("evidence") or {}
        result = decide(build_evidence(ev))

        forbidden = set(gt.get("forbidden_actions") or [])
        rows.append({
            "id": sid,
            "expected": gt.get("cause"),
            "got": result.cause,
            "cause_ok": result.cause == gt.get("cause"),
            "rebuild_expected": gt.get("requires_bank_rebuild"),
            "rebuild_got": result.requires_bank_rebuild,
            "confidence": result.confidence,
            "violated": sorted(forbidden & set(result.recommended_actions)),
            "blocked": bool(result.blocking_reason),
            "assumed": "TODO(" in yaml.dump(s, allow_unicode=True),
            "ratio": ev.get("score_ratio"),
            "position": ev.get("score_vs_threshold"),
        })

    mode = "score_ratio 에서 계산" if USE_COMPUTED_POSITION else "시나리오 라벨 사용"
    print(f"decide() 규칙 정확도 — 근거는 정답 그대로, 스윕 없음, 판별 3번은 {mode}\n")
    head = f"{'id':<12} {'정답':<19} {'판정':<19} {'맞음':<5} {'재구성':<7} {'확신':<7} 금지위반"
    print(head)
    print("-" * len(head))
    for r in rows:
        got = r["got"] or ("보류" if r["blocked"] else "없음")
        rb = "일치" if r["rebuild_got"] == r["rebuild_expected"] else "어긋남"
        mark = "O" if r["cause_ok"] else "X"
        viol = ", ".join(r["violated"]) or "—"
        tag = " *" if r["assumed"] else ""
        print(f"{r['id']+tag:<12} {str(r['expected']):<19} {got:<19} {mark:<5} "
              f"{rb:<7} {r['confidence']:<7} {viol}")

    total = len(rows)
    ok = sum(1 for r in rows if r["cause_ok"])
    rb_ok = sum(1 for r in rows if r["rebuild_got"] == r["rebuild_expected"])
    viol = [r for r in rows if r["violated"]]
    field = [r for r in rows if not r["assumed"]]
    field_ok = sum(1 for r in field if r["cause_ok"])

    print(f"\n원인 일치        {ok}/{total}")
    print(f"  현장 근거      {field_ok}/{len(field)}")
    print(f"  기술적 가정(*) {ok - field_ok}/{total - len(field)}")
    print(f"재구성 여부 일치 {rb_ok}/{total}")
    print(f"금지 행위 위반   {len(viol)}건 " + ", ".join(r["id"] for r in viol))

    wrong = [r for r in rows if not r["cause_ok"]]
    if wrong:
        print("\n틀린 것")
        for r in wrong:
            got = r["got"] or ("보류" if r["blocked"] else "없음")
            print(f"  {r['id']}  {r['expected']} → {got}   "
                  f"(score_vs_threshold={r['position']}, score_ratio={r['ratio']})")


if __name__ == "__main__":
    main()
