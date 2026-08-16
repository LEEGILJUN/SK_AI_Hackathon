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

from inspection.sweep import FeasibilityVerdict  # noqa: E402
from agents.diagnose import Evidence, decide  # noqa: E402

# 판별 3번을 라벨에서 가져올지 비율에서 계산할지. 운영에서는 계산이다.
USE_COMPUTED_POSITION = "--computed" in sys.argv
NEAR_RATIO = 0.9  # inspection/types.py score_position() 의 기본값


def _visibility(flag: object) -> str:
    """판별 1번 — 시나리오의 참/거짓을 판정 단어로.

    **판별 5번과 어휘가 다르다.** 1번은 "이 사진에 보이는가"이고 5번은
    "그 패치가 무엇인가"다. 전에는 둘 다 defect/normal 이라 섞여 읽혔다.
    """
    if flag is None:
        return ""
    return "visible" if flag else "not_visible"


def _patch(flag: object) -> str:
    """판별 5번 — 뱅크의 그 패치가 잘못 들어간 결함인가 진짜 정상품인가."""
    if flag is None:
        return ""
    return "defect" if flag else "genuine_normal"


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
        Evidence(1, "defect_visible", _visibility(ev.get("defect_visible")),
                 "vlm", ev.get("defect_visible") is not None),
        Evidence(2, "quality_within_baseline", ev.get("quality_within_baseline"),
                 "compute", ev.get("quality_within_baseline") is not None),
        Evidence(3, "score_position", _position(ev),
                 "lookup", _position(ev) is not None),
        Evidence(4, "nearest_patch", patch_ref, "trace", patch_ref is not None),
        Evidence(5, "nearest_patch_is_defect", _patch(ev.get("nearest_patch_is_defect")),
                 "vlm", ev.get("nearest_patch_is_defect") is not None),
        Evidence(6, "coverage_present", ev.get("coverage_present"),
                 "lookup", ev.get("coverage_present") is not None),
        Evidence(7, "criteria_verdict", ev.get("criteria_verdict"),
                 "lookup", ev.get("criteria_verdict") is not None),
    ]


def build_sweep(ev: dict) -> "FeasibilityVerdict | None":
    """시나리오의 `threshold_sweep` 을 진단이 받는 형태로.

    **없으면 None 이다.** 스윕이 없는 시나리오까지 지어내면 측정이 아니라
    연출이 된다.

    두 원인의 정의가 곧 이 값이다 — 임계값 문제는 "내리면 해결된다"이고
    정상 분포 중첩은 "내려도 과검만 는다"이다. `score_ratio` 로는 두 구간이
    0.93~0.97 에서 겹쳐 원리적으로 못 가른다.
    """
    sweep = ev.get("threshold_sweep")
    if not sweep:
        return None
    fpr = sweep.get("resulting_fpr")
    return FeasibilityVerdict(
        achievable=bool(sweep.get("achievable")),
        target_detection=1.0,
        max_acceptable_fpr=0.05,
        required_threshold=None,
        resulting_fpr=fpr,
        resulting_detection=1.0 if sweep.get("achievable") else None,
        auroc=0.0,
        reason=(
            f"임계값 조정으로 목표 검출률에 "
            f"{'닿는다' if sweep.get('achievable') else '닿지 못한다'}"
            + (f" (그때 과검률 {fpr:.0%})" if fpr is not None else "")
        ),
    )


def main() -> None:
    data = yaml.safe_load((REPO / "data" / "scenarios.yaml").read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])

    seen: set[str] = set()
    rows = []
    skipped_example = 0
    for s in scenarios:
        sid = s.get("id", "?")
        if sid in seen:
            continue  # 같은 id 가 두 번. 앞의 것을 남긴다
        seen.add(sid)
        # ── 파일 끝의 스키마 예시는 채점 대상이 아니다 ──────────────────
        #
        # 도메인 담당이 "예시 1건 — 아래 형식을 그대로 복제해 나머지를 채웁니다"
        # 라고 적어 둔 골격이다. 대상 라인·품목이 공장 구성에 없어
        # `data/build_factory.py` 도 걸러 낸다.
        #
        # **전에는 실제 시나리오와 id 가 겹쳐서 위 중복 검사에 걸렸다.**
        # 겹친 id 를 떼어 내자 거름망이 풀려 채점 대상이 24 → 25 로 늘었고,
        # 정확도가 19/24(79%)에서 20/25(80%)로 **좋아진 것처럼 보였다.**
        # 목표가 80% 라 그대로 두면 "달성" 으로 읽힌다.
        if sid.startswith("SC-TEMPLATE"):
            skipped_example += 1
            continue

        gt = s.get("ground_truth") or {}
        ev = gt.get("evidence") or {}
        result = decide(build_evidence(ev), sweep=build_sweep(ev))

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
    print(f"decide() 규칙 정확도 — 근거는 정답 그대로, 판별 3번은 {mode}\n")
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

    if skipped_example:
        print(f"\n스키마 예시 {skipped_example}건은 채점에서 뺐습니다 "
              f"(도메인 담당이 형식 견본으로 둔 골격).")
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
