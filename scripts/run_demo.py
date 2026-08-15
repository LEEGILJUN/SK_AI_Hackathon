"""웹 화면과 **똑같은 것**을 터미널에서 돌린다.

    .venv/bin/python scripts/run_demo.py
    .venv/bin/python scripts/run_demo.py --patch-override defect
    .venv/bin/python scripts/run_demo.py --item line_02/pcb2

`app/main.py` 가 부르는 `run_pipeline()` 을 그대로 부르고, 화면에 뜨는 단계와
줄을 그대로 찍는다. **화면에 새로 붙은 표시가 여기에도 나온다** — 각자
스크립트를 조립하면 한쪽에만 보이는 것이 생겨서, 무엇을 보고 있는지가
장비마다 달라진다.

`demo_diagnose.py` 와 다르다. 그쪽은 도구를 손으로 순서대로 부르며 각 단계의
내부 값을 보여 주는 학습용이고, 이쪽은 **실제 실행 경로**다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.adapters import build_adapters  # noqa: E402
from app.pipeline import DEMO_ITEMS, DemoFactory, default_issue, run_pipeline  # noqa: E402

RULE = "─" * 72


def main() -> int:
    parser = argparse.ArgumentParser(description="전 구간 한 번 실행")
    parser.add_argument("--item", default="", help="line_01/pcb1 형식. 비우면 기본 품목")
    parser.add_argument("--issue", default="", help="이슈 원문. 비우면 기본 문안")
    parser.add_argument(
        "--patch-override", default=None, choices=["defect", "normal"],
        help="판별 5번을 손으로 지정. **기본은 지정하지 않는다** — 모델에 묻는다",
    )
    parser.add_argument("--threshold", type=float, default=2.20)
    parser.add_argument(
        "--no-form", action="store_true",
        help="양식 값을 비운다. 언어 모델이 원문에서 다 뽑는지 보려면 이것",
    )
    args = parser.parse_args()

    started = time.time()
    factory = DemoFactory()
    print(f"공장 준비 {time.time() - started:.1f}초")

    line, object_name = DEMO_ITEMS[0][0], DEMO_ITEMS[0][1]
    if args.item:
        line, object_name = args.item.split("/")

    issue = args.issue or default_issue(factory)
    llm, vlm = build_adapters()
    print(f"언어 모델 {llm.describe()}")
    print(f"시각 모델 {vlm.describe()}\n")

    ran = time.time()
    outcome = run_pipeline(
        factory, issue,
        patch_override=args.patch_override,
        adapters=(llm, vlm),
        threshold=args.threshold,
        # 웹 화면에는 이슈 원문 옆에 양식이 있다. 여기도 같게 채운다 —
        # **언어 모델이 원문에서 뽑으면 그 값이 이기고**, 못 뽑았을 때만
        # 이것이 쓰인다. `--no-form` 으로 비우면 모델만으로 도는지 볼 수 있다.
        context={} if args.no_form else {
            "line": line, "object_name": object_name,
            "defect_type": "스크래치", "product_id": factory.reported_product,
        },
    )
    elapsed = time.time() - ran

    print(f"{RULE}\n이슈 원문\n{RULE}\n{outcome.issue_text}\n")
    print(f"{RULE}\n도구 호출 — {outcome.driver}\n{RULE}")
    print(outcome.driver_note)
    for index, result in enumerate(outcome.agent_run.tool_results, 1):
        mark = "성공" if result.ok else f"실패 — {result.error}"
        print(f"  {index}. {result.name:<18} {mark}")
    if outcome.agent_run.stopped_reason:
        print(f"  멈춘 이유: {outcome.agent_run.stopped_reason}")

    print(f"\n{RULE}\n단계\n{RULE}")
    for stage in outcome.stages:
        print(f"\n[{stage.status}] {stage.title} — {stage.headline}")
        if stage.detail:
            print(f"  {stage.detail}")
        for label, value in stage.rows:
            print(f"    {label:<16} {value}")
        if stage.note:
            print(f"  ※ {stage.note}")

    print(f"\n{RULE}\n판별 7항목\n{RULE}")
    diagnosis = outcome.diagnosis
    if diagnosis is None:
        print("  진단에 도달하지 못했습니다.")
    else:
        for item in diagnosis.evidence:
            usable = "○" if item.usable else "×"
            print(f"  {item.item_no} {item.name:<24} {usable} {item.value}")
            if item.detail:
                print(f"      {item.detail}")
        print(f"\n  원인      {diagnosis.cause}")
        print(f"  재구성필요 {diagnosis.requires_bank_rebuild}")
        print(f"  확신      {diagnosis.confidence}")
        if diagnosis.blocking_reason:
            print(f"  보류 이유  {diagnosis.blocking_reason}")
        if diagnosis.reasoning:
            print(f"  근거      {diagnosis.reasoning}")

    print(f"\n{RULE}\n조회\n{RULE}")
    for call in outcome.retrievals:
        arguments = " · ".join(f"{k}={v}" for k, v in call["arguments"].items())
        print(f"  {call['name']:<22} [{call['kind']}] {arguments}")

    print(f"\n{RULE}")
    print(f"진단 이미지  {outcome.query_image or '—'}")
    print(f"뱅크         {outcome.bank_version or '—'}")
    print(f"격자         {outcome.grid}")
    print(f"승인 요청    {'생성됨' if outcome.approval_markdown else '없음'}")
    print(f"실행 시간    {elapsed:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
