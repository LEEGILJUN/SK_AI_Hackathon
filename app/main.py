"""이슈 접수 웹 인터페이스 (작업 11).

현장에서 자연어로 이슈를 올리면 접수부터 승인 요청까지가 한 화면에서 진행되는
것을 보여준다. 시연용이며 운영 화면이 아니다.

실행:
    .venv/bin/python -m uvicorn app.main:app --port 8000
    또는
    .venv/bin/python app/main.py

첫 요청에서 합성 가상 공장을 만들고 뱅크를 구성하므로 몇 초 걸린다.
그 뒤로는 즉시 응답한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, Form  # noqa: E402
from fastapi.responses import HTMLResponse, PlainTextResponse  # noqa: E402

from app.pipeline import DemoFactory, RunOutcome, default_issue, run_pipeline  # noqa: E402
from app.view import render_page  # noqa: E402

app = FastAPI(title="검사 AI 자율 운영 에이전트")

_factory: DemoFactory | None = None
_last: RunOutcome | None = None


def factory() -> DemoFactory:
    """가상 공장은 한 번만 만든다. 뱅크 구성이 매 요청마다 돌면 느리다."""
    global _factory
    if _factory is None:
        _factory = DemoFactory()
    return _factory


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """첫 화면.

    이슈 원문에 제품명이 들어 있다. 언어 모델이 뽑을 것이 있어야 MES 조회가
    의미를 가진다. **모델이 붙어 있으면 아래 칸을 비워 두면 되고**, 안 붙어
    있으면 채워야 한다 — 인테이크가 추측으로 채우지 않기 때문이다. 그래서
    Mac 처럼 모델이 없는 환경에서는 미리 채워 둔다.
    """
    f = factory()
    prefill = {"line": "line_02", "object_name": "capsules",
               "defect_type": "dent", "product_id": f.reported_product}
    return render_page(None, default_issue(f), context=prefill)


@app.post("/run", response_class=HTMLResponse)
def run(
    issue_text: str = Form(""),
    patch_verdict: str = Form("defect"),
    line: str = Form(""),
    object_name: str = Form(""),
    defect_type: str = Form(""),
    product_id: str = Form(""),
) -> str:
    """이슈를 접수하고 전 구간을 실행한다.

    라인·품목·제품명은 **이슈 원문에서 언어 모델이 뽑는 것이 우선**이고,
    양식은 모델이 못 뽑은 자리만 채운다. 양식이 다 채워져 있으면 추출이 할 일이
    없어져 언어 모델을 쓰는 의미가 사라진다. 모델이 없을 때만 양식이 주 입력이
    되며, 그때도 인테이크는 추측으로 채우지 않고 비면 되묻는다.

    patch_verdict
        판별 5번을 손으로 지정한다. 시연에서 이 값을 바꿔 가며 같은 이미지·
        같은 점수에서 조치가 정반대로 갈리는 것을 보여준다. "ask_model" 이면
        역추적이 가리킨 자리를 잘라 시각 언어 모델에 묻는다.
    """
    global _last
    override = None if patch_verdict == "ask_model" else patch_verdict
    context = {k: v for k, v in
               (("line", line), ("object_name", object_name),
                ("defect_type", defect_type), ("product_id", product_id)) if v}
    issue_text = issue_text or default_issue(factory())
    _last = run_pipeline(
        factory(), issue_text=issue_text, patch_override=override,
        context=context or None,
    )
    return render_page(_last, issue_text, patch_verdict, context or None)


@app.get("/approval", response_class=PlainTextResponse)
def approval() -> str:
    """승인 요청 문서 원문."""
    if _last is None or not _last.approval_markdown:
        return "아직 생성된 승인 요청이 없습니다."
    return _last.approval_markdown


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
