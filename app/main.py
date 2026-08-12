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

from app.pipeline import DEFAULT_ISSUE, DemoFactory, RunOutcome, run_pipeline  # noqa: E402
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
    return render_page(None, DEFAULT_ISSUE)


@app.post("/run", response_class=HTMLResponse)
def run(
    issue_text: str = Form(DEFAULT_ISSUE),
    patch_verdict: str = Form("defect"),
) -> str:
    """이슈를 접수하고 전 구간을 실행한다.

    patch_verdict
        판별 5번을 손으로 지정한다. 시각 언어 모델이 아직 붙지 않았으므로,
        시연에서는 이 값을 바꿔 가며 같은 이미지·같은 점수에서 조치가
        정반대로 갈리는 것을 보여준다.
    """
    global _last
    override = None if patch_verdict == "ask_model" else patch_verdict
    _last = run_pipeline(factory(), issue_text=issue_text, patch_override=override)
    return render_page(_last, issue_text, patch_verdict)


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
