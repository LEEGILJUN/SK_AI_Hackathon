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

import io  # noqa: E402

from fastapi import FastAPI, Form, HTTPException, Response  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse  # noqa: E402

from app.pipeline import (  # noqa: E402
    DEMO_CONFIG,
    DemoFactory,
    RunOutcome,
    default_issue,
    run_pipeline,
)
from inspection.crop import crop_patch  # noqa: E402
from inspection.types import PatchRef  # noqa: E402
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
    """첫 화면 — **입력은 이슈 원문 하나다.**

    라인·품목 칸을 미리 띄워 채워 두면 사람이 그것부터 채우게 되고, 언어 모델이
    원문에서 뽑을 일이 없어져 자연어 입력이 장식이 된다. 그래서 비워 둔다.

    모델이 없으면 추출이 비고 인테이크가 되묻는다. 그것을 감추지 않는다 —
    "정보가 부족하면 추측하지 않고 되묻는다"가 인테이크의 설계이고, 되물었을
    때 비로소 보충 칸이 열린다.
    """
    return render_page(None, default_issue(factory()))


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


@app.get("/image/{relative_path:path}")
def image(relative_path: str) -> FileResponse:
    """가상 공장 이미지 한 장. 라인 시뮬레이터가 가져다 쓴다.

    **공장 루트 밖으로 나가지 못하게 막는다.** 경로를 그대로 이어 붙이면
    `../../etc/passwd` 같은 요청에 파일을 내주게 된다. 실제 경로로 풀어서
    루트 아래인지 확인한 뒤에만 응답한다.
    """
    root = factory().root.resolve()
    try:
        target = (root / relative_path).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="없는 경로입니다.")

    if not target.is_file() or target.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=404, detail="없는 이미지입니다.")
    return FileResponse(target)


@app.get("/crop/{relative_path:path}")
def crop(relative_path: str, row: int, col: int, grid_h: int, grid_w: int,
         margin: int = 12) -> Response:
    """패치 한 칸을 원본에서 잘라 돌려준다 — 역추적 근거를 눈으로 보게.

    "이 미검출 이미지가 뱅크의 저 정상 패치와 가까웠다"를 글로만 적으면
    확인할 방법이 없다. 그 자리를 실제로 잘라 보여주면 사람이 직접 판단할 수
    있고, 그것이 판별 5번이 하는 일이기도 하다.

    좌표 계산은 `inspection.crop` 이 한다. 화면이 따로 계산하면 두 벌이 되고
    한쪽만 고쳐져 엉뚱한 자리를 자르게 된다.
    """
    root = factory().root.resolve()
    try:
        target = (root / relative_path).resolve()
        target.relative_to(root)
    except (ValueError, OSError):
        raise HTTPException(status_code=404, detail="없는 경로입니다.")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="없는 이미지입니다.")
    if grid_h <= 0 or grid_w <= 0 or not (0 <= row < grid_h) or not (0 <= col < grid_w):
        raise HTTPException(status_code=400, detail="격자 좌표가 범위를 벗어났습니다.")

    ref = PatchRef(source_image=relative_path, row=row, col=col,
                   patch_index=row * grid_w + col)
    patch = crop_patch(target, ref, (grid_h, grid_w), DEMO_CONFIG,
                       margin=max(0, min(margin, 256)), enlarge_to=192)

    buffer = io.BytesIO()
    patch.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return "ok"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
