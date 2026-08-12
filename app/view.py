"""화면 렌더링.

템플릿 파일을 따로 두지 않고 한곳에 모았다. 시연용 화면 하나뿐이라
파일을 나누면 오히려 찾기 번거롭다.

색은 판정 성격에 따라 고정한다. 진행됨·차단됨·건너뜀이 한눈에 갈려야
"어디서 멈췄는가"가 바로 보인다.
"""

from __future__ import annotations

from html import escape

from app.pipeline import RunOutcome, Stage

STATUS_LABEL = {"done": "진행", "blocked": "중단", "skipped": "건드리지 않음", "pending": "대기"}

STYLE = """
:root{
  --bg:#eef1f3; --panel:#fff; --panel2:#f6f8f9; --ink:#161d22; --ink2:#4a5860;
  --ink3:#77878f; --rule:#d3dbdf; --rule2:#e5eaed; --accent:#0d6f78;
  --ok:#2c7148; --ok-bg:#dcebe2; --stop:#9c3527; --stop-bg:#f3ded9;
  --skip:#9a6410; --skip-bg:#f2e7d0;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0f1417;--panel:#161d21;--panel2:#1b2429;--ink:#e2e9ec;--ink2:#a3b1b8;
    --ink3:#74858d;--rule:#2a353b;--rule2:#222c31;--accent:#4fb3bd;
    --ok:#6cbb8c;--ok-bg:#173023;--stop:#e08a7c;--stop-bg:#3a1f1a;
    --skip:#d3a256;--skip-bg:#33280f;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:"Pretendard","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;
  font-size:15px;line-height:1.6}
.wrap{max-width:1080px;margin:0 auto;padding:36px 22px 80px;display:flex;
  flex-direction:column;gap:26px}
header h1{margin:0 0 6px;font-size:25px;letter-spacing:-.02em}
header p{margin:0;color:var(--ink2);font-size:14.5px;max-width:70ch}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.15em;
  text-transform:uppercase;color:var(--accent);margin-bottom:8px}
form{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
  padding:18px;display:flex;flex-direction:column;gap:12px}
label{font-size:12.5px;font-family:var(--mono);letter-spacing:.06em;
  text-transform:uppercase;color:var(--ink3)}
textarea,select{width:100%;font:inherit;color:var(--ink);background:var(--panel2);
  border:1px solid var(--rule);border-radius:5px;padding:10px}
textarea{min-height:70px;resize:vertical}
.controls{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end}
.controls>div{flex:1;min-width:230px;display:flex;flex-direction:column;gap:5px}
button{background:var(--accent);color:#fff;border:0;border-radius:5px;
  padding:11px 22px;font:inherit;font-weight:600;cursor:pointer}
button:hover{filter:brightness(1.08)}
.hint{font-size:12.5px;color:var(--ink3)}
.flow{display:flex;flex-direction:column;gap:12px}
.stage{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
  padding:16px 18px;display:flex;flex-direction:column;gap:10px;
  border-left:3px solid var(--rule)}
.stage.done{border-left-color:var(--ok)}
.stage.blocked{border-left-color:var(--stop)}
.stage.skipped{border-left-color:var(--skip)}
.stage-head{display:flex;align-items:baseline;justify-content:space-between;
  gap:12px;flex-wrap:wrap}
.stage-title{font-size:16px;font-weight:680}
.chip{font-family:var(--mono);font-size:10.5px;letter-spacing:.08em;
  padding:2px 9px;border-radius:3px;font-weight:600;white-space:nowrap}
.chip.done{color:var(--ok);background:var(--ok-bg)}
.chip.blocked{color:var(--stop);background:var(--stop-bg)}
.chip.skipped{color:var(--skip);background:var(--skip-bg)}
.headline{font-size:15.5px;font-weight:620;color:var(--accent)}
.detail{color:var(--ink2);font-size:14.5px;margin:0}
table{width:100%;border-collapse:collapse;font-size:13.5px}
td{padding:5px 10px 5px 0;border-bottom:1px solid var(--rule2);vertical-align:top}
td:first-child{font-family:var(--mono);color:var(--ink3);white-space:nowrap;width:34%}
.note{font-size:12.5px;color:var(--ink3);border-top:1px dashed var(--rule2);
  padding-top:8px;margin:0}
.doc{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:18px}
.doc h2{margin:0 0 10px;font-size:15px;font-family:var(--mono);letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink3);font-weight:600}
pre{background:var(--panel2);border:1px solid var(--rule2);border-radius:6px;
  padding:14px;overflow-x:auto;font-family:var(--mono);font-size:12.5px;
  line-height:1.55;margin:0;max-height:460px}
.banner{background:var(--panel2);border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:6px;padding:12px 15px;font-size:13.5px;color:var(--ink2)}
a{color:var(--accent)}
"""


def _stage_html(stage: Stage) -> str:
    rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in stage.rows
    )
    return f"""
    <section class="stage {stage.status}">
      <div class="stage-head">
        <span class="stage-title">{escape(stage.title)}</span>
        <span class="chip {stage.status}">{STATUS_LABEL.get(stage.status, stage.status)}</span>
      </div>
      {f'<div class="headline">{escape(stage.headline)}</div>' if stage.headline else ''}
      {f'<p class="detail">{escape(stage.detail)}</p>' if stage.detail else ''}
      {f'<table>{rows}</table>' if rows else ''}
      {f'<p class="note">{escape(stage.note)}</p>' if stage.note else ''}
    </section>
    """


def render_page(outcome: RunOutcome | None, issue_text: str, patch_verdict: str = "defect") -> str:
    options = [
        ("defect", "결함이다 → 뱅크 오염"),
        ("normal", "진짜 정상품이다 → 정상 분포 중첩"),
        ("unknown", "판단 불가"),
        ("ask_model", "모델에게 묻기 (미연결 시 판정 보류)"),
    ]
    select = "".join(
        f'<option value="{v}"{" selected" if v == patch_verdict else ""}>{escape(t)}</option>'
        for v, t in options
    )

    body = ""
    if outcome:
        body += '<div class="flow">' + "".join(_stage_html(s) for s in outcome.stages) + "</div>"
        if outcome.approval_markdown:
            body += f"""
            <div class="doc">
              <h2>승인 요청 문서</h2>
              <pre>{escape(outcome.approval_markdown)}</pre>
              <p class="note">원문: <a href="/approval">/approval</a></p>
            </div>
            """

    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>검사 AI 자율 운영 에이전트</title><style>{STYLE}</style></head>
<body><div class="wrap">
  <header>
    <div class="eyebrow">Self-Healing Vision Ops · 시연</div>
    <h1>못 잡는 불량을 접수하면 승인 요청까지</h1>
    <p>자연어로 이슈를 올리면 진단·데이터 선별·뱅크 재구성·성능 검증을 거쳐
       승인 요청 문서까지 만듭니다. <strong>배포는 실행되지 않습니다.</strong></p>
  </header>

  <div class="banner">
    조회 계층은 목이고 데이터는 합성 이미지입니다. 성능 수치가 아니라
    <strong>경로가 이어지는지</strong>를 보는 화면입니다.
  </div>

  <form method="post" action="/run">
    <div>
      <label for="issue">이슈 내용</label>
      <textarea id="issue" name="issue_text">{escape(issue_text)}</textarea>
    </div>
    <div class="controls">
      <div>
        <label for="patch">판별 5번 — 최근접 패치가 무엇인가</label>
        <select id="patch" name="patch_verdict">{select}</select>
        <span class="hint">이 값 하나로 조치가 정반대로 갈립니다.</span>
      </div>
      <button type="submit">접수하고 실행</button>
    </div>
  </form>

  {body}
</div></body></html>"""
