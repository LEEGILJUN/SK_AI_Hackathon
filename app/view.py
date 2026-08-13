"""화면 렌더링.

템플릿 파일을 따로 두지 않고 한곳에 모았다. 시연용 화면 하나뿐이라
파일을 나누면 오히려 찾기 번거롭다.

색은 판정 성격에 따라 고정한다. 진행됨·차단됨·건너뜀이 한눈에 갈려야
"어디서 멈췄는가"가 바로 보인다.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

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
input{width:100%;font:inherit;color:var(--ink);background:var(--panel2);
  border:1px solid var(--rule);border-radius:5px;padding:9px 10px}
.supplement{border-top:1px dashed var(--rule2);padding-top:12px;margin-top:2px}
.supplement summary{font-size:13px;color:var(--ink3);cursor:pointer;padding:2px 0}
.supplement[open] summary{color:var(--ink2);margin-bottom:10px}
.supplement .controls{margin-top:10px}
.ask{margin:0 0 8px;color:var(--stop);font-size:14px;font-weight:600}
a{color:var(--accent)}

/* ── 라인 시뮬레이터 ───────────────────────────────────────────────── */
.sim{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
  padding:18px;display:flex;flex-direction:column;gap:14px}
.sim-head{display:flex;justify-content:space-between;align-items:baseline;
  gap:12px;flex-wrap:wrap}
.sim-title{font-size:16px;font-weight:680}
.sim-state{font-family:var(--mono);font-size:12px;letter-spacing:.06em;
  color:var(--accent);font-weight:600}
.belt{position:relative;background:var(--panel2);border:1px solid var(--rule2);
  border-radius:6px;height:150px;overflow:hidden}
.belt::after{content:"";position:absolute;left:0;right:0;bottom:26px;height:2px;
  background:repeating-linear-gradient(90deg,var(--rule) 0 14px,transparent 14px 28px)}
.piece{position:absolute;bottom:34px;width:74px;transition:left .5s linear;
  display:flex;flex-direction:column;align-items:center;gap:4px}
.piece img{width:64px;height:64px;object-fit:cover;border-radius:4px;
  border:2px solid var(--rule);background:var(--panel)}
.piece.defect img{border-color:var(--stop)}
.piece.pass img{border-color:var(--ok)}
.piece .tag{font-family:var(--mono);font-size:10px;padding:1px 6px;border-radius:3px;
  white-space:nowrap}
.piece.defect .tag{color:var(--stop);background:var(--stop-bg)}
.piece.pass .tag{color:var(--ok);background:var(--ok-bg)}
.scanner{position:absolute;top:0;bottom:24px;left:50%;width:2px;
  background:var(--accent);opacity:.55}
.scanner::before{content:"검사 지점";position:absolute;top:4px;left:6px;
  font-family:var(--mono);font-size:10px;color:var(--accent);letter-spacing:.1em;
  white-space:nowrap}
.bar{height:5px;background:var(--rule2);border-radius:3px;overflow:hidden}
.bar>span{display:block;height:100%;width:0;background:var(--accent);
  transition:width .4s ease}
.tally{display:flex;gap:10px;flex-wrap:wrap}
.tally div{flex:1;min-width:120px;background:var(--panel2);border:1px solid var(--rule2);
  border-radius:6px;padding:9px 12px}
.tally b{display:block;font-size:20px;font-variant-numeric:tabular-nums;line-height:1.2}
.tally span{font-size:11.5px;color:var(--ink3);font-family:var(--mono);letter-spacing:.05em}
.tally .good b{color:var(--ok)}
.tally .bad b{color:var(--stop)}
@media (prefers-reduced-motion:reduce){.piece{transition:none}}
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


def _simulator_html(outcome: RunOutcome) -> str:
    """가상 라인 시뮬레이터 — 새 코어셋이 실제로 무엇을 잡는지 흘려 보여준다.

    **지어낸 애니메이션이 아니다.** 흘러가는 판정은 전부 `shadow_compare` 가
    실제로 낸 값이고, 화면은 그것을 한 장씩 재생할 뿐이다. 그래서 마지막에
    쌓인 숫자가 게이트 지표와 정확히 같다.

    두 뱅크를 나란히 보여주는 것이 요점이다. 신규 뱅크만 돌려서는 "좋아졌다"를
    말할 수 없다 — 양산 데이터에는 정답이 없으므로 **같은 이미지에 두 모델을
    돌려 갈리는 것을 보는 것**이 섀도 평가다.
    """
    shadow = outcome.shadow
    if shadow is None or not shadow.cases:
        return ""

    cases = json.dumps(
        [
            {
                "src": f"/image/{c.image}",
                "name": Path(c.image).name,
                "before": c.current_verdict,
                "after": c.candidate_verdict,
                "beforeScore": round(c.current_score, 3),
                "afterScore": round(c.candidate_score, 3),
                "agreed": c.agreed,
            }
            for c in shadow.cases
        ],
        ensure_ascii=False,
    )
    return f"""
    <div class="sim">
      <div class="sim-head">
        <span class="sim-title">코어셋 검증 — 가상 라인</span>
        <span class="sim-state" id="sim-state">대기</span>
      </div>
      <p class="detail">
        신규 코어셋 <code>{escape(shadow.candidate_version)}</code> 을 현행
        <code>{escape(shadow.current_version)}</code> 과 나란히 돌립니다.
        <strong>신규 뱅크는 실제 판정에 쓰이지 않습니다.</strong>
      </p>
      <div class="belt" id="belt"><div class="scanner"></div></div>
      <div class="bar"><span id="sim-bar"></span></div>
      <div class="tally">
        <div><b id="t-total">0</b><span>검사 완료</span></div>
        <div class="good"><b id="t-caught">0</b><span>새로 잡음</span></div>
        <div class="bad"><b id="t-lost">0</b><span>새로 놓침</span></div>
        <div><b id="t-same">0</b><span>판정 동일</span></div>
      </div>
      <p class="note" id="sim-note">
        흘러가는 판정은 전부 섀도 비교가 실제로 낸 값입니다. 화면은 그것을
        한 장씩 재생합니다.
      </p>
    </div>
    <script>
    (function() {{
      const cases = {cases};
      const belt = document.getElementById("belt");
      const state = document.getElementById("sim-state");
      const bar = document.getElementById("sim-bar");
      const out = {{
        total: document.getElementById("t-total"),
        caught: document.getElementById("t-caught"),
        lost: document.getElementById("t-lost"),
        same: document.getElementById("t-same"),
      }};
      let done = 0, caught = 0, lost = 0, same = 0, i = 0;

      function release() {{
        if (i >= cases.length) {{
          state.textContent = "검증 완료 — 사람 승인 대기";
          document.getElementById("sim-note").textContent =
            "판정이 갈린 " + (caught + lost) + "장만 사람이 확인하면 됩니다. " +
            "나머지 " + same + "장은 두 뱅크가 같게 판정했습니다.";
          return;
        }}
        const c = cases[i++];
        state.textContent = "코어셋 검증 중입니다 \\u2014 " + i + "/" + cases.length;

        const piece = document.createElement("div");
        piece.className = "piece " + (c.after === "defect" ? "defect" : "pass");
        piece.style.left = "-80px";
        piece.innerHTML =
          '<img src="' + c.src + '" alt="' + c.name + '">' +
          '<span class="tag">' + (c.after === "defect" ? "불량" : "양품") +
          " " + c.afterScore + "</span>";
        belt.appendChild(piece);

        requestAnimationFrame(function() {{ piece.style.left = "45%"; }});

        setTimeout(function() {{
          done++;
          if (c.agreed) same++;
          else if (c.before === "pass" && c.after === "defect") caught++;
          else lost++;
          out.total.textContent = done;
          out.caught.textContent = caught;
          out.lost.textContent = lost;
          out.same.textContent = same;
          bar.style.width = (done / cases.length * 100) + "%";
          piece.style.left = "108%";
          setTimeout(function() {{ piece.remove(); }}, 600);
        }}, 620);

        setTimeout(release, 780);
      }}

      const start = function() {{ setTimeout(release, 300); }};
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
        // 애니메이션을 끈 환경에서는 결과만 즉시 채운다.
        cases.forEach(function(c) {{
          done++;
          if (c.agreed) same++;
          else if (c.before === "pass" && c.after === "defect") caught++;
          else lost++;
        }});
        out.total.textContent = done; out.caught.textContent = caught;
        out.lost.textContent = lost; out.same.textContent = same;
        bar.style.width = "100%";
        state.textContent = "검증 완료 — 사람 승인 대기";
      }} else {{
        start();
      }}
    }})();
    </script>
    """


def _driver_html(outcome: RunOutcome) -> str:
    """도구 순서를 누가 정했는가.

    모델이 안 붙어 있는데 화면이 아무 말도 하지 않으면 "에이전트가 판단한 것"
    처럼 보인다. 시연에서 가장 오해받기 쉬운 지점이라 위에 못 박아 둔다.
    """
    by_model = outcome.driver == "model"
    label = "언어 모델이 도구 순서를 정했습니다" if by_model else "고정 순서로 실행했습니다"
    trace = "".join(
        f'<tr><td>{i}. {escape(name)}</td><td>{escape(status)}</td></tr>'
        for i, (name, status) in enumerate(outcome.tool_trace, start=1)
    )
    stopped = outcome.agent_run.stopped_reason if outcome.agent_run else ""
    return f"""
    <div class="banner{'' if by_model else ' warn'}">
      <strong>{escape(label)}</strong> — {escape(outcome.driver_note)}
      {f'<table>{trace}</table>' if trace else ''}
      {f'<p class="note">{escape(stopped)}</p>' if stopped else ''}
    </div>
    """


def render_page(outcome: RunOutcome | None, issue_text: str, patch_verdict: str = "defect",
                context: dict[str, str] | None = None) -> str:
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
    # ── 입력은 프롬프트 하나다 ──────────────────────────────────────────
    #
    # 라인·품목·제품명 칸을 처음부터 띄워 놓으면 사람이 그것부터 채우게 되고,
    # 그러면 언어 모델이 원문에서 뽑을 일이 없어져 자연어 입력이 장식이 된다.
    # 그래서 **기본 화면은 이슈 원문과 실행 단추뿐**이다.
    #
    # 칸은 인테이크가 되물었을 때만 열린다. 그것이 원래 설계된 동작이다 —
    # 정보가 부족하면 추측하지 않고 무엇이 필요한지 되묻는다. 모델이 붙어
    # 있으면 원문에서 뽑으므로 이 칸은 끝까지 안 나온다.
    asked = bool(outcome and outcome.intake and outcome.intake.verdict == "need_more_info")
    ctx = context or {}
    if outcome and outcome.intake:
        report = outcome.intake.report
        ctx = {k: (ctx.get(k) or getattr(report, k, "") or "")
               for k in ("line", "object_name", "defect_type", "product_id")}

    fields = "".join(
        f"""<div>
          <label for="{key}">{escape(label)}</label>
          <input id="{key}" name="{key}" value="{escape(ctx.get(key, '') or '')}"
                 placeholder="{escape(hint)}">
        </div>"""
        for key, label, hint in (
            ("line", "라인", "예: line_02"),
            ("object_name", "품목", "예: capsules"),
            ("defect_type", "결함 유형", "예: dent"),
            ("product_id", "제품명", "예: CAPSULES-02-defect_002"),
        )
    )

    question = outcome.intake.question if asked else ""
    supplement = f"""
    <details class="supplement"{' open' if asked else ''}>
      <summary>{escape('인테이크가 되물었습니다 — 값을 채워 주세요' if asked
                       else '보충 입력 (모델이 원문에서 못 뽑을 때만 필요)')}</summary>
      {f'<p class="ask">{escape(question)}</p>' if question else ''}
      <p class="hint">
        언어 모델이 붙어 있으면 <strong>이슈 원문에서 직접 뽑습니다.</strong>
        여기 채운 값은 모델이 못 뽑은 자리에만 들어갑니다.
      </p>
      <div class="controls">{fields}</div>
    </details>
    """

    body = ""
    if outcome:
        body += _driver_html(outcome)
        # 시뮬레이터는 섀도 단계 바로 앞에 끼운다. 숫자만 적힌 표보다
        # 무엇이 어떻게 갈렸는지가 먼저 보여야 한다.
        stages_html = []
        for stage in outcome.stages:
            if stage.key == "shadow":
                stages_html.append(_simulator_html(outcome))
            stages_html.append(_stage_html(stage))
        body += '<div class="flow">' + "".join(stages_html) + "</div>"
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
      <label for="issue">현장에서 올라온 이슈</label>
      <textarea id="issue" name="issue_text"
                placeholder="예) 2라인 캡슐 표면 찍힘이 며칠째 계속 빠집니다. 제품 CAPSULES-02-defect_002 건입니다."
      >{escape(issue_text)}</textarea>
      <span class="hint">
        이것만 적으면 됩니다. 라인·품목·제품명은 <strong>언어 모델이 여기서 뽑습니다.</strong>
      </span>
    </div>
    <div class="controls">
      <button type="submit">접수하고 실행</button>
      <div>
        <label for="patch">시연 조정 · 판별 5번</label>
        <select id="patch" name="patch_verdict">{select}</select>
        <span class="hint">이 값 하나로 조치가 정반대로 갈립니다.</span>
      </div>
    </div>
    {supplement}
  </form>

  {body}
</div></body></html>"""
