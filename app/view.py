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
from urllib.parse import quote, urlencode

from agents.ontology import CAUSES, action_label, cause_names
from app.pipeline import RunOutcome, Stage
from lookup.base import RETRIEVAL_LABEL

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
/* 그대로 실측으로 읽으면 안 되는 상태 — 고정 순서 재생, 합성 데이터. */
.banner.warn{border-left-color:var(--skip)}
.banner code{font-family:var(--mono);font-size:12.5px;background:var(--panel);
  border:1px solid var(--rule2);border-radius:3px;padding:1px 5px}
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

/* ── 진단 근거 시각화 ──────────────────────────────────────────────── */
.evidence{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
  padding:18px;display:flex;flex-direction:column;gap:14px}
.ev-head{display:flex;justify-content:space-between;align-items:baseline;
  gap:12px;flex-wrap:wrap}
.ev-grid{display:flex;gap:20px;flex-wrap:wrap}
.ev-grid>div{flex:1;min-width:250px;display:flex;flex-direction:column;gap:8px}
.ev-label{font-family:var(--mono);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink3)}
.heat{display:grid;gap:3px;background:var(--panel2);padding:8px;border-radius:6px;
  border:1px solid var(--rule2);aspect-ratio:1;max-width:340px}
.heat i{background:var(--stop);border-radius:2px;min-height:12px;display:block}
/* 역추적이 지목한 칸. 이 한 칸이 진단의 출발점이라 눈에 먼저 들어와야 한다. */
.heat i.hot{outline:3px solid var(--accent);outline-offset:2px;position:relative;z-index:1}
.score{height:12px;background:var(--panel2);border:1px solid var(--rule2);
  border-radius:6px;overflow:hidden}
.score>span{display:block;height:100%;background:linear-gradient(90deg,var(--ok),var(--stop))}
.score-num{margin:0;font-family:var(--mono);font-size:14px}
.score-num em{font-style:normal;font-size:11px;padding:2px 8px;border-radius:3px;
  margin-left:8px;letter-spacing:.06em}
.score-num em.under{color:var(--stop);background:var(--stop-bg)}
.score-num em.over{color:var(--ok);background:var(--ok-bg)}
.pair{display:flex;align-items:center;gap:16px;flex-wrap:wrap;
  background:var(--panel2);border:1px solid var(--rule2);border-radius:6px;padding:14px}
.pair figure{margin:0;display:flex;flex-direction:column;gap:6px;align-items:center}
.pair img{width:120px;height:120px;object-fit:cover;border-radius:5px;
  border:2px solid var(--rule)}
.pair figcaption{font-family:var(--mono);font-size:11px;color:var(--ink2);
  text-align:center;line-height:1.5}
.pair figcaption span{color:var(--ink3)}
.arrow{font-family:var(--mono);font-size:11px;color:var(--ink3);text-align:center;
  letter-spacing:.08em}
.arrow b{display:block;font-size:15px;color:var(--accent);margin-top:3px}

/* ── 조회 방식 ─────────────────────────────────────────────────────── */
.chips{display:flex;gap:8px;flex-wrap:wrap}
.kind{font-family:var(--mono);font-size:11px;padding:2px 9px;border-radius:3px;
  font-weight:600;white-space:nowrap}
.kind.join{color:var(--ok);background:var(--ok-bg)}
.kind.aggregate{color:var(--accent);background:var(--panel2)}
.kind.graph{color:var(--skip);background:var(--skip-bg)}
.kind.llm,.kind.vlm{color:var(--stop);background:var(--stop-bg)}
.kind.unknown{color:var(--ink3);background:var(--panel2)}
table.ret td{font-size:13px}
table.ret td:first-child{width:auto;white-space:nowrap}
.legend{margin:0;padding-left:0;list-style:none;display:flex;flex-direction:column;
  gap:5px;font-size:12.5px;color:var(--ink3)}

/* ── 진단 지식 체계 ────────────────────────────────────────────────── */
table.tax td{font-size:13px;vertical-align:top}
table.tax td:first-child{white-space:nowrap;font-weight:600}
table.tax tr.here td{background:var(--panel2)}
table.tax tr.here td:first-child{color:var(--accent)}
.rb{font-family:var(--mono);font-size:10.5px;padding:2px 7px;border-radius:3px;
  white-space:nowrap}
.rb.yes{color:var(--stop);background:var(--stop-bg)}
.rb.no{color:var(--ok);background:var(--ok-bg)}
.items{font-family:var(--mono);font-size:11px;color:var(--ink3);white-space:nowrap}
.kind.schema{color:var(--accent);background:var(--panel2)}

/* ── 이슈 이력 그래프 ──────────────────────────────────────────────── */
.query-node{background:var(--panel2);border:1px solid var(--accent);border-radius:6px;
  padding:11px 14px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.query-node code{font-size:13px;font-weight:700;color:var(--accent)}
.onodes{display:flex;gap:12px;flex-wrap:wrap}
.onode{flex:1;min-width:290px;background:var(--panel2);border:1px solid var(--rule2);
  border-radius:6px;padding:12px 14px;display:flex;flex-direction:column;gap:8px}
.onode.hit{border-color:var(--stop);background:var(--stop-bg)}
.onode-head{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
.onode-head code{font-size:13px;font-weight:700}
.sim-score{font-family:var(--mono);font-size:13px;color:var(--accent);font-weight:700}
.edges{display:flex;gap:5px;flex-wrap:wrap}
.edge{font-family:var(--mono);font-size:10.5px;padding:2px 7px;border-radius:3px;
  background:var(--panel);border:1px solid var(--rule2);color:var(--ink3);white-space:nowrap}
.edge.on{border-color:var(--accent);color:var(--accent);font-weight:600}
.chain{display:flex;gap:6px;flex-wrap:wrap;align-items:center;font-size:12.5px}
.chain .rel{font-family:var(--mono);font-size:10px;color:var(--ink3);letter-spacing:.06em}
.chain .rel::before{content:"─["}
.chain .rel::after{content:"]→"}
.chain .cause{font-weight:650;color:var(--stop)}
.chain .act{color:var(--ink2)}
.chain .res{font-family:var(--mono);font-size:10.5px;padding:1px 7px;border-radius:3px}
.chain .res.ok{color:var(--ok);background:var(--ok-bg)}
.chain .res.no{color:var(--skip);background:var(--skip-bg)}

/* ── 단계 이동 바 ──────────────────────────────────────────────────── */
.nav{position:sticky;top:0;z-index:20;margin:0 -22px;padding:9px 22px;
  background:var(--bg);border-bottom:1px solid var(--rule);
  display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.nav-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink3);margin-right:4px}
.nav a{font-family:var(--mono);font-size:11.5px;line-height:1.35;
  text-decoration:none;color:var(--ink2);background:var(--panel);
  border:1px solid var(--rule2);border-left-width:3px;border-radius:4px;
  padding:4px 9px;white-space:nowrap}
.nav a:hover{border-color:var(--accent);color:var(--accent)}
.nav a.done{border-left-color:var(--ok)}
.nav a.blocked{border-left-color:var(--stop)}
.nav a.skipped{border-left-color:var(--skip)}
.nav a.pending{border-left-color:var(--rule)}
.nav a.key{border-left-color:var(--accent);color:var(--accent);font-weight:650}

/* 앵커로 뛰었을 때 이동 바에 제목이 가리지 않게 */
.stage,.evidence,.sim,.doc{scroll-margin-top:70px}

/* ── 논거 블록은 일반 단계와 격을 나눈다 ───────────────────────────── */
/* 진단 근거·조회 방식·이력 그래프가 이 과제의 논거가 실린 자리다. 8번 게이트와
   같은 무게로 놓이면 화면이 그것을 말해 주지 못한다. */
.evidence{border-left:3px solid var(--accent)}
.evidence .ev-head .stage-title{color:var(--accent);font-size:16.5px}

/* ── 부차 단계는 표를 접는다 ───────────────────────────────────────── */
/* 절 자체를 숨기지 않는다. 숨기면 앵커로 이동해도 내용이 안 열려 이동이
   무의미해지고, "무엇이 어디까지 돌았는가"도 접힌 채로 안 읽힌다. */
.stage>details>summary{font-family:var(--mono);font-size:11.5px;
  letter-spacing:.06em;color:var(--ink3);cursor:pointer;padding:2px 0;
  list-style:none}
.stage>details>summary::-webkit-details-marker{display:none}
.stage>details>summary::before{content:"▸ "}
.stage>details[open]>summary::before{content:"▾ "}
.stage>details>summary:hover{color:var(--accent)}

/* 확인 표시 — 못 채운 항목이 눈에 띄어야 한다 */
td .mark{font-family:var(--mono);font-weight:700;margin-right:7px}
td .mark.yes{color:var(--ok)}
td .mark.no{color:var(--stop)}
"""


#: 표를 접지 않는 단계. 판별 항목과 진단은 근거 자체라 접으면 볼 것이 없다.
OPEN_STAGES = {"evidence", "diagnose"}


def _mark(value: str) -> str:
    """앞머리의 확인 표시에만 색을 준다. 나머지 본문은 그대로 escape 한다.

    판별 7항목에서 **확인하지 못한 항목이 눈에 띄어야** 한다. ○ 와 × 가 같은
    먹색이면 6/7 이라는 숫자를 읽기 전까지 어느 항목이 빈 자리인지 모른다.
    """
    text = str(value)
    for sign, kind in (("○", "yes"), ("×", "no")):
        if text.startswith(sign):
            return f'<span class="mark {kind}">{sign}</span>{escape(text[1:])}'
    return escape(text)


def _stage_html(stage: Stage) -> str:
    rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{_mark(v)}</td></tr>" for k, v in stage.rows
    )
    body = (f"<table>{rows}</table>" if rows else "") + (
        f'<p class="note">{escape(stage.note)}</p>' if stage.note else ""
    )

    # 부차 단계는 표만 접는다. 제목·판정·한 줄 요약은 남으므로 접힌 상태에서도
    # "무엇이 어디까지 돌았는가"가 읽히고, 앵커로 뛰어도 빈 자리에 떨어지지 않는다.
    if body and stage.key not in OPEN_STAGES:
        body = f"<details><summary>자세히</summary>{body}</details>"

    return f"""
    <section class="stage {stage.status}" id="stage-{escape(stage.key)}">
      <div class="stage-head">
        <span class="stage-title">{escape(stage.title)}</span>
        <span class="chip {stage.status}">{STATUS_LABEL.get(stage.status, stage.status)}</span>
      </div>
      {f'<div class="headline">{escape(stage.headline)}</div>' if stage.headline else ''}
      {f'<p class="detail">{escape(stage.detail)}</p>' if stage.detail else ''}
      {body}
    </section>
    """


def _nav_html(marks: list[tuple[str, str, str]]) -> str:
    """단계 이동 바.

    전 구간 화면은 세로로 매우 길다(실측 7,466px). 시연 중에 보고 싶은 자리로
    못 가면 스크롤이 발표를 잡아먹는다. **자바스크립트를 붙이지 않는다** —
    앵커와 position:sticky 로 되는 일이고, 화면 하나짜리 시연에 스크립트를
    더하면 시연 중에 깨질 자리만 늘어난다.

    칩의 왼쪽 색이 단계 상태다. 어디서 멈췄는지가 스크롤하지 않고 보인다.
    """
    if not marks:
        return ""
    items = "".join(
        f'<a class="{kind}" href="#{anchor}">{escape(label)}</a>' for anchor, label, kind in marks
    )
    return f'<nav class="nav"><span class="nav-label">단계</span>{items}</nav>'


CAUSE_KO = {
    "threshold": "임계값 문제", "bank_contamination": "뱅크 오염",
    "coverage_gap": "커버리지 부족", "normal_overlap": "정상 분포 중첩",
    "equipment_optics": "설비·광학", "criteria": "기준 문제",
}


def _taxonomy_html(outcome: RunOutcome) -> str:
    """진단 지식 체계 — 원인 6종과 무엇으로 갈리는가.

    **화면에 뜨는 이 표가 판정에 쓰인 표 그 자체다.** `agents/ontology.py` 가
    `diagnose.py` 의 재구성 여부·조치·금지를 그대로 가져오므로, 여기 적힌 값과
    `decide()` 가 쓴 값이 어긋날 수 없다. 설명용으로 따로 그린 그림이면
    한쪽만 고쳐지고 화면이 조용히 거짓말을 하게 된다.

    언어 모델은 이 표를 `lookup_ontology` 도구로 **읽을 수만** 있다. 원인은
    판별 7항목으로 `decide()` 가 낸다. 이 경계가 이 화면에 적혀 있어야 하는
    이유는, 모델에게 지식을 주는 순간 "모델이 원인을 골랐다"로 오해되기
    때문이다.
    """
    diagnosis = outcome.diagnosis
    if diagnosis is None:
        return ""

    rows = ""
    for cause in cause_names():
        node = CAUSES[cause]
        data = node.to_dict()
        here = cause == diagnosis.cause
        rebuild = data["requires_bank_rebuild"]
        forbidden = ", ".join(action_label(a) for a in data["forbidden_actions"]) or "—"
        rows += (
            f'<tr class="{"here" if here else ""}">'
            f'<td>{escape(node.label)}{" ←" if here else ""}</td>'
            f'<td><span class="rb {"yes" if rebuild else "no"}">'
            f'{"재구성" if rebuild else "재구성 아님"}</span></td>'
            f'<td class="items">판별 {"·".join(str(n) for n in data["decided_by"])}</td>'
            f'<td>{escape(forbidden)}</td></tr>'
        )

    asked = [name for name, _ in outcome.tool_trace if name == "lookup_ontology"]
    state = (
        f"이번 실행에서 모델이 {len(asked)}회 조회했습니다"
        if asked else
        "이번 실행에서는 조회되지 않았습니다 — 물어볼 언어 모델이 없으면 부르지 않습니다"
    )

    rebuild_causes = [CAUSES[c].label for c in cause_names()
                      if CAUSES[c].to_dict()["requires_bank_rebuild"]]
    return f"""
    <div class="evidence" id="block-taxonomy">
      <div class="ev-head">
        <span class="stage-title">진단 지식 체계 — 무엇으로 갈리는가</span>
        <span class="kind schema">스키마 조회</span>
      </div>
      <table class="tax">{rows}</table>
      <p class="detail">
        원인 {len(cause_names())}종 중 <strong>뱅크 재구성이 답인 것은
        {len(rebuild_causes)}종뿐</strong>입니다({escape(", ".join(rebuild_causes))}).
        나머지는 다시 만들어도 해결되지 않거나 오히려 나빠집니다.
        <strong>뱅크 오염과 정상 분포 중첩은 판별 5번 하나로 갈리고</strong>
        조치가 정반대입니다 — 그래서 5번을 얻지 못하면 판정하지 않습니다.
      </p>
      <p class="note">
        언어 모델은 이 표를 <code>lookup_ontology</code> 도구로 <strong>읽을 수만</strong>
        있습니다. <strong>이 조회는 원인을 정하지 않습니다</strong> — 판정은 판별
        7항목을 모아 <code>decide()</code> 가 규칙으로 냅니다. {escape(state)}.
      </p>
    </div>
    """


def _ontology_html(outcome: RunOutcome) -> str:
    """이슈 이력 그래프 — 온톨로지가 실제로 쓰이는 유일한 자리.

    조회 계층의 나머지 일곱은 조인이다. 여기만 그래프인 이유는 운영 이력이
    **개체 사이의 관계 자체가 답**이기 때문이다. "이 증상이 다른 라인에서 어떤
    원인으로 규명돼 어떤 조치로 해결됐나"는 이슈→원인→조치→결과를 따라가야
    나온다.

    **경로를 그린다.** 유사도 숫자만 띄우면 왜 비슷하다고 봤는지 검증할 수
    없고, 그러면 중복 차단이라는 역할도 못 맡긴다.

    역할이 좁은 것은 설계다. 과거 사례가 비슷하다고 이번 원인을 그것으로 정하면
    진단이 유사도 맞히기가 된다. 그래프는 "이미 답이 나온 일인가"만 묻는다.
    """
    intake = outcome.intake
    if intake is None or not intake.similar:
        return ""

    query = intake.report
    blocked = intake.verdict == "duplicate"

    def node_html(issue) -> str:
        matched = set(issue.matched_on or [])
        chips = "".join(
            f'<span class="edge{" on" if key in matched else ""}">{escape(label)} {escape(value)}</span>'
            for key, label, value in (
                ("line", "발생_라인", issue.line),
                ("object_name", "대상_품목", issue.object_name),
                ("defect_type", "결함_유형", issue.defect_type or "—"),
            )
        )
        return f"""
        <div class="onode{' hit' if blocked and intake.duplicate_of == issue.issue_id else ''}">
          <div class="onode-head">
            <code>{escape(issue.issue_id)}</code>
            <span class="sim-score">{issue.similarity:.2f}</span>
          </div>
          <div class="edges">{chips}</div>
          <div class="chain">
            <span class="cause">{escape(CAUSE_KO.get(issue.cause, issue.cause))}</span>
            <span class="rel">조치</span>
            <span class="act">{escape(issue.action)}</span>
            <span class="rel">결과</span>
            <span class="res {'ok' if issue.resolved else 'no'}">
              {'해결' if issue.resolved else '미해결'}</span>
          </div>
          <p class="hint">{escape(issue.summary)}</p>
        </div>
        """

    verdict_note = (
        f"<strong>중복으로 끊었습니다</strong> — {escape(intake.duplicate_of or '')} 과 "
        f"같은 라인·같은 증상이며 이미 조치가 끝났습니다. 진단하지 않습니다."
        if blocked else
        "<strong>중복이 아니라 진행합니다.</strong> 유사도가 높은 건도 "
        "<em>라인이 다릅니다</em> — 라인마다 뱅크가 따로이므로 1라인 뱅크가 "
        "뱅크 오염됐다고 2라인도 그렇다는 뜻이 아닙니다. 관련 사례로만 넘깁니다."
    )
    return f"""
    <div class="evidence" id="block-ontology">
      <div class="ev-head">
        <span class="stage-title">이슈 이력 그래프 — 이미 답이 나온 일인가</span>
        <span class="kind graph">그래프 검색</span>
      </div>
      <div class="query-node">
        <code>이번 이슈</code>
        <span class="edge on">발생_라인 {escape(query.line or '—')}</span>
        <span class="edge on">대상_품목 {escape(query.object_name or '—')}</span>
        <span class="edge on">결함_유형 {escape(query.defect_type or '—')}</span>
      </div>
      <div class="onodes">{''.join(node_html(i) for i in intake.similar[:4])}</div>
      <p class="detail">{verdict_note}</p>
      <p class="note">
        진한 간선이 이번 이슈와 겹친 자리입니다. <strong>이 그래프는 원인을
        정하지 않습니다</strong> — 과거가 비슷하다고 이번 원인을 그것으로 정하면
        진단이 유사도 맞히기가 됩니다. 원인은 판별 7항목으로 매번 새로 규명합니다.
      </p>
    </div>
    """


def _evidence_visual_html(outcome: RunOutcome) -> str:
    """진단 근거를 눈으로 — 히트맵, 이상 점수, 역추적한 두 자리.

    이 과제의 주장은 "판단 근거가 모델 안에 이미 있다"이다. 그런데 화면이
    문장으로만 *"격자(6,5), 거리 0.0059"* 라고 적으면 확인할 방법이 없다.
    그 자리를 실제로 잘라 나란히 놓으면 사람이 직접 판단할 수 있다.

    **여기 뜨는 값은 전부 실제 추론 결과다.** 히트맵은 `patch_distances` 를
    그대로 칠한 것이고, 잘라낸 조각은 `inspection.crop` 이 같은 좌표계로
    낸 것이다. 그림을 따로 계산하면 두 벌이 되고 한쪽만 틀어진다.
    """
    result = outcome.inference
    if result is None or not outcome.query_image or not result.patch_distances:
        return ""

    top = result.top_match
    grid_h, grid_w = outcome.grid
    flat = [v for row in result.patch_distances for v in row]
    lo, hi = min(flat), max(flat)
    span = (hi - lo) or 1.0

    # 역추적이 지목한 칸에만 붙는 표시. f-string 식 안에 역슬래시를 두면
    # Python 3.11 에서 SyntaxError 다(3.12 의 PEP 701 부터 허용). 대상 환경이
    # 3.11 이므로 따옴표를 밖으로 뺀다.
    hot_attr = " class='hot'"
    cells = "".join(
        f'<i style="opacity:{(v - lo) / span:.3f}"'
        f'{hot_attr if top and r == top.query.row and c == top.query.col else ""}'
        f' title="({r},{c}) {v:.4f}"></i>'
        for r, row in enumerate(result.patch_distances)
        for c, v in enumerate(row)
    )

    threshold = outcome.threshold or 1.0
    # 점수 막대는 임계값을 눈금 100% 로 잡는다. 넘으면 100% 에서 멈추되
    # 숫자는 그대로 적는다 — 막대가 잘렸다고 값이 바뀐 것은 아니다.
    fill = min(result.score / threshold, 1.0) * 100 if threshold else 0.0
    verdict = "검출" if result.score >= threshold else "미검"

    def crop_url(image: str, row: int, col: int) -> str:
        query = urlencode({"row": row, "col": col, "grid_h": grid_h,
                           "grid_w": grid_w, "margin": 24})
        return f"/crop/{quote(image)}?{query}"

    traced = ""
    if top:
        q, b = top.query, top.bank
        traced = f"""
        <div class="pair">
          <figure>
            <img src="{escape(crop_url(outcome.query_image, q.row, q.col))}" alt="질의 패치">
            <figcaption>질의 패치 ({q.row},{q.col})<br><span>못 잡은 이미지의 이 자리</span></figcaption>
          </figure>
          <div class="arrow">최근접<br><b>{top.distance:.4f}</b></div>
          <figure>
            <img src="{escape(crop_url(b.source_image, b.row, b.col))}" alt="뱅크 패치">
            <figcaption>뱅크 패치 ({b.row},{b.col})<br>
              <span>{escape(Path(b.source_image).name)}</span></figcaption>
          </figure>
        </div>
        """
    return f"""
    <div class="evidence" id="block-evidence">
      <div class="ev-head">
        <span class="stage-title">진단 근거 — 모델이 어디를 보고 통과시켰나</span>
        <span class="sim-state">{escape(outcome.bank_version)}</span>
      </div>
      <div class="ev-grid">
        <div>
          <div class="ev-label">이상 점수 히트맵 · {grid_h}×{grid_w}</div>
          <div class="heat" style="grid-template-columns:repeat({grid_w},1fr)">{cells}</div>
          <p class="hint">진할수록 정상에서 멀다. 테두리 친 칸이 가장 높은 자리다.</p>
        </div>
        <div>
          <div class="ev-label">이상 점수</div>
          <div class="score"><span style="width:{fill:.1f}%"></span></div>
          <p class="score-num">
            <b>{result.score:.4f}</b> / 임계값 {threshold:.2f}
            <em class="{'over' if verdict == '검출' else 'under'}">{verdict}</em>
          </p>
          <p class="hint">
            임계값 아래라 양품으로 나갔습니다. <strong>점수가 낮다고 이상이
            없는 것이 아닙니다</strong> — 어디가 이상한지는 히트맵이 압니다.
          </p>
        </div>
      </div>
      {traced}
      <p class="note">
        역추적한 두 자리를 같은 좌표계로 잘라 나란히 놓은 것입니다.
        <strong>이 뱅크 패치가 결함이면 뱅크 오염, 진짜 정상품이면 정상 분포
        중첩이며 조치가 정반대입니다</strong>(판별 5번).
      </p>
    </div>
    """


def _retrieval_html(outcome: RunOutcome) -> str:
    """어떤 자료를 **어떤 방식으로** 찾았는가.

    "전부 RAG 로 찾습니다"가 보기에는 좋지만 사실이 아니고, 사실이 아닌 것을
    띄우면 심사에서 한 번만 파고들어도 무너진다. 그리고 이 구분 자체가 이
    과제의 논거다 — **진단의 신뢰도는 벡터 검색이 아니라 결정론적 조회에서
    나온다.** MES 와 이미지 메타데이터를 임베딩하면 비슷한 로트를 섞어 온다.

    목록은 조회 계층이 남긴 실제 호출 기록이다. 지어낸 것이 아니다.
    """
    if not outcome.retrievals:
        return ""

    counts: dict[str, int] = {}
    for call in outcome.retrievals:
        counts[call["kind"]] = counts.get(call["kind"], 0) + 1

    chips = "".join(
        f'<span class="kind {escape(kind)}">{escape(RETRIEVAL_LABEL.get(kind, (kind, ""))[0])}'
        f' {n}</span>'
        for kind, n in sorted(counts.items(), key=lambda kv: -kv[1])
    )
    rows = "".join(
        f'<tr><td><span class="kind {escape(c["kind"])}">'
        f'{escape(RETRIEVAL_LABEL.get(c["kind"], (c["kind"], ""))[0])}</span></td>'
        f'<td><code>{escape(c["name"])}</code></td>'
        f'<td>{escape(", ".join(f"{k}={v}" for k, v in c["arguments"].items()) or "—")}</td></tr>'
        for c in outcome.retrievals
    )
    legend = "".join(
        f"<li><span class=\"kind {escape(kind)}\">{escape(label)}</span> {escape(why)}</li>"
        for kind, (label, why) in RETRIEVAL_LABEL.items()
        if kind in counts
    )
    return f"""
    <div class="evidence" id="block-retrieval">
      <div class="ev-head">
        <span class="stage-title">무엇을 어떻게 찾았나</span>
        <span class="sim-state">{escape(str(len(outcome.retrievals)))}회 조회</span>
      </div>
      <div class="chips">{chips}</div>
      <p class="detail">
        <strong>대부분이 조인입니다.</strong> "3라인 A-217 로트 캡슐 이미지 목록"은
        정확히 답할 문제이고, 임베딩하면 비슷한 로트를 섞어 옵니다.
        벡터·그래프 검색은 <strong>과거 유사 사례 하나</strong>에만 쓰며,
        그 역할도 진단 근거가 아니라 중복 작업 차단입니다.
      </p>
      <table class="ret">{rows}</table>
      <ul class="legend">{legend}</ul>
      <p class="note">조회 계층이 남긴 실제 호출 기록입니다.</p>
    </div>
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
    <div class="sim" id="block-simulator">
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


def _source_banner(on_visa: bool) -> str:
    """어떤 데이터로 섰는지 화면이 스스로 말하게 한다.

    **합성으로 떨어진 것을 모르고 보면 수치를 실측으로 오해한다.** 반대로 VisA
    로 섰는데 "합성입니다"가 계속 떠 있으면 실측 결과를 스스로 깎는다. 둘 다
    시연에서 손해라 표시를 데이터에 묶는다.
    """
    if on_visa:
        return """
  <div class="banner">
    <strong>VisA 실데이터로 돌고 있습니다.</strong> 이미지와 뱅크는 실제이고,
    조회 계층(MES·이슈 이력)만 목입니다.
  </div>"""
    return """
  <div class="banner warn">
    <strong>합성 이미지로 돌고 있습니다.</strong> 조회 계층도 목입니다.
    성능 수치가 아니라 <strong>경로가 이어지는지</strong>를 보는 화면입니다.
    VisA 원본을 저장소 아래 <code>VisA_20220922/</code> 에 두면 실데이터로 섭니다.
  </div>"""


def render_page(outcome: RunOutcome | None, issue_text: str, patch_verdict: str = "defect",
                context: dict[str, str] | None = None, on_visa: bool = False) -> str:
    options = [
        ("defect", "결함이다 → 뱅크 오염"),
        ("genuine_normal", "진짜 정상품이다 → 정상 분포 중첩"),
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
            ("line", "라인", "예: line_01"),
            ("object_name", "품목", "예: pcb1"),
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
        stages_html: list[str] = []
        #: 이동 바에 실을 자리들. (앵커, 이름, 칩 종류)
        marks: list[tuple[str, str, str]] = []

        def add_block(html: str, anchor: str, label: str) -> None:
            """논거 블록을 끼우고 이동 바에도 올린다.

            블록은 조건이 안 맞으면 빈 문자열을 돌려준다. 그때 이동 바에만
            남으면 눌러도 아무 데도 안 간다.
            """
            if not html.strip():
                return
            stages_html.append(html)
            marks.append((anchor, label, "key"))

        for stage in outcome.stages:
            # 시뮬레이터는 섀도 단계 바로 앞에 끼운다. 숫자만 적힌 표보다
            # 무엇이 어떻게 갈렸는지가 먼저 보여야 한다.
            if stage.key == "shadow":
                add_block(_simulator_html(outcome), "block-simulator", "코어셋 검증")
            stages_html.append(_stage_html(stage))
            marks.append((f"stage-{stage.key}", stage.title, stage.status))
            # 진단 바로 뒤에 근거를 그린다. 문장으로만 적으면 확인할 방법이 없다.
            if stage.key == "diagnose":
                add_block(_evidence_visual_html(outcome), "block-evidence", "진단 근거")
                # 근거 다음에 체계를 놓는다. "이 근거가 왜 이 원인이 되는가"는
                # 표를 봐야 답이 되고, 표가 앞에 오면 결론부터 읽게 된다.
                add_block(_taxonomy_html(outcome), "block-taxonomy", "원인 체계")
            if stage.key == "evidence":
                add_block(_retrieval_html(outcome), "block-retrieval", "조회 방식")
            # 그래프는 인테이크 바로 뒤. "이미 답이 나온 일인가"를 묻는 자리다.
            if stage.key == "intake":
                add_block(_ontology_html(outcome), "block-ontology", "이력 그래프")

        doc = ""
        if outcome.approval_markdown:
            # 10번 단계 이름이 "승인 요청"이라 문서 쪽은 다르게 적는다.
            marks.append(("doc-approval", "승인 문서", "key"))
            doc = f"""
            <div class="doc" id="doc-approval">
              <h2>승인 요청 문서</h2>
              <pre>{escape(outcome.approval_markdown)}</pre>
              <p class="note">원문: <a href="/approval">/approval</a></p>
            </div>
            """

        body = (
            _nav_html(marks)
            + _driver_html(outcome)
            + '<div class="flow">'
            + "".join(stages_html)
            + "</div>"
            + doc
        )

    source_banner = _source_banner(on_visa)

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

  {source_banner}

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
