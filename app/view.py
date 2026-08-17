"""화면 렌더링.

템플릿 파일을 따로 두지 않고 한곳에 모았다. 시연용 화면 하나뿐이라
파일을 나누면 오히려 찾기 번거롭다.

색은 판정 성격에 따라 고정한다. 진행됨·차단됨·건너뜀이 한눈에 갈려야
"어디서 멈췄는가"가 바로 보인다.
"""

from __future__ import annotations

import json
import pathlib
import re
from html import escape
from inspect import signature
from typing import Any
from pathlib import Path
from urllib.parse import quote, urlencode

from agents.ontology import ACTION_LABEL_KO, CAUSES, CHECKS, action_label, cause_names
from app.pipeline import FALLBACK_SEQUENCE, RunOutcome, Stage, run_pipeline
from lookup.base import RETRIEVAL_LABEL

STATUS_LABEL = {"done": "진행", "blocked": "중단", "skipped": "건드리지 않음",
                "pending": "미도달"}

#: 멈춘 단계에 붙는 이름표. 같은 "중단"이라도 고장과 설계된 정지는 다르다.
WHY_LABEL = {"blocked": "왜 멈췄나", "skipped": "왜 건드리지 않았나"}

#: 판정 임계값의 기본값. **여기 숫자를 적지 않는다** — `run_pipeline` 의 기본값을
#: 그대로 읽는다. 화면에 손으로 적어 두면 파이프라인이 값을 바꿨을 때 화면만
#: 옛 숫자를 계속 보여준다.
DEFAULT_THRESHOLD = signature(run_pipeline).parameters["threshold"].default

# ── 화면에 영어로 나오던 것들 ───────────────────────────────────────────
#
# 지적을 받았다. *"각 작업 이름이 영어거나 이해하기 어려운 단어로 표시된다"*.
# 실제로 화면에 `intake_issue` · `defect_visible` · `pass` · `auroc` 가 그대로
# 나가고 있었다.
#
# **원래 이름을 바꾸지 않고 한국어를 덧붙인다.** 이름의 출처는 `agents/` 와
# `app/pipeline.py` 이고, 화면에서 갈아치우면 같은 것의 이름이 둘이 되어
# 한쪽만 고쳐진다. 그래서 화면은 한국어를 앞에 놓고 원래 이름을 함께 남긴다.
#
# 판별 7항목은 여기 적지 않는다 — `agents/ontology.py` 의 `CHECKS` 에 이미
# 한국어 물음이 있어 그것을 읽는다.

#: 값에 그대로 나오던 영어. 낱말 단위로만 바꾼다.
VALUE_KO: dict[str, str] = {
    "pass": "양품",
    "defect": "불량",
    "review": "사람 확인 필요",
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
    "none": "없음",
    "above": "임계값 위",
    "near": "임계값 근처",
    "below": "임계값 아래",
    "unknown": "판독 못 함",
    "not_visible": "결함이 안 보임",
    "visible": "결함이 보임",
    "genuine_normal": "진짜 정상품",
    "normal": "정상",
    "true": "예",
    "false": "아니오",
    "True": "예",
    "False": "아니오",
}

#: 조치 id 도 값 자리에 그대로 나온다. 진단 카드의 권고·금지 조치가
#: `remove_contaminated_samples, rebuild_bank` 로 찍혀 있었다(4090 실행 화면).
#:
#: **한국어 이름을 여기 적지 않는다** — `agents/ontology.py` 의
#: `ACTION_LABEL_KO` 를 그대로 가져온다. 원인 체계 표는 이미 그것을 쓰고
#: 있어서, 화면이 따로 적으면 같은 조치가 표에서는 "혼입 이미지 제거",
#: 진단 카드에서는 다른 이름으로 나가게 된다.
_TRANSLATE = {**VALUE_KO, **ACTION_LABEL_KO}

#: 낱말 경계만으로는 **경로 안까지 바꾼다.** 진단 근거에 파일 경로가 그대로
#: 실리는데 `line_01/pcb1/defect/defect_000.png` 의 가운데가 `불량` 이 되어,
#: 화면에 적힌 경로가 실제 파일을 안 가리켰다. 앞뒤에 `/` `.` `-` `_` 가
#: 붙어 있으면 낱말이 아니라 이름의 조각이므로 건드리지 않는다.
_VALUE_RE = re.compile(
    r"(?<![\w/.\-])(?:"
    + "|".join(re.escape(k) for k in sorted(_TRANSLATE, key=len, reverse=True))
    + r")(?![\w/.\-])"
)

#: 굵게 표시. **다른 모듈이 쓴 마크다운이 화면에 글자 그대로 나왔다** —
#: 4090 화면에서 판별 7항목 설명이 `**7번 면적에 기준선이 없어...**` 로
#: 별표째 찍혔다. 하필 "이 값은 판정 기준과 대조할 수 없다"는 경고라
#: 강조가 살아야 하는 자리다. escape 한 뒤에 바꾸므로 태그가 새로 들어올
#: 일은 없다.
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)

#: 표 왼쪽 칸에 영어 식별자로 나오던 것. 게이트 지표 이름이 대부분이다.
ROW_LABEL_KO: dict[str, str] = {
    "detection_rate": "검출률 (불량을 실제로 잡는 비율)",
    "false_positive_rate": "과검률 (양품을 불량이라 하는 비율)",
    "auroc": "분리도 AUROC (임계값과 무관한 구분 능력)",
    "newly_missed": "새로 놓친 건수 (전에는 잡던 것을 못 잡게 된 수)",
    "improvement": "개선 폭 (이전 뱅크보다 나아졌는가)",
}

#: 과제 고유 용어의 한 줄 풀이. 물음표 표시로 붙는다.
#:
#: **정의의 출처는 `docs/용어사전.md` 다.** 여기 있는 것은 화면에 걸 한 줄
#: 요약이며, 뜻이 갈리면 용어사전 쪽이 맞다. 새 용어를 만들지 않는다.
TERM_KO: dict[str, str] = {
    "뱅크": "정상 이미지를 잘라 만든 특징값 모음. 검사할 때 여기서 가장 가까운 것을 찾아 거리로 판정한다",
    "품목 뱅크": "품목마다 따로 있는 뱅크. 캡슐 뱅크로 기판을 판정할 수 없다",
    "코어셋": "뱅크에서 서로 먼 것만 추려 크기를 줄인 것. 전부 담으면 느려서 대표만 남긴다",
    "역추적": "미검출 이미지가 뱅크의 어느 정상 패치와 가까웠는지 되돌아 찾는 것",
    "패치": "이미지를 격자로 자른 칸 하나",
    # 「섀도」는 기획서에 정의된 용어라 문서에서는 살린다. 화면은 새 이름을
    # 쓰되 뜻풀이는 양쪽에 걸어 둔다 — 문서를 먼저 읽고 온 사람과 화면을
    # 먼저 본 사람이 같은 설명에 닿아야 한다.
    "신구 비교": "새 뱅크를 실제 판정에 쓰지 않고 같은 이미지에 나란히 돌려, 판정이 서로 다른 것만 뽑는 검증(섀도 평가)",
    "섀도": "새 뱅크를 실제 판정에 쓰지 않고 같은 이미지에 나란히 돌려, 판정이 서로 다른 것만 뽑는 검증",
    "홀드아웃": "학습에 안 쓰고 남겨 둔 이미지. 성능을 재는 데 쓴다",
    "게이트": "새 뱅크를 배포 후보로 넘길지 정하는 통과 기준",
    "성능 검증": "새 뱅크를 배포 후보로 넘길지 정하는 통과 기준. 통과해도 배포되지 않는다",
    "미검": "불량인데 양품으로 판정한 것. 이 과제가 다루는 문제",
    "미검출": "불량인데 양품으로 판정한 것. 이 과제가 다루는 문제",
    "과검": "양품인데 불량으로 판정한 것. 임계값을 내리면 늘어난다",
    "과검출": "양품인데 불량으로 판정한 것. 임계값을 내리면 늘어난다",
    "임계값": "이 값을 넘으면 불량이라 판정하는 경계 점수",
    "이상 점수": "가장 가까운 정상 패치와의 거리. 클수록 정상에서 멀다",
    "혼입": "정상만 들어가야 할 뱅크에 결함 이미지가 잘못 섞여 들어간 것",
    "인테이크": "이슈를 접수해 정보가 충분한지, 이미 해결된 건인지 판단하는 단계",
    "큐레이션": "뱅크에 무엇을 넣고 무엇을 뺄지 정하는 단계",
    "데이터 선별": "뱅크에 무엇을 넣고 무엇을 뺄지 정하는 단계",
    "MES": "생산 실행 시스템. 어느 제품이 어느 로트에서 언제 나왔는지가 여기 있다",
    "로트": "같은 조건에서 함께 생산된 묶음",
}


#: 긴 것부터 본다. "미검출"이 "미검"보다 먼저 걸려야 뒤에 "출"만 남지 않는다.
_TERMS = sorted(TERM_KO, key=len, reverse=True)


def _gloss(text: str) -> str:
    """본문에 나온 과제 용어에 뜻풀이를 붙인다 — **처음 한 번만.**

    같은 낱말이 나올 때마다 표시가 붙으면 문장이 읽히지 않는다. 그리고
    이 함수는 **이미 escape 된 문자열을 받는다** — 안 그러면 붙인 표시가
    다시 escape 되어 화면에 태그가 그대로 보인다.

    **원문을 한 번만 훑는다.** 낱말마다 `replace` 를 돌리면 앞서 끼운 뜻풀이
    안을 다시 뒤진다. 뜻풀이 문장에도 용어가 들어 있어서(예: "품목 뱅크" 의
    풀이에 "뱅크" 가 있다) `title` 속성 안에 `<abbr>` 이 또 들어가고, 그러면
    설명이 통째로 깨진다. 실제로 그렇게 깨져 있었다.
    """
    used: set[str] = set()
    out: list[str] = []
    i = 0
    while i < len(text):
        for term in _TERMS:
            if term in used or not text.startswith(term, i):
                continue
            out.append(f'<abbr title="{escape(TERM_KO[term])}">{escape(term)}</abbr>')
            used.add(term)
            i += len(term)
            break
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _value_ko(text: str) -> str:
    """값에 남은 영어를 한국어로. 낱말 전체가 맞을 때만 바꾼다.

    부분 일치로 바꾸면 `defect_visible` 의 `defect` 까지 건드려 말이 깨진다.
    낱말 경계를 쓰면 밑줄은 낱말 문자라 `genuine_normal` 안의 `normal` 도
    안 걸린다.
    """
    return _VALUE_RE.sub(lambda m: _TRANSLATE[m.group(0)], text)


def _bold(text: str) -> str:
    """`**굵게**` 를 태그로. **escape 뒤에 부른다.**

    앞에서 부르면 여기서 만든 태그가 escape 돼 화면에 `&lt;strong&gt;` 이
    보인다. 뒤에서 부르므로 별표 사이에 들어올 수 있는 것은 이미 escape 된
    글자뿐이다.
    """
    return _BOLD_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", text)


def _row_label(key: str) -> str:
    """표 왼쪽 칸을 사람 말로. 원래 이름은 함께 남긴다.

    판별 항목은 `agents/ontology.py` 의 `CHECKS` 에서 한국어 물음을 읽는다.
    화면이 따로 적으면 정의가 둘이 되고, 그러면 규칙이 바뀌었을 때 화면만
    옛 물음을 보여준다.
    """
    raw = str(key)

    # "3. score_position" 꼴 — 판별 7항목.
    head = raw.split(".", 1)
    if len(head) == 2 and head[0].strip().isdigit():
        item_no = int(head[0].strip())
        name = head[1].strip()
        item = next((c for c in CHECKS if c.item_no == item_no), None)
        if item is not None:
            return (f'{item_no}. {_gloss(escape(item.question))}'
                    f'<small>{escape(name)}</small>')

    if raw in ROW_LABEL_KO:
        return f'{escape(ROW_LABEL_KO[raw])}<small>{escape(raw)}</small>'
    return _gloss(escape(raw))

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
/* 갈린 건은 흘려보내고 끝내지 않는다. 숫자만 쌓이면 "무엇이 왜 갈렸는가"가
   남지 않아 사람이 확인할 목록이 되지 못한다. */
.piece.flip img{box-shadow:0 0 0 3px var(--accent)}
.flips{display:flex;flex-direction:column;gap:7px}
.flip-row{display:flex;gap:11px;align-items:center;background:var(--panel2);
  border:1px solid var(--rule2);border-radius:6px;padding:8px 11px}
.flip-row.focus{border-color:var(--accent)}
.flip-row img{width:48px;height:48px;object-fit:cover;border-radius:4px;
  border:1px solid var(--rule);flex:0 0 auto}
.flip-row .txt{min-width:0;display:flex;flex-direction:column;gap:2px}
.flip-row .nm{font-family:var(--mono);font-size:11.5px;color:var(--ink3);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.flip-row .mv{font-family:var(--mono);font-size:13px;color:var(--ink)}
.flip-row .mv s{text-decoration:none;color:var(--ink3)}
.flip-row .mv b{color:var(--accent)}
.flip-row .kd{font-family:var(--mono);font-size:10.5px;padding:2px 8px;
  border-radius:3px;margin-left:auto;white-space:nowrap;flex:0 0 auto}
.flip-row .kd.up{color:var(--ok);background:var(--ok-bg)}
.flip-row .kd.down{color:var(--stop);background:var(--stop-bg)}
.flip-row .why2{font-size:12px;color:var(--ink3)}

/* ── 진단 근거 시각화 ──────────────────────────────────────────────── */
.evidence{background:var(--panel);border:1px solid var(--rule);border-radius:8px;
  padding:18px;display:flex;flex-direction:column;gap:14px}
.ev-head{display:flex;justify-content:space-between;align-items:baseline;
  gap:12px;flex-wrap:wrap}
/* 히트맵·역추적 두 자리를 한 줄에 놓는다. 세로로 쌓으면 "이 이미지의 이
   자리가 뱅크의 저 자리와 가까웠다"가 한눈에 안 이어진다. */
.ev-grid{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}
.ev-grid>div{flex:1;min-width:250px;display:flex;flex-direction:column;gap:8px}
.ev-grid>div.wide{flex:1.35 1 320px}
.ev-label{font-family:var(--mono);font-size:11px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink3)}
/* 격자 칸 수가 이미지 해상도를 따라간다. 합성 64px 에서는 8x8 인데 VisA
   512px 에서는 **64x64 로 4,096 칸**이다. 칸마다 최소 높이를 주면 64행이
   640px 을 요구해 300px 상자를 넘쳐 흐르고, 아래 글자 위로 붉은 줄이 덮인다.
   실제로 4090 화면이 그랬다. 칸 크기는 상자가 정하게 두고 넘치면 자른다. */
.heat{display:grid;gap:var(--cell-gap,3px);grid-auto-rows:1fr;overflow:hidden;
  background:var(--panel2);padding:8px;border-radius:6px;
  border:1px solid var(--rule2);aspect-ratio:1;max-width:300px}
.heat i{background:var(--stop);border-radius:var(--cell-r,2px);min-height:0;
  min-width:0;display:block}
/* 역추적이 지목한 칸. 이 한 칸이 진단의 출발점이라 눈에 먼저 들어와야 한다. */
.heat i.hot{outline:3px solid var(--accent);outline-offset:2px;position:relative;z-index:1}

/* 점수 눈금. 임계값을 눈금 위 한 자리에 세워, 점수가 그보다 왼쪽이라
   양품으로 나갔다는 것이 막대 하나로 읽히게 한다. */
.scale{display:flex;align-items:center;gap:7px;font-family:var(--mono);
  font-size:10.5px;color:var(--ink3)}
.scale i{flex:1;height:9px;border-radius:5px;border:1px solid var(--rule2);
  background:linear-gradient(90deg,var(--panel2),var(--stop))}
.gauge{position:relative;height:22px;background:var(--panel2);
  border:1px solid var(--rule2);border-radius:6px;overflow:hidden}
.gauge>b{position:absolute;left:0;top:0;bottom:0;
  background:linear-gradient(90deg,var(--ok),var(--stop));opacity:.75}
.gauge>i{position:absolute;top:0;bottom:0;width:2px;background:var(--ink)}
.gauge>u{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);
  text-decoration:none}
.ticks{display:flex;justify-content:space-between;font-family:var(--mono);
  font-size:11px;color:var(--ink3);margin:0}
.score-num{margin:0;font-family:var(--mono);font-size:14px}
.score-num em{font-style:normal;font-size:11px;padding:2px 8px;border-radius:3px;
  margin-left:8px;letter-spacing:.06em}
.score-num em.under{color:var(--stop);background:var(--stop-bg)}
.score-num em.over{color:var(--ok);background:var(--ok-bg)}
.after{font-size:13px;color:var(--ink2);margin:0;background:var(--ok-bg);
  border-radius:5px;padding:8px 11px}
.pair{display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  background:var(--panel2);border:1px solid var(--rule2);border-radius:6px;padding:14px}
.pair figure{margin:0;display:flex;flex-direction:column;gap:6px;align-items:center}
.pair img{width:116px;height:116px;object-fit:cover;border-radius:5px;
  border:2px solid var(--rule)}
.pair figure.bank img{border-color:var(--accent)}
.pair figcaption{font-family:var(--mono);font-size:11px;color:var(--ink2);
  text-align:center;line-height:1.5}
.pair figcaption span{color:var(--ink3)}
.pair figcaption b{display:block;font-size:10.5px;letter-spacing:.06em;
  padding:2px 7px;border-radius:3px;margin-top:3px;color:var(--accent);
  background:var(--panel)}
.arrow{font-family:var(--mono);font-size:11px;color:var(--ink3);text-align:center;
  letter-spacing:.08em}
.arrow b{display:block;font-size:15px;color:var(--accent);margin-top:3px}

/* ── 뱅크 구성 ─────────────────────────────────────────────────────── */
/* "제거 2장"이라는 숫자만으로는 왜 빼는지 알 수 없다는 지적을 받았다.
   뱅크를 통째로 늘어놓고 뺄 것에 표시를 붙인다. */
.bank-row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}
.bm-grid{flex:1;min-width:260px;display:grid;
  grid-template-columns:repeat(auto-fill,minmax(52px,1fr));gap:5px;
  background:var(--panel2);border:1px solid var(--rule2);border-radius:6px;
  padding:9px;max-height:330px;overflow-y:auto}
.bm{margin:0;position:relative}
.bm img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:3px;
  border:2px solid var(--rule);display:block;background:var(--panel)}
.bm.out img{border-color:var(--stop)}
.bm figcaption{position:absolute;left:0;right:0;bottom:0;font-family:var(--mono);
  font-size:9px;text-align:center;color:#fff;background:var(--stop);
  border-radius:0 0 3px 3px;padding:1px 0}
.bm.big{flex:0 0 132px}
.bm.big img{width:132px;height:132px;border-color:var(--accent)}
.bm.big figcaption{position:static;color:var(--ink2);background:none;
  font-size:11px;padding-top:5px}
.picked{margin:0;padding-left:18px;font-size:13px;color:var(--ink2);
  display:flex;flex-direction:column;gap:4px}
.picked code{font-family:var(--mono);font-size:12px;color:var(--stop)}
.decide{background:var(--panel2);border:1px solid var(--rule2);padding:14px;gap:10px}
button.ghost{background:var(--panel);color:var(--ink2);border:1px solid var(--rule)}
button.ghost:hover{border-color:var(--stop);color:var(--stop);filter:none}
.chip.ok{color:var(--ok);background:var(--ok-bg)}
.chip.no{color:var(--skip);background:var(--skip-bg)}
.fgrid{display:flex;gap:12px;flex-wrap:wrap}
.fcard{flex:1;min-width:190px;background:var(--panel2);border:1px solid var(--rule2);
  border-radius:6px;padding:12px;display:flex;flex-direction:column;gap:9px}
.fhead{display:flex;gap:8px;align-items:baseline}
.fhead code{font-family:var(--mono);font-size:13px;font-weight:700;color:var(--accent)}
.fhead span{font-size:14px;font-weight:650}
.fthumbs{display:flex;gap:6px}
.fthumbs img{width:64px;height:64px;object-fit:cover;border-radius:4px;
  border:1px solid var(--rule);background:var(--panel)}
.fcard table{font-size:12.5px}
.fcard td{padding:2px 6px 2px 0;border:0}
.fcard td:first-child{width:58%;color:var(--ink3);white-space:normal}
.bm-grid.wide{max-height:none;grid-template-columns:repeat(auto-fill,minmax(64px,1fr))}
.bm.flip img{border-color:var(--accent);border-width:3px}
.bm figcaption.two{position:static;background:none;color:var(--ink3);
  font-family:var(--mono);font-size:9.5px;line-height:1.25;padding-top:2px;
  text-align:center}
.bm figcaption.two b{color:var(--accent);font-size:10.5px}

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
table.tax th{font-size:11px;font-family:var(--mono);letter-spacing:.06em;
  color:var(--ink3);text-align:left;font-weight:600;padding:0 10px 6px 0;
  border-bottom:1px solid var(--rule);white-space:nowrap}
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

/* ── 진행 레일 ─────────────────────────────────────────────────────── */
.nav{position:sticky;top:0;z-index:20;margin:0 -22px;padding:10px 22px 9px;
  background:var(--bg);border-bottom:1px solid var(--rule);
  display:flex;flex-direction:column;gap:8px}
.nav-label{font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;
  text-transform:uppercase;color:var(--ink3);margin-right:2px}
.rail{list-style:none;margin:0;padding:0;display:flex;gap:4px;flex-wrap:wrap}
.rail li{flex:1 1 84px;min-width:0;display:flex}
/* 마디 이름. 열 칸이 평평하게 나열되면 어디까지가 규명이고 어디부터가
   조치인지 안 보인다. */
.rail li.ph{flex:0 0 auto;align-items:center;padding:0 3px 0 5px}
.rail li.ph span{font-family:var(--mono);font-size:9.5px;letter-spacing:.04em;
  color:var(--ink3);writing-mode:vertical-rl;text-orientation:mixed;
  max-height:38px;overflow:hidden;border-left:2px solid var(--rule);padding-left:3px}
.phase{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
  margin:6px 0 -2px;padding:0 2px}
.phase b{font-size:15px;letter-spacing:-.01em}
.phase span{font-size:13px;color:var(--ink3)}
.rail a{flex:1;min-width:0;display:flex;flex-direction:column;gap:1px;
  text-decoration:none;color:var(--ink2);background:var(--panel);
  border:1px solid var(--rule2);border-top:3px solid var(--rule);
  border-radius:4px;padding:4px 7px 5px}
.rail a:hover{border-color:var(--accent);color:var(--accent)}
.rail .no{font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;
  color:var(--ink3)}
.rail .nm{font-size:11.5px;line-height:1.25;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis}
.rail .done{border-top-color:var(--ok)}
.rail .done .no{color:var(--ok)}
.rail .blocked{border-top-color:var(--stop);background:var(--stop-bg)}
.rail .blocked .no{color:var(--stop)}
.rail .skipped{border-top-color:var(--skip);background:var(--skip-bg)}
.rail .skipped .no{color:var(--skip)}
/* 미도달은 흐리게. **지우지 않는다** — 여섯 원인 중 넷은 재구성이 답이 아니라
   중간에 멈추는 것이 맞는 동작이고, 남은 칸이 회색으로 보이는 것 자체가
   보여줄 것이다. 실행된 칸만 그리면 그 판단이 화면에서 사라진다. */
.rail .pending{opacity:.45;border-style:dashed}
.rail .at{outline:2px solid var(--accent);outline-offset:1px}
.rail .at .nm{color:var(--accent);font-weight:650}
.rail em{font-style:normal;color:var(--accent);font-weight:700;margin-left:4px}
.jump{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.jump a{font-family:var(--mono);font-size:11.5px;line-height:1.35;
  text-decoration:none;color:var(--accent);background:var(--panel);
  border:1px solid var(--rule2);border-left:3px solid var(--accent);
  border-radius:4px;padding:3px 9px;white-space:nowrap;font-weight:650}
.jump a:hover{border-color:var(--accent)}

/* 앵커로 뛰었을 때 이동 바에 제목이 가리지 않게 */
.stage,.evidence,.sim,.doc{scroll-margin-top:96px}

/* ── 멈춘 이유는 그 자리에 ─────────────────────────────────────────── */
/* 지금까지 중단 사유는 화면 맨 위 배너에만 있었다. 세로 7,000px 짜리 화면에서
   맨 위와 멈춘 자리는 서로 안 보인다. 그래서 단계 안으로 내린다. */
.why{background:var(--panel2);border:1px solid var(--rule2);
  border-left:3px solid var(--stop);border-radius:5px;padding:10px 13px}
.stage.skipped .why{border-left-color:var(--skip)}
.why b{display:block;font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink3);margin-bottom:3px;font-weight:600}
.why p{margin:0;color:var(--ink);font-size:14.5px}
.stage.thin{padding:11px 18px;gap:7px}
.stage.thin .stage-title{font-size:14px;font-weight:600;color:var(--ink3)}
.stage.pending{opacity:.72;border-style:dashed;border-left-style:solid}
.chip.pending{color:var(--ink3);background:var(--panel2)}
.halt{background:var(--panel);border:1px solid var(--rule);
  border-left:3px solid var(--stop);border-radius:8px;padding:16px 18px;
  display:flex;flex-direction:column;gap:9px}
.halt.by-design{border-left-color:var(--accent)}
.halt-head{display:flex;align-items:baseline;justify-content:space-between;
  gap:12px;flex-wrap:wrap}
.halt-head b{font-size:15.5px}
.halt-head span{font-family:var(--mono);font-size:11.5px;color:var(--ink3)}
.halt p{margin:0;font-size:14.5px;color:var(--ink2)}
.halt .rest{font-family:var(--mono);font-size:12px;color:var(--ink3)}
.halt .fail{font-family:var(--mono);font-size:12.5px;color:var(--stop);
  background:var(--stop-bg);border-radius:4px;padding:6px 10px}

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

/* ── 영어 이름은 지우지 않고 아래에 작게 남긴다 ─────────────────────── */
/* 한국어로 갈아치우면 모델이 실제로 부른 도구 이름과 화면이 달라져, 호출
   기록이라는 성격 자체가 없어진다. 그래서 병기한다. */
small{display:block;font-family:var(--mono);font-size:10.5px;color:var(--ink3);
  font-weight:400;letter-spacing:0;margin-top:1px}
td:first-child{white-space:normal}
/* 뜻풀이가 붙은 낱말. 밑줄만으로는 누를 수 있다는 것이 안 보인다. */
abbr{text-decoration:none;border-bottom:1px dotted var(--accent);cursor:help}

/* ── 설정값 ────────────────────────────────────────────────────────── */
/* 판정이 어떤 숫자 위에 서 있는지가 화면 어디에도 없었다. 값만 적으면
   "누가 정한 값인가"를 못 물으므로 출처와 소유자를 함께 적는다. */
table.cfg td{font-size:13.5px}
table.cfg td:first-child{width:38%}
table.cfg td:nth-child(2){font-family:var(--mono);font-weight:650;color:var(--ink);
  white-space:nowrap;width:16%}
.src{font-family:var(--mono);font-size:10.5px;padding:2px 8px;border-radius:3px;
  white-space:nowrap}
.src.editable{color:var(--accent);background:var(--panel2)}
.src.fixed{color:var(--ink3);background:var(--panel2)}
.src.derived{color:var(--skip);background:var(--skip-bg)}
/* 아직 확인 안 된 자리. 비워 두면 "안 적었다"인지 "없다"인지 갈리지 않는다. */
.pend{color:var(--ink3);font-style:italic}

/* ── 실행 중 덮개 ──────────────────────────────────────────────────── */
/* 열 단계가 한 덩어리로 돌고 끝나야 화면이 뜬다. 실모델에서는 몇 분 동안
   흰 화면이라 고장으로 읽힌다. **진행률을 지어내지 않는다** — 어디까지
   갔는지는 파이프라인만 알고, 화면이 아는 것은 경과 시간뿐이다. */
.veil{position:fixed;inset:0;z-index:99;background:var(--bg);
  display:none;align-items:center;justify-content:center;padding:22px}
.veil.on{display:flex}
.veil-box{background:var(--panel);border:1px solid var(--rule);border-radius:10px;
  padding:26px 28px;max-width:560px;width:100%;display:flex;
  flex-direction:column;gap:15px}
.veil-box h2{margin:0;font-size:18px;display:flex;align-items:center;gap:10px}
.spin{width:13px;height:13px;border-radius:50%;background:var(--accent);
  animation:pulse 1.1s ease-in-out infinite;flex:0 0 auto}
@keyframes pulse{0%,100%{opacity:.25;transform:scale(.8)}50%{opacity:1;transform:scale(1)}}
@media (prefers-reduced-motion:reduce){.spin{animation:none;opacity:.8}}
.veil .rail a{pointer-events:none}
.elapsed{font-family:var(--mono);font-size:30px;font-variant-numeric:tabular-nums;
  color:var(--accent);line-height:1}
.veil p{margin:0;font-size:13.5px;color:var(--ink2)}
.veil .stopwarn{background:var(--stop-bg);color:var(--stop);border-radius:6px;
  padding:11px 14px;font-size:13.5px;font-weight:600}
.nostop{font-size:12.5px;color:var(--stop);font-weight:600}
"""


#: 표를 접지 않는 단계. 판별 항목과 진단은 근거 자체라 접으면 볼 것이 없다.
OPEN_STAGES = {"evidence", "diagnose"}

#: 도구 이름 → (단계 key, 진행 레일에 쓸 짧은 이름).
#:
#: **순서는 여기서 정하지 않는다** — `FALLBACK_SEQUENCE` 가 정하고 이 표는
#: 이름만 붙인다. 짧은 이름은 레일 전용이며, 실행된 단계의 이름은 언제나
#: `stage.title`(파이프라인이 붙인 것)이라 단계 이름이 두 벌이 되지 않는다.
#: 짧은 이름이 필요한 자리는 **아직 실행되지 않아 Stage 가 없는 칸** 하나다.
#:
#: `lookup_ontology` 는 여기 없다. 단계가 아니라 순서 제약 없는 조회이고,
#: 고정 순서 재생 목록에도 들어 있지 않다.
#: 값은 (단계 key, 짧은 이름, 이 단계가 무엇을 하는가 한 줄).
STEP_NAMES: dict[str, tuple[str, str, str]] = {
    "intake_issue": ("intake", "이슈 접수",
                     "올라온 글에서 라인·품목을 뽑고, 정보가 모자라면 되묻습니다"),
    "lookup_mes": ("mes", "생산 정보 조회",
                   "제품명·로트로 검사할 이미지를 찾고, 그 품목의 뱅크를 확인합니다"),
    "run_inspection": ("inspect", "재검사",
                       "찾은 이미지를 뱅크로 판정해 미검·과검을 가려냅니다"),
    "run_checks": ("evidence", "근거 수집",
                   "원인을 구분하기 위한 일곱 가지를 측정합니다"),
    "diagnose_issue": ("diagnose", "원인 규명",
                       "일곱 가지를 모아 원인 6종 중 하나로 규명합니다"),
    "plan_curation": ("curate", "데이터 선별",
                      "뱅크에서 무엇을 빼고 무엇을 채울지 정합니다"),
    "rebuild_bank": ("rebuild", "뱅크 재구성",
                     "계획대로 새 뱅크를 만듭니다. 배포하지 않습니다"),
    "evaluate_gate": ("gate", "성능 검증",
                      "새 뱅크가 배포 후보가 될 만한지 기준과 대조합니다"),
    "shadow_compare": ("shadow", "신구 비교",
                       "새 뱅크를 판정에 쓰지 않고 나란히 돌려, 판정이 서로 다른 것만 뽑습니다"),
    "prepare_release": ("release", "승인 요청",
                        "배포 패키지와 승인 요청 문서를 만듭니다. 배포는 사람이 결정합니다"),
}

#: 화면이 훑을 열 단계. (단계 key, 짧은 이름, 도구 이름).
PIPELINE_STEPS: list[tuple[str, str, str]] = [
    (STEP_NAMES[name][0], STEP_NAMES[name][1], name)
    for name, _ in FALLBACK_SEQUENCE if name in STEP_NAMES
]


#: 열 단계를 세 마디로 묶는다. (마지막 단계 번호, 마디 이름, 한 줄).
#:
#: 평평하게 열 개가 나열되면 **어디까지가 규명이고 어디부터가 조치인지** 안
#: 보인다. 이 과제의 논지가 "재구성이 답이 아니면 멈춘다"인데, 규명과 조치가
#: 갈려 있지 않으면 멈춘 자리가 무슨 뜻인지도 안 읽힌다.
PHASES: list[tuple[int, str, str]] = [
    (5, "무엇이 문제인가", "접수부터 원인 규명까지. 여기까지는 뱅크를 건드리지 않습니다"),
    (7, "무엇을 할 것인가", "원인이 정해졌습니다. 이 원인에서 재구성이 답인지부터 판단합니다"),
    (10, "그것이 맞는가", "새 뱅크가 나아졌는지 검증합니다. 배포는 사람이 결정합니다"),
]


def _phase_of(number: int) -> tuple[int, str, str] | None:
    return next((ph for ph in PHASES if number <= ph[0]), None)


def _step_views(outcome: RunOutcome) -> list[tuple[int, str, str, Stage | None, str, str]]:
    """열 단계를 순서대로 훑는다 — (번호, key, 짧은 이름, Stage, 상태, 오류).

    **값을 만들지 않는다.** 상태는 Stage 가 있으면 그 status 를 그대로 쓰고,
    Stage 가 없으면 도구 호출 기록에 남은 성공·실패를 읽고, 그것도 없으면
    미도달이다. 도구가 앞 단계를 요구하며 거절하면 Stage 가 안 생기므로
    (`rebuild_bank` 가 그렇다) 호출 기록을 함께 봐야 그 자리가 왜 비었는지
    화면이 말할 수 있다.
    """
    stages = {s.key: s for s in outcome.stages}
    last_call: dict[str, str] = {}
    for name, status in outcome.tool_trace:
        last_call[name] = status

    views: list[tuple[int, str, str, Stage | None, str, str]] = []
    for number, (key, short, tool) in enumerate(PIPELINE_STEPS, start=1):
        stage = stages.get(key)
        error = ""
        if stage is not None:
            status = stage.status
        else:
            call = last_call.get(tool, "")
            status, error = ("blocked", call) if call.startswith("실패") else ("pending", "")
        views.append((number, key, short, stage, status, error))

    # 표에 없는 단계가 생기면 뒤에 붙인다. 조용히 빠뜨리면 실제로 돈 단계를
    # 화면이 "미도달"로 그려 거짓말을 하게 된다.
    known = {key for key, _, _ in PIPELINE_STEPS}
    extra = [s for s in outcome.stages if s.key not in known]
    for offset, stage in enumerate(extra, start=len(views) + 1):
        views.append((offset, stage.key, stage.title, stage, stage.status, ""))
    return views


def _mark(value: str) -> str:
    """앞머리의 확인 표시에만 색을 준다. 나머지 본문은 그대로 escape 한다.

    판별 7항목에서 **확인하지 못한 항목이 눈에 띄어야** 한다. ○ 와 × 가 같은
    먹색이면 6/7 이라는 숫자를 읽기 전까지 어느 항목이 빈 자리인지 모른다.
    """
    text = str(value)
    for sign, kind in (("○", "yes"), ("×", "no")):
        if text.startswith(sign):
            body = _gloss(_value_ko(escape(text[1:])))
            return f'<span class="mark {kind}">{sign}</span>{body}'
    return _gloss(_value_ko(escape(text)))


def _stage_html(stage: Stage) -> str:
    rows = "".join(
        f"<tr><td>{_row_label(k)}</td><td>{_mark(v)}</td></tr>" for k, v in stage.rows
    )
    body = (f"<table>{rows}</table>" if rows else "") + (
        f'<p class="note">{_bold(_gloss(_value_ko(escape(stage.note))))}</p>' if stage.note else ""
    )

    # 부차 단계는 표만 접는다. 제목·판정·한 줄 요약은 남으므로 접힌 상태에서도
    # "무엇이 어디까지 돌았는가"가 읽히고, 앵커로 뛰어도 빈 자리에 떨어지지 않는다.
    #
    # **멈춘 단계는 접지 않는다.** 거기 실린 표가 곧 멈춘 근거다 — 게이트라면
    # 어느 지표가 미달인지, 큐레이션이라면 대신 무엇을 하라는 것인지가 표에
    # 있고, 그것을 접으면 "왜 멈췄나"가 한 줄 요약으로만 남는다.
    if body and stage.key not in OPEN_STAGES and stage.status == "done":
        body = f"<details><summary>자세히</summary>{body}</details>"

    # 멈춘 이유는 다른 단계의 설명문과 같은 먹색으로 두지 않는다. 화면에서
    # 제일 먼저 찾는 것이 "어디서 왜 멈췄나"인데 지금은 그것이 done 단계의
    # 부연과 구분되지 않아 스크롤로 찾아야 했다.
    # 한 줄 요약이 이유를 통째로 품고 있으면 같은 문장이 두 번 나온다.
    # 큐레이션이 뱅크를 안 건드리기로 한 화면이 그렇다 — `summary()` 가
    # `reason` 을 그대로 담아서, 제목 아래와 「왜 건드리지 않았나」 상자에
    # 똑같은 문단이 연달아 찍힌다. **값을 고치는 것이 아니라 겹친 자리만
    # 지운다** — 이유는 이름표가 붙은 상자 쪽에 남는 편이 읽힌다.
    headline = stage.headline
    if stage.detail and len(stage.detail) > 20 and stage.detail in headline:
        headline = headline.replace(stage.detail, "").rstrip(" :·,")

    why = WHY_LABEL.get(stage.status)
    if stage.detail and why:
        detail = (f'<div class="why"><b>{why}</b>'
                  f'<p>{_bold(_gloss(_value_ko(escape(stage.detail))))}</p></div>')
    elif stage.detail:
        detail = f'<p class="detail">{_bold(_gloss(_value_ko(escape(stage.detail))))}</p>'
    else:
        detail = ""

    return f"""
    <section class="stage {stage.status}" id="stage-{escape(stage.key)}">
      <div class="stage-head">
        <span class="stage-title">{_gloss(escape(stage.title))}</span>
        <span class="chip {stage.status}">{STATUS_LABEL.get(stage.status, stage.status)}</span>
      </div>
      {f'<div class="headline">{_bold(_gloss(_value_ko(escape(headline))))}</div>' if headline else ''}
      {detail}
      {body}
    </section>
    """


def _unreached_html(number: int, key: str, short: str, error: str) -> str:
    """실행되지 않은 단계 한 칸. **빈 자리로 두지 않는다.**

    레일이 가리키는 자리라 없으면 눌러도 아무 데도 안 가고, 무엇보다 "열
    단계 중 여기서 멈췄다"가 화면에서 읽히지 않는다. 도구가 앞 단계를
    요구하며 거절했으면 그 거절 문구를 이 자리에 적는다 — 위쪽 배너까지
    올라가 보지 않아도 왜 안 돌았는지 알 수 있어야 한다.
    """
    kind = "blocked" if error else "pending"
    return f"""
    <section class="stage {kind} thin" id="stage-{escape(key)}">
      <div class="stage-head">
        <span class="stage-title">{number}. {escape(short)}</span>
        <span class="chip {kind}">{STATUS_LABEL[kind]}</span>
      </div>
      {f'<div class="why"><b>도구가 거절했습니다</b><p>{escape(error)}</p></div>' if error else ''}
    </section>
    """


def _halt_html(outcome: RunOutcome, views: list) -> str:
    """어디서 왜 멈췄는가 — 마지막으로 실행된 단계 바로 뒤에 놓는다.

    이 문장은 지금까지 화면 맨 위 배너 안에만 있었다. 전 구간 화면이 세로로
    7,000px 이라 맨 위와 멈춘 자리는 서로 안 보인다.

    **멈춤이 곧 고장은 아니다.** 여섯 원인 중 넷은 뱅크 재구성이 답이 아니고,
    그때 큐레이션이 뱅크를 건드리지 않기로 하면 파이프라인은 거기서 끝난다.
    그것은 설계대로 동작한 것이라 색을 달리 쓴다.
    """
    pending = [v for v in views if v[4] == "pending"]
    if not pending:
        return ""

    ran = [v for v in views if v[4] != "pending"]
    last = ran[-1] if ran else None
    by_design = last is not None and last[4] == "skipped"

    reason = outcome.agent_run.stopped_reason if outcome.agent_run else ""
    failed = [f"{name}: {status}" for name, status in outcome.tool_trace
              if status.startswith("실패")]
    rest = ", ".join(f"{v[0]}. {v[2]}" for v in pending)

    design_note = (
        "<p>재구성이 답이 아닌 원인이라 여기서 끝난 것입니다. "
        "<strong>여섯 원인 중 넷은 뱅크를 다시 만드는 것이 답이 아닙니다.</strong> "
        "멈춘 것이 결론이며 고장이 아닙니다.</p>"
        if by_design else ""
    )
    return f"""
    <div class="halt{' by-design' if by_design else ''}" id="block-halt">
      <div class="halt-head">
        <b>{escape(f'열 단계 중 {len(ran)}단계를 실행했습니다')}</b>
        <span>{escape(f'{len(pending)}단계 미도달')}</span>
      </div>
      {f'<p>{escape(reason)}</p>' if reason else ''}
      {design_note}
      {''.join(f'<p class="fail">{escape(f)}</p>' for f in failed)}
      <p class="rest">{escape(f'남은 단계: {rest}')}</p>
    </div>
    """


def _nav_html(views: list, blocks: list[tuple[str, str]]) -> str:
    """진행 레일 — 열 단계 중 지금 어디이고 무엇이 멈췄는가.

    전 구간 화면은 세로로 매우 길다(실측 7,466px). 단계 표가 세로로 쌓이기만
    하면 어디까지 갔는지를 스크롤해서 세어야 하고, 시연 중에는 그 시간이
    그대로 발표를 잡아먹는다. **자바스크립트를 붙이지 않는다** — 앵커와
    position:sticky 로 되는 일이고, 화면 하나짜리 시연에 스크립트를 더하면
    깨질 자리만 늘어난다.

    칸 위쪽 색이 상태이고, 실행되지 않은 칸은 흐리게 남긴다. 넉 칸만 그리고
    나머지를 지우면 "여기서 멈추는 것이 맞다"는 판단이 화면에서 사라진다.

    아래 줄은 논거 블록으로 가는 자리다. 단계와 같은 줄에 섞여 있으면 열
    단계가 몇 개인지부터 안 읽힌다.
    """
    if not views:
        return ""

    ran = [i for i, v in enumerate(views) if v[4] != "pending"]
    at = ran[-1] if ran else -1

    items = ""
    seen_phase = None
    for index, (number, key, short, stage, status, _error) in enumerate(views):
        phase = _phase_of(number)
        if phase is not None and phase is not seen_phase:
            seen_phase = phase
            items += f'<li class="ph"><span>{escape(phase[1])}</span></li>'
        title = stage.title if stage is not None else short
        here = ' <em>지금</em>' if index == at else ""
        items += (
            f'<li><a class="step {status}{" at" if index == at else ""}"'
            f' href="#stage-{escape(key)}" title="{escape(title)}">'
            f'<span class="no">{number}{here}</span>'
            f'<span class="nm">{escape(short)}</span></a></li>'
        )

    jump = "".join(f'<a href="#{anchor}">{escape(label)}</a>' for anchor, label in blocks)
    return (
        '<nav class="nav">'
        f'<ol class="rail">{items}</ol>'
        + (f'<div class="jump"><span class="nav-label">근거</span>{jump}</div>' if jump else "")
        + "</nav>"
    )


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
        "이번 실행에서는 조회되지 않았습니다. 물어볼 언어 모델이 없으면 부르지 않습니다"
    )

    rebuild_causes = [CAUSES[c].label for c in cause_names()
                      if CAUSES[c].to_dict()["requires_bank_rebuild"]]
    return f"""
    <div class="evidence" id="block-taxonomy">
      <div class="ev-head">
        <span class="stage-title">원인 6종과 판별 기준</span>
        <span class="kind schema">스키마 조회</span>
      </div>
      <table class="tax">
        <thead><tr><th>미검출 원인</th><th>뱅크를 다시 만들어야 하나</th>
          <th>무엇으로 구분하나</th><th>하면 안 되는 조치</th></tr></thead>
        {rows}</table>
      <p class="detail">
        원인 {len(cause_names())}종 중 <strong>뱅크 재구성이 답인 것은
        {len(rebuild_causes)}종뿐</strong>입니다({escape(", ".join(rebuild_causes))}).
        나머지는 다시 만들어도 해결되지 않거나 오히려 나빠집니다.
        <strong>판별 5번이 결함이면 뱅크 오염으로 정해집니다.</strong>
        진짜 정상품이면 커버리지 부족·임계값 문제·정상 분포 중첩으로 나뉘고,
        판별 6번과 임계값 스윕이 그것을 가릅니다. <strong>뱅크 오염과 정상 분포
        중첩은 조치가 정반대라, 5번을 얻지 못하면 판정하지 않습니다.</strong>
      </p>
      <p class="note">
        언어 모델은 이 표를 <code>lookup_ontology</code> 도구로 <strong>읽을 수만</strong>
        있습니다. <strong>이 조회는 원인을 정하지 않습니다.</strong> 판정은 판별
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
    원인 규명이 아니라 유사 사례 찾기가 된다. 그래프는 "이미 답이 나온 일인가"만 묻는다.
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
        f"<strong>중복이라 여기서 끊었습니다.</strong> {escape(intake.duplicate_of or '')} 과 "
        f"같은 라인·같은 증상이며 이미 조치가 끝났습니다. 진단하지 않습니다."
        if blocked else
        "<strong>중복이 아니라 진행합니다.</strong> 유사도가 높은 건도 "
        "<em>라인이 다릅니다.</em> 라인마다 뱅크가 따로이므로 1라인 뱅크에 "
        "오염이 있다고 2라인도 그렇다는 뜻이 아닙니다. 관련 사례로만 넘깁니다."
    )
    return f"""
    <div class="evidence" id="block-ontology">
      <div class="ev-head">
        <span class="stage-title">이슈 이력 그래프 검색</span>
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
        정하지 않습니다.</strong> 과거가 비슷하다고 이번 원인을 그것으로 정하면
        원인 규명이 아니라 유사 사례 찾기가 됩니다. 원인은 판별 7항목으로 매번 새로 규명합니다.
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

    # 칸 사이 간격과 모서리는 **격자 크기를 따라간다.** 상자 안쪽이 284px 인데
    # 64칸이면 칸 하나가 4.4px 다. 거기에 3px 간격을 주면 간격이 칸보다 넓어져
    # 히트맵이 아니라 세로줄 무늬가 된다(4090 화면이 그랬다). 모서리 둥글리기도
    # 4px 칸에서는 칸을 동그라미로 만든다.
    cell_gap = 3 if grid_w <= 16 else 2 if grid_w <= 32 else 1 if grid_w <= 48 else 0

    # 한 칸이 얼마나 큰가. **격자 수만 적으면 "64×64" 가 무슨 크기인지 안
    # 잡힌다.** 화소로 적으려면 원본을 열어 크기를 재야 하는데, 화면이 파일을
    # 읽는 것은 이 자리가 할 일이 아니다. **비율은 파일 없이 정확하다.**
    cell_px = f"가로 1/{grid_w} · 세로 1/{grid_h}" if grid_w and grid_h else ""
    cell_radius = 2 if cell_gap >= 2 else 0

    # 역추적이 지목한 칸에만 붙는 표시. f-string 식 안에 역슬래시를 두면
    # Python 3.11 에서 SyntaxError 다(3.12 의 PEP 701 부터 허용). 대상 환경이
    # 3.11 이므로 따옴표를 밖으로 뺀다.
    hot_attr = " class='hot'"

    # 칸마다 붙는 툴팁을 **큰 격자에서는 빼 준다.** 64x64 면 4,096개라 HTML 이
    # 209KB 가 되는데, 칸이 4.4px 라 애초에 마우스로 짚을 수 없다. 짚을 수 없는
    # 설명에 페이지 무게의 절반을 쓰는 셈이다. 역추적이 지목한 칸은 하나뿐이니
    # 그것만 남긴다.
    def cell(r: int, c: int, v: float) -> str:
        hot = bool(top and r == top.query.row and c == top.query.col)
        tip = f' title="({r},{c}) {v:.4f}"' if hot or cell_gap else ""
        return (f'<i style="opacity:{(v - lo) / span:.3f}"'
                f'{hot_attr if hot else ""}{tip}></i>')

    cells = "".join(
        cell(r, c, v)
        for r, row in enumerate(result.patch_distances)
        for c, v in enumerate(row)
    )

    threshold = outcome.threshold or 1.0
    verdict = "검출" if result.score >= threshold else "미검"
    # 눈금은 점수와 임계값 중 큰 쪽에 여유를 둔 길이다. **그림의 길이일 뿐
    # 값이 아니다** — 점수도 임계값도 숫자를 그대로 함께 적는다. 전에는 임계값을
    # 눈금 끝으로 잡아, 임계값을 넘긴 점수가 막대 끝에서 잘려 얼마나 넘겼는지
    # 안 보였다.
    span_max = max(result.score, threshold) * 1.15 or 1.0
    score_at = min(result.score / span_max, 1.0) * 100
    thr_at = min(threshold / span_max, 1.0) * 100

    def crop_url(image: str, row: int, col: int) -> str:
        query = urlencode({"row": row, "col": col, "grid_h": grid_h,
                           "grid_w": grid_w, "margin": 24})
        return f"/crop/{quote(image)}?{query}"

    # 판별 5번은 **화면이 다시 판정하지 않는다.** 4번 단계가 이미 낸 값을
    # 그 자리에 옮겨 적을 뿐이다. 뱅크 크롭 옆이 그 값이 있어야 할 자리다 —
    # 이 한 값으로 원인이 뱅크 오염과 정상 분포 중첩으로 갈린다.
    check5 = ""
    stage = next((s for s in outcome.stages if s.key == "evidence"), None)
    if stage is not None:
        check5 = next((str(v) for k, v in stage.rows if str(k).startswith("5.")), "")

    traced = ""
    if top:
        q, b = top.query, top.bank
        badge = f"<b>판별 5번 {_value_ko(escape(check5))}</b>" if check5 else ""
        traced = f"""
        <div class="pair">
          <figure>
            <img src="{escape(crop_url(outcome.query_image, q.row, q.col))}" alt="질의 패치">
            <figcaption>질의 패치 ({q.row},{q.col})<br><span>못 잡은 이미지의 이 자리</span></figcaption>
          </figure>
          <div class="arrow">최근접<br><b>{top.distance:.4f}</b></div>
          <figure class="bank">
            <img src="{escape(crop_url(b.source_image, b.row, b.col))}" alt="뱅크 패치">
            <figcaption>뱅크 패치 ({b.row},{b.col})<br>
              <span>{escape(Path(b.source_image).name)}</span>{badge}</figcaption>
          </figure>
        </div>
        """

    # 시연에서 판별 5번을 손으로 지정했으면 그렇다고 적는다. 지우면 모델이
    # 판독한 값으로 보이고, 그 한 값이 원인을 정반대로 돌린다.
    override = ""
    if outcome.patch_override:
        override = (
            f'<p class="note">판별 5번은 <strong>시연을 위해 지정한 값</strong>입니다'
            f'({_value_ko(escape(outcome.patch_override))}). 역추적이 가리킨 자리를'
            f' 시각 언어 모델에 물으려면 "모델에게 묻기"로 다시 실행하세요.</p>'
        )

    # 같은 이미지가 신규 뱅크에서 어떻게 갈렸는가. 섀도가 실제로 낸 값이고,
    # 없으면(섀도까지 못 갔거나 그 이미지가 대상이 아니면) 적지 않는다.
    after = ""
    shadow = outcome.shadow
    if shadow is not None:
        case = next((c for c in shadow.cases if c.image == outcome.query_image), None)
        if case is not None and not case.agreed:
            after = (
                f'<p class="after">재구성 뒤 이 이미지는 '
                f'<strong>{case.current_score:.3f} → {case.candidate_score:.3f}</strong> '
                f'({_value_ko(escape(case.current_verdict))} → '
                f'{_value_ko(escape(case.candidate_verdict))}, '
                f'임계값 {shadow.current_threshold:.2f} → {shadow.candidate_threshold:.2f}). '
                f'신구 비교가 실제로 낸 값입니다.</p>'
            )

    return f"""
    <div class="evidence" id="block-evidence">
      <div class="ev-head">
        <span class="stage-title">진단 근거: 히트맵과 역추적 패치</span>
        <span class="sim-state">{escape(outcome.bank_version)}</span>
      </div>
      <div class="ev-grid">
        <div>
          <div class="ev-label">이상 점수 히트맵 · {grid_h}×{grid_w}</div>
          <div class="heat" style="grid-template-columns:repeat({grid_w},1fr);
               --cell-gap:{cell_gap}px;--cell-r:{cell_radius}px">{cells}</div>
          <div class="scale"><span>닮음</span><i></i><span>덜 닮음</span></div>
          <p class="hint">
            검사한 이미지를 <strong>{grid_h}×{grid_w} 칸</strong>으로 자른 것입니다
            (한 칸이 이미지의 {cell_px}). 칸마다 <strong>뱅크에서 가장 닮은
            정상 패치와 얼마나 먼가</strong>를 재어 색을 칠했습니다. 진할수록
            정상에서 멉니다.
          </p>
          <p class="hint">테두리 친 칸이 가장 먼 자리이고, 아래 두 조각이
             그 칸과 <strong>뱅크에서 그 칸과 가장 닮았던 자리</strong>를
             잘라낸 것입니다.</p>
        </div>
        <div class="wide">
          <div class="ev-label">이상 점수 · 임계값</div>
          <div class="gauge">
            <b style="width:{score_at:.1f}%"></b>
            <i style="left:{score_at:.1f}%" title="이상 점수"></i>
            <u style="left:{thr_at:.1f}%" title="임계값"></u>
          </div>
          <p class="ticks"><span>0</span><span>{span_max:.2f}</span></p>
          <p class="score-num">
            <b>{result.score:.4f}</b> / 임계값 {threshold:.2f}
            <em class="{'over' if verdict == '검출' else 'under'}">{verdict}</em>
          </p>
          <p class="hint">
            임계값 아래라 양품으로 판정됐습니다. <strong>점수가 낮다고 이상이
            없는 것은 아닙니다.</strong> 어느 자리가 높았는지는 왼쪽 히트맵에 나타납니다.
          </p>
          {after}
        </div>
      </div>
      {traced}
      <p class="note">
        역추적한 두 자리를 같은 좌표계로 잘라 나란히 놓은 것입니다.
        <strong>이 뱅크 패치가 결함이면 뱅크 오염 하나로 정해집니다.</strong>
        진짜 정상품이면 커버리지 부족·임계값 문제·정상 분포 중첩 중 하나이고,
        판별 6번과 임계값 스윕이 그것을 가릅니다(판별 5번).
      </p>
      {override}
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
        <span class="stage-title">조회 방식별 호출 기록</span>
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


def _decision_html(outcome: RunOutcome, decision: Any = None) -> str:
    """승인·비승인을 기록한다. **배포하지 않는다.**

    원안은 *"승인 누르면 배포가 완료되었습니다 문구가 뜬다"* 였는데 그것은
    만들지 않는다. `CLAUDE.md` 2번 규칙이고 기획서 전체가 그 경계 위에 서
    있다. 누르면 **누가·언제·왜 승인했는지를 기록하고 전환 명령을 띄운다.**
    실제 반영은 사람이 그 명령을 실행한다.

    **오히려 시연이 세진다** — 승인을 눌렀는데도 배포가 안 되는 것을
    보여주기 때문이다. 버튼이 없으면 "안 만든 것"이고, 있는데 멈추면
    "일부러 멈춘 것"이다.
    """
    # **후보 저장 자리가 없어도 띄운다.** 합성으로 서면 뱅크 저장소가 없어
    # `candidate_path` 가 비는데, 그렇다고 승인 판단이 없어지는 것은 아니다.
    # 여기서 결정하는 것은 파일 자리가 아니라 **새 뱅크 판을 쓸 것인가**다.
    if outcome.package is None:
        return ""

    if decision is not None:
        state = "승인됨 · 반영 대기" if decision.approved else "비승인 · 후보 유지"
        cls = "ok" if decision.approved else "no"
        note = (
            "<p class=\"detail\"><strong>배포는 아직 실행되지 않았습니다.</strong> "
            "아래 명령을 사람이 직접 실행해야 운영 뱅크가 바뀝니다.</p>"
            f"<pre>{escape(decision.command)}</pre>"
            if decision.approved else
            "<p class=\"detail\">후보 뱅크는 지워지지 않고 저장소에 남습니다. "
            "사유가 기록되어 다음 판단의 재료가 됩니다.</p>"
        )
        return f"""
        <div class="evidence" id="block-decision">
          <div class="ev-head">
            <span class="stage-title">승인 기록</span>
            <span class="chip {cls}">{escape(state)}</span>
          </div>
          <table class="cfg">
            <tr><td>결정</td><td>{"승인" if decision.approved else "비승인"}</td><td></td></tr>
            <tr><td>승인자</td><td>{escape(decision.by)}</td><td></td></tr>
            <tr><td>시각</td><td>{escape(decision.at)}</td><td></td></tr>
            <tr><td>사유</td><td>{escape(decision.reason)}</td><td></td></tr>
            <tr><td>문서번호</td><td>{escape(decision.document_no)}</td><td></td></tr>
          </table>
          {f'<p class="detail">이 건이 <strong>이슈 이력에 {escape(decision.issue_id)} 로 남았습니다.</strong> 다음에 같은 라인·같은 증상이 올라오면 접수 단계가 이 기록을 찾아 중복인지 판단합니다.</p>' if decision.issue_id else ''}
          {note}
          <p class="note">
            기록은 덧붙이기만 하며 고치거나 지우지 않습니다. 되돌린 이력도
            그대로 남습니다.
          </p>
        </div>
        """

    return f"""
    <div class="evidence" id="block-decision">
      <div class="ev-head">
        <span class="stage-title">승인 결정</span>
        <span class="chip pending">기록 대기</span>
      </div>
      <p class="detail">
        <strong>이 화면은 배포하지 않습니다.</strong> 승인을 눌러도 운영 뱅크는
        바뀌지 않고, 누가 언제 왜 승인했는지가 기록되며 전환 명령이 표시됩니다.
        실제 반영은 사람이 그 명령을 실행합니다.
      </p>
      <form method="post" action="/decision" class="decide">
        <div class="controls">
          <div><label for="by">승인자</label>
            <input id="by" name="by" placeholder="이름 또는 사번" required></div>
          <div><label for="why">사유</label>
            <input id="why" name="reason" placeholder="예: 근거 확인함. 신구 비교 불일치 1건 검토 완료" required></div>
        </div>
        <div class="controls">
          <button type="submit" name="decision" value="approve">승인</button>
          <button type="submit" name="decision" value="reject" class="ghost">비승인</button>
        </div>
      </form>
      <p class="note">
        승인자와 사유가 없으면 기록하지 않습니다. <strong>누가 왜 승인했는지
        없는 전환은 나중에 되짚을 수 없습니다.</strong>
      </p>
    </div>
    """


def _posix_rel(path: Any, root: Any) -> str:
    """공장 루트 기준 상대 경로를 **항상 슬래시로** 만든다.

    **`str(Path(...))` 를 쓰면 윈도우에서 역슬래시가 나온다.** 시연은
    윈도우에서 돌고 시험은 맥에서 도니, 경로를 문자열로 견주는 자리는 전부
    이 함정이 있다. 맥에서 통과한 것이 증거가 되지 못한다.

    실제로 당했다. 뱅크 구성 화면에서 뺄 이미지에 표시가 **4090 에서만 하나도
    안 붙었다** — `plan.remove` 쪽은 저장할 때부터 슬래시인데 이쪽만
    역슬래시라 비교가 전부 어긋났다. 화면은 "빨간 테두리가 뺄 2장입니다"라고
    적으면서 한 장도 칠하지 않았다.

    `/image/` 주소로도 쓰이므로 역슬래시가 섞이면 이미지 자체가 404 가 된다.
    """
    try:
        return pathlib.Path(path).relative_to(root).as_posix()
    except (ValueError, TypeError):
        return ""


def _posix(path: Any) -> str:
    """비교하려는 쪽도 슬래시로 맞춘다. 한쪽만 고치면 또 어긋난다."""
    return str(path).replace("\\", "/")


def _factory_html(factory: Any = None) -> str:
    """무엇을 검사하는 공장인가 — 첫 화면의 현황판.

    지적을 받았다. *"심사위원이 무엇을 검사하는 공장인지 모르는 채 진단부터
    본다."* 맞다. 라인이 몇 개인지, 무엇을 만드는지, 뱅크가 걸려 있는지를
    모르면 "1라인 pcb1" 이 무슨 말인지 알 수가 없다.

    **읽기 전용이다.** 입력 칸을 미리 채우지 않는다 — 채워 두면 사람이
    그것부터 채우게 되고 언어 모델이 원문에서 뽑을 일이 없어져 자연어 입력이
    장식이 된다(`CLAUDE.md`). 여기는 보여 주기만 한다.
    """
    if factory is None or not getattr(factory, "items", None):
        return ""

    def rel(path) -> str:
        return _posix_rel(path, factory.root)

    cards = ""
    for item in factory.items.values():
        sample = next((r for r in (rel(p) for p in item.bank_normal) if r), "")
        defect = next((r for r in (rel(p) for p in item.holdout_defect) if r), "")
        thumbs = "".join(
            f'<img loading="lazy" src="/image/{quote(img)}" alt="{escape(label)}"'
            f' title="{escape(label)}">'
            for img, label in ((sample, "정상 예시"), (defect, "결함 예시")) if img
        )
        cards += f"""
        <div class="fcard">
          <div class="fhead"><code>{escape(item.line)}</code>
            <span>{escape(item.object_name)}</span></div>
          <div class="fthumbs">{thumbs}</div>
          <table><tr><td>배포된 뱅크</td><td>{escape(item.bank.version)}</td></tr>
            <tr><td>뱅크 정상 이미지</td><td>{len(item.bank_normal)}장</td></tr>
            <tr><td>검증용 정상·결함</td><td>{len(item.holdout_normal)}·{len(item.holdout_defect)}장</td></tr>
          </table>
        </div>"""

    return f"""
    <div class="evidence" id="block-factory">
      <div class="ev-head">
        <span class="stage-title">가상 공장 현황</span>
        <span class="sim-state">라인 {len(factory.items)}개</span>
      </div>
      <div class="fgrid">{cards}</div>
      <p class="note">
        <strong>뱅크는 품목마다 따로입니다.</strong> 캡슐 뱅크로 기판을 판정할
        수 없어서, 이슈가 어느 라인의 어느 품목인지가 먼저 정해져야 합니다.
        아래 접수 칸은 <strong>비워 두는 것이 설계입니다.</strong> 라인과 품목은
        올라온 글에서 언어 모델이 뽑습니다.
      </p>
    </div>
    """


def _bank_html(outcome: RunOutcome, factory: Any = None) -> str:
    """뱅크에 무엇이 들어 있고 무엇을 빼는가.

    지적을 받았다. *"재구성하는데 이미지를 왜 제거하는지 흐름이 이해 안 가다가
    갑자기 제거한다니까 ???? 이런 느낌"*. 맞다. 지금까지 화면은 **"제거 2장"
    이라는 숫자뿐**이었고, 뱅크가 무엇으로 이루어져 있는지 본 적이 없으니
    빼는 것이 무슨 뜻인지 알 수가 없었다.

    뱅크를 통째로 늘어놓고 뺄 것을 표시한다. 그 옆에 못 잡은 이미지를 둔다.
    **뱅크 오염을 한 화면으로 보여주는 자리**다 — 정상이라고 등록된 것들
    사이에 결함이 섞여 있고, 그것이 같은 유형의 불량을 정상으로 끌어당겼다.

    **판정하지 않는다.** 뺄 목록은 `plan.remove` 가 정한 것이고 화면은 그
    목록에 있는 것에 표시만 붙인다.
    """
    plan = outcome.plan
    if factory is None or plan is None or not plan.remove:
        return ""

    # 품목은 **뱅크 판으로 찾는다.** 인테이크가 뽑은 라인·품목은 모델이 자연어를
    # 넣을 수 있어 열쇠로 못 쓴다.
    item = next((it for it in factory.items.values()
                 if it.bank.version == outcome.bank_version), None)
    if item is None:
        return ""

    def rel(path) -> str:
        return _posix_rel(path, factory.root)

    removing = {_posix(c.image) for c in plan.remove}
    reasons = {_posix(c.image): c.reason for c in plan.remove}
    members = [r for r in (rel(p) for p in list(item.bank_normal) + list(item.contaminants)) if r]
    if not members:
        return ""

    # **뺄 것을 앞으로 올린다.** 격자 순서에는 뜻이 없다 — 뱅크 구성은 집합이지
    # 차례가 아니다. 그런데 혼입 이미지가 목록 뒤에 붙는 탓에 뺄 것이 항상
    # 맨 끝으로 갔고, 4090 실측에서 132칸 중 116·117번이라 **스크롤 없이는
    # 똑같은 타일 다섯 줄만 보였다.** 캡션은 "빨간 테두리가 뺄 2장입니다"라고
    # 말하는데 화면에 그것이 없는 상태였다. 심사위원이 스크롤을 내릴 이유가 없다.
    members.sort(key=lambda m: m not in removing)

    cells = "".join(
        f'<figure class="bm{" out" if m in removing else ""}">'
        f'<img loading="lazy" src="/image/{quote(m)}" alt="{escape(Path(m).name)}">'
        f'{"<figcaption>뺀다</figcaption>" if m in removing else ""}</figure>'
        for m in members
    )
    picked = "".join(
        f'<li><code>{escape(Path(img).name)}</code> {_gloss(escape(reasons.get(img, "")))}</li>'
        for img in sorted(removing)
    )
    query = (
        f'<figure class="bm big"><img src="/image/{quote(outcome.query_image)}"'
        f' alt="못 잡은 이미지"><figcaption>못 잡은 이미지</figcaption></figure>'
        if outcome.query_image else ""
    )
    return f"""
    <div class="evidence" id="block-bank">
      <div class="ev-head">
        <span class="stage-title">뱅크 구성과 뺄 이미지</span>
        <span class="sim-state">{escape(outcome.bank_version)} · {len(members)}장</span>
      </div>
      <div class="bank-row">
        {query}
        <div class="arrow">가장 가까웠던<br><b>정상 패치</b></div>
        <div class="bm-grid">{cells}</div>
      </div>
      <p class="detail">
        왼쪽이 검사에서 <strong>양품으로 나간 이미지</strong>이고, 오른쪽이
        그 판정의 기준이 된 <strong>뱅크 {len(members)}장</strong>입니다.
        정상만 들어가야 하는데 <strong>결함이 섞여 있습니다.</strong> 섞인
        것이 같은 유형의 불량을 정상 쪽으로 끌어당겨 못 잡게 만들었습니다.
      </p>
      <ul class="picked">{picked}</ul>
      <p class="note">
        <strong>맨 앞 {len(removing)}장</strong>이 뺄 것이고 빨간 테두리를
        둘렀습니다. 근거는 위
        목록에 있고, <strong>고립도만으로는 뽑지 않습니다.</strong> 정상
        이미지를 잘못 빼면 커버리지 부족을 스스로 만들기 때문입니다.
      </p>
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

    **흘려보내고 끝내지 않는다.** 갈린 건은 아래 목록에 쌓인다. 숫자만 남으면
    "몇 장이 갈렸다"까지이고, 사람이 확인해야 할 것이 무엇인지는 안 남는다.
    """
    shadow = outcome.shadow
    if shadow is None or not shadow.cases:
        return ""

    # 갈린 건의 성격(새로 검출인가 새로 놓침인가)과 신규 뱅크에서의 최근접
    # 이미지는 `disagreements` 에만 있다. 이미지 이름으로 맞춰 붙인다.
    detail = {d.image: d for d in shadow.disagreements}

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
                # 갈린 건은 **왜 갈렸는지**를 함께 들고 간다. 어느 쪽으로
                # 갈렸는지(kind)와 신규 뱅크에서 무엇과 가까웠는지가 그것이다.
                # 없으면 빈 문자열이고, 화면은 없는 것을 지어내지 않는다.
                "kind": detail[c.image].kind if c.image in detail else "",
                "nearest": (
                    Path(detail[c.image].candidate_nearest_image).name
                    if c.image in detail and detail[c.image].candidate_nearest_image
                    else ""
                ),
                # 이번에 진단한 바로 그 이미지인가. 시연에서 제일 먼저 봐야 할 줄이다.
                "focus": c.image == outcome.query_image,
            }
            for c in shadow.cases
        ],
        ensure_ascii=False,
    )
    # ⑨ 숫자 대신 그림으로. **14장을 전부 늘어놓는다** — "14장 중 1장이
    # 갈렸다"는 문장은 그 1장이 무엇인지 안 보여준다.
    #
    # **히트맵은 여기 넣지 않는다.** 64x64 면 한 장이 4,096칸이라 14장이면
    # 57,000 요소다. 지금도 페이지가 4.3MB 인데 그것을 얹으면 못 버틴다.
    # 갈린 것이 왜 갈렸는지는 아래 목록과 진단 근거 블록이 답한다.
    tiles = "".join(
        f'<figure class="bm{"" if c.agreed else " flip"}">'
        f'<img loading="lazy" src="/image/{quote(c.image)}" alt="{escape(Path(c.image).name)}">'
        f'<figcaption class="two">{c.current_score:.2f}<br><b>{c.candidate_score:.2f}</b></figcaption>'
        f'</figure>'
        for c in shadow.cases
    )
    moved = (
        f"임계값 {shadow.current_threshold:.2f} → {shadow.candidate_threshold:.2f}"
        if shadow.current_threshold or shadow.candidate_threshold else ""
    )
    return f"""
    <div class="sim" id="block-simulator">
      <div class="sim-head">
        <span class="sim-title">신구 비교 재생 (가상 라인)</span>
        <span class="sim-state" id="sim-state">대기</span>
      </div>
      <p class="detail">
        <strong>신구 비교(섀도 평가)</strong>입니다.
        새 뱅크 <code>{escape(shadow.candidate_version)}</code> 을 현행
        <code>{escape(shadow.current_version)}</code> 과 나란히 돌립니다.
        {f'<code>{escape(moved)}</code>. ' if moved else ''}
        <strong>신규 뱅크는 실제 판정에 쓰이지 않습니다.</strong>
      </p>
      <div class="bm-grid wide" id="shadow-grid">{tiles}</div>
      <p class="hint">
        <strong>{shadow.total}장을 전부 돌렸습니다.</strong> 위가 현행 뱅크, 아래가 새 뱅크의
        점수입니다. 테두리가 굵은 것이 <strong>판정이 서로 다른 {shadow.review_count}장</strong>이고,
        사람이 확인할 것은 그것뿐입니다.
      </p>
      <div class="belt" id="belt"><div class="scanner"></div></div>
      <div class="bar"><span id="sim-bar"></span></div>
      <div class="tally">
        <div><b id="t-total">0</b><span>검사 완료</span></div>
        <div class="good"><b id="t-caught">0</b><span>새로 잡음</span></div>
        <div class="bad"><b id="t-lost">0</b><span>새로 놓침</span></div>
        <div><b id="t-same">0</b><span>판정 동일</span></div>
      </div>
      <div class="ev-label">판정이 서로 다른 건 (사람이 확인할 목록)</div>
      <div class="flips" id="flips"></div>
      <p class="note" id="sim-note">
        흘러가는 판정은 전부 신구 비교가 실제로 낸 값입니다. 화면은 그것을
        한 장씩 재생합니다.
      </p>
    </div>
    <script>
    (function() {{
      const cases = {cases};
      const belt = document.getElementById("belt");
      const state = document.getElementById("sim-state");
      const bar = document.getElementById("sim-bar");
      const flips = document.getElementById("flips");
      const out = {{
        total: document.getElementById("t-total"),
        caught: document.getElementById("t-caught"),
        lost: document.getElementById("t-lost"),
        same: document.getElementById("t-same"),
      }};
      let done = 0, caught = 0, lost = 0, same = 0, i = 0;

      function text(tag, cls, value) {{
        const el = document.createElement(tag);
        if (cls) el.className = cls;
        el.textContent = value;
        return el;
      }}

      // 갈린 건 한 줄. **여기 적히는 값은 전부 섀도가 낸 것**이고 화면이
      // 더하는 것은 순서와 배치뿐이다. 지어낸 숫자를 끼우면 애니메이션이
      // 근거가 아니라 장식이 된다.
      function addFlip(c) {{
        const up = c.before !== "defect" && c.after === "defect";
        const row = document.createElement("div");
        row.className = "flip-row" + (c.focus ? " focus" : "");

        const img = document.createElement("img");
        img.src = c.src;
        img.alt = c.name;
        row.appendChild(img);

        const txt = document.createElement("div");
        txt.className = "txt";
        txt.appendChild(text("span", "nm",
          c.name + (c.focus ? "  \\u2190 이번에 진단한 그 이미지" : "")));

        const mv = document.createElement("div");
        mv.className = "mv";
        mv.appendChild(text("s", "", (c.before === "defect" ? "불량" : "양품")
          + " " + c.beforeScore));
        mv.appendChild(text("span", "", "  \\u2192  "));
        mv.appendChild(text("b", "", (c.after === "defect" ? "불량" : "양품")
          + " " + c.afterScore));
        txt.appendChild(mv);

        if (c.nearest) {{
          txt.appendChild(text("span", "why2",
            "신규 뱅크에서의 최근접 정상 패치 출처: " + c.nearest));
        }}
        row.appendChild(txt);
        row.appendChild(text("span", "kd " + (up ? "up" : "down"),
          c.kind === "newly_detected" ? "새로 검출"
            : c.kind === "newly_missed" ? "새로 놓침"
            : up ? "새로 검출" : "새로 놓침"));
        flips.appendChild(row);
      }}

      function count(c) {{
        done++;
        if (c.agreed) same++;
        else if (c.before === "pass" && c.after === "defect") caught++;
        else lost++;
        out.total.textContent = done;
        out.caught.textContent = caught;
        out.lost.textContent = lost;
        out.same.textContent = same;
      }}

      function finish() {{
        state.textContent = "검증 완료. 사람 승인 대기";
        document.getElementById("sim-note").textContent =
          "판정이 서로 다른 " + (caught + lost) + "장만 사람이 확인하면 됩니다. " +
          "나머지 " + same + "장은 두 뱅크가 같게 판정했습니다.";
      }}

      function release() {{
        if (i >= cases.length) {{ finish(); return; }}
        const c = cases[i++];
        state.textContent = "신구 비교 중입니다 " + i + "/" + cases.length;

        const piece = document.createElement("div");
        piece.className = "piece " + (c.after === "defect" ? "defect" : "pass")
          + (c.agreed ? "" : " flip");
        piece.style.left = "-80px";
        piece.innerHTML =
          '<img src="' + c.src + '" alt="' + c.name + '">' +
          '<span class="tag">' + (c.after === "defect" ? "불량" : "양품") +
          " " + c.afterScore + "</span>";
        belt.appendChild(piece);

        requestAnimationFrame(function() {{ piece.style.left = "45%"; }});

        setTimeout(function() {{
          count(c);
          if (!c.agreed) addFlip(c);
          bar.style.width = (done / cases.length * 100) + "%";
          piece.style.left = "108%";
          setTimeout(function() {{ piece.remove(); }}, 600);
        }}, 620);

        // 갈린 건은 조금 더 세워 둔다. 눈에 걸려야 왜 갈렸는지를 보게 된다.
        setTimeout(release, c.agreed ? 780 : 1400);
      }}

      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {{
        // 애니메이션을 끈 환경에서는 결과만 즉시 채운다.
        cases.forEach(function(c) {{
          count(c);
          if (!c.agreed) addFlip(c);
        }});
        bar.style.width = "100%";
        finish();
      }} else {{
        setTimeout(release, 300);
      }}
    }})();
    </script>
    """


def _settings_html(outcome: RunOutcome) -> str:
    """이 판정이 어떤 숫자 위에 서 있는가 — 값과 **출처와 소유자**.

    지적을 받았다. *"사용자가 설정할 수 있는 임계값들을 어디에서 보여주고
    조절할 수 있어야 하지 않나."* 맞다. 지금까지 화면 어디에도 없었고, 값이
    안 보이면 "누가 정한 숫자인가"를 물을 수조차 없다.

    **조절은 판정 임계값 하나뿐이다.** 게이트 통과 기준과 판정 기준은
    `data/gate.yaml` · `data/criteria.yaml` 에서 오고 도메인 담당 소유이며, 그
    파일들의 규칙이 **값마다 근거를 함께 적는 것**이다. 화면에서 즉석으로
    바꾸면 근거 없는 숫자가 되어 그 규칙이 무너진다.

    신규 뱅크 임계값은 사람이 정하는 값이 아니라 스윕이 계산해 낸 값이라
    표시만 한다.
    """
    rows = ""

    def row(label: str, value: str, kind: str, source: str) -> str:
        return (f'<tr><td>{_gloss(escape(label))}</td>'
                f'<td>{escape(value)}</td>'
                f'<td><span class="src {kind}">{escape(source)}</span></td></tr>')

    if outcome.threshold:
        rows += row("판정 임계값 (이 값을 넘으면 불량이라 판정)",
                    f"{outcome.threshold:.2f}", "editable", "화면에서 조절")

    shadow = outcome.shadow
    if shadow is not None and shadow.candidate_threshold:
        rows += row("신규 뱅크의 임계값 (전건을 잡는 지점으로 스윕이 계산)",
                    f"{shadow.candidate_threshold:.2f}", "derived", "자동 계산")

    gate = outcome.gate
    if gate is not None:
        for check in gate.checks:
            # `improvement` 는 빼놓는다. 그 기준값은 사람이 정한 설정이 아니라
            # **이전 뱅크의 실측 AUROC** 라, 설정값 표에 넣으면 도메인 담당이 정한
            # 숫자로 읽힌다. 게이트 단계 표에는 그대로 남아 있다.
            if check.name == "improvement":
                continue
            rows += row(ROW_LABEL_KO.get(check.name, check.name),
                        str(check.threshold), "fixed", "data/gate.yaml · 도메인 담당")

    if not rows:
        return ""

    # ── 임계값을 바꾸면 진단 결과가 달라진다. 숨기지 않는다 ─────────────
    #
    # 전에 여기 "임계값을 바꿔도 원인은 그대로입니다" 라고 적어 두었다가
    # **VisA 실측에 반증됐다.** 합성이 실데이터에 반증된 네 번째다.
    #
    # 그런데 **뭉뚱그려 "임계값이 바뀌면 원인도 바뀐다"로 적으면 그것도
    # 틀린다.** 실측이 말하는 것은 그게 아니다 —
    #
    #     1.8  대표 000.JPG · 판별 1번 not_visible          → 보류
    #     2.2  대표 006.JPG · 최근접 001(48,34) · 5번 결함    → 뱅크 오염
    #     3.0  대표 002.JPG · 최근접 001(48,37) · 5번 정상품  → 임계값 문제
    #
    # 임계값이 미검으로 걸리는 장수를 바꾸고(0 → 2 → 3 → 12), 그래서 **대표로
    # 뽑히는 이미지가 달라지고**, 그 이미지의 판별 결과가 다르니 원인도 다르다.
    # **입력이 바뀐 것이지 같은 것에 대한 판정이 흔들린 것이 아니다.**
    # 뭉뚱그리면 우리 진단이 임계값을 따라 흔들리는 것처럼 읽힌다.
    #
    # **1.8 을 "최근접 패치가 진짜 정상품이라 보류"로 적었다가 틀렸다.**
    # `decide()` 는 판별 1번이 `not_visible` 이면 최근접 패치를 읽기 전에
    # 반환한다(`agents/diagnose.py:412`). 거꾸로도 틀린다 — 판별 5번이
    # `genuine_normal` 이면 보류가 아니라 커버리지 부족·임계값 문제·정상 분포
    # 중첩 중 하나로 간다. 3.0 줄이 바로 그것이라, **같은 근거로 다른 결과를
    # 설명하는 화면**이 되어 있었다.
    #
    # ── 이유 절은 경로가 하나로 정해질 때만 적는다 ─────────────────────
    #
    # 그것을 고치면서 3.0 에 "스윕에서 해결 가능하다고 나옴"을 적었는데
    # **그것도 추론이었다.** `threshold` 로 가는 길이 셋이다
    # (`diagnose.py` 488 · 526 · 554). 스윕이 없으면
    # `_threshold_feasibility()` 가 `None` 을 돌려주고 다른 두 길로 가는데,
    # 그 둘은 `confidence=low` 에 `needs_human=True` 라 화면 설명과 정반대
    # 인상이 된다. 어느 길이었는지는 그 실행의 `reasoning` 을 봐야 갈린다.
    #
    # 1.8 과 같은 종류의 오류다 — 코드를 읽고 "이렇게 갔겠지"로 적은 것이다.
    # **관측된 것만 적고, 이유는 경로가 하나뿐일 때만 적는다.**
    # 2.2 의 "최근접 패치가 결함"은 `bank_contamination` 으로 가는 길이
    # 하나뿐이라(458) 코드로 정해진다. 1.8 은 4090 로그에서 판별 1번
    # `not_visible` 을 직접 봤다. 3.0 만 비워 둔다.
    #
    # **"같은 이미지를 놓고 임계값만 바꾸면 원인이 그대로인가"는 아직 안 쟀다.**
    # 원래 문구가 주장하던 것이 사실 이것인데, 이번 측정은 대표 이미지가 함께
    # 움직여 그 질문에 답하지 못한다. 반증된 것이 아니라 미측정이므로 **어느
    # 쪽으로도 적지 않는다.**
    measured = (
        "VisA 실데이터 · 임계값 1.2 / 1.8 / 2.2 / 3.0 · 2026-08-16 측정"
    )
    return f"""
    <div class="evidence" id="block-settings">
      <div class="ev-head">
        <span class="stage-title">판정 기준값과 출처</span>
        <span class="kind schema">설정값</span>
      </div>
      <table class="cfg">{rows}</table>
      <p class="detail">
        <strong>바꿀 수 있는 것은 판정 임계값 하나입니다.</strong> 성능 검증 통과
        기준은 <code>data/gate.yaml</code> 에서 오고 값마다 근거가 함께 적혀
        있습니다. 화면에서 즉석으로 바꾸면 근거 없는 숫자가 됩니다.
      </p>
      <p class="detail">
        임계값을 바꾸면 <strong>진단 결과가 달라집니다.</strong> 다만
        <strong>달라지는 이유는 판정이 흔들려서가 아닙니다.</strong> 임계값은 미검으로 걸리는 장수를 바꾸고, 그래서
        <strong>진단 대상으로 뽑히는 이미지 자체가 바뀝니다.</strong> 다른
        이미지는 최근접 패치가 다르고, 그러면 원인도 당연히 다릅니다.
        <strong>입력이 바뀐 것이지 같은 것에 대한 답이 바뀐 것이 아닙니다.</strong>
      </p>
      <details class="supplement">
        <summary>임계값별 실측 결과</summary>
        <table class="cfg">
          <tr><td>1.2</td><td>진단 미도달</td>
              <td>미검 0장, 과검 75장. 볼 것이 없어 3단계에서 멈춥니다</td></tr>
          <tr><td>1.8</td><td>판정 보류</td>
              <td>미검 2장. 결함이 이미지에서 확인되지 않아 원인을 판정하지 않음</td></tr>
          <tr><td>2.2</td><td>뱅크 오염</td>
              <td>미검 3장. 대표 이미지의 최근접 패치가 결함</td></tr>
          <tr><td>3.0</td><td>임계값 문제</td>
              <td>미검 12장. <span class="pend">어느 경로로 이 원인이 됐는지는
                  확인 중입니다</span></td></tr>
        </table>
        <p class="hint">{escape(measured)} · 자세한 것은
          <code>docs/실험_임계값.md</code></p>
        <p class="hint">
          <strong>1.2 와 1.8 은 둘 다 멈췄지만 이유가 다릅니다.</strong>
          1.2 는 과검이 86%라 진단할 미검이 없어서, 1.8 은 미검을 찾았는데
          그 이미지에서 결함이 확인되지 않아서입니다. 둘 다 고장이 아닙니다. <strong>근거가 모자라면 판정하지 않는다</strong>는
          규칙이 서로 다른 자리에서 나타난 것입니다.
        </p>
      </details>
      <p class="note">
        임계값을 내리면 미검은 줄고 <strong>과검이 늘어납니다.</strong> 올리면
        반대입니다. 어느 쪽으로 옮겨도 <strong>뱅크에 섞여 들어간 결함은 섞인
        채</strong>이고, 증상이 옮겨 다닐 뿐입니다. <strong>임계값 조절로 풀리는 문제가
        아니라는 것이 이 서비스가 있는 이유</strong>입니다.
      </p>
    </div>
    """


def _driver_html(outcome: RunOutcome) -> str:
    """도구 순서를 누가 정했는가.

    모델이 안 붙어 있는데 화면이 아무 말도 하지 않으면 "에이전트가 판단한 것"
    처럼 보인다. 시연에서 가장 오해받기 쉬운 지점이라 위에 못 박아 둔다.
    """
    by_model = outcome.driver == "model"
    label = "언어 모델이 도구 순서를 정했습니다" if by_model else "고정 순서로 실행했습니다"

    # 이 표가 `intake_issue` · `lookup_mes` 로만 적혀 있었다. 무엇을 한 것인지
    # 읽을 수 없다는 지적을 받았다. **원래 이름을 지우지 않고** 한국어를 앞에
    # 놓는다 — 이름을 갈아치우면 모델이 실제로 부른 도구와 화면에 적힌 것이
    # 달라져, 도구 호출 기록이라는 성격 자체가 없어진다.
    trace = "".join(
        f'<tr><td>{i}. {escape(STEP_NAMES.get(name, ("", name, ""))[1])}'
        f'<small>{escape(name)}</small></td>'
        f'<td>{escape(status)}'
        f'{f"<small>{escape(STEP_NAMES[name][2])}</small>" if name in STEP_NAMES else ""}'
        f'</td></tr>'
        for i, (name, status) in enumerate(outcome.tool_trace, start=1)
    )
    stopped = outcome.agent_run.stopped_reason if outcome.agent_run else ""
    return f"""
    <div class="banner{'' if by_model else ' warn'}">
      <strong>{escape(label)}.</strong> {escape(outcome.driver_note)}
      {f'<table>{trace}</table>' if trace else ''}
      {f'<p class="note">{escape(stopped)}</p>' if stopped else ''}
    </div>
    """


def _veil_html(on_visa: bool) -> str:
    """실행 중 덮개 — "고장난 것이 아니라 돌고 있다"를 말해 준다.

    지적을 받았다. *"고장나서 멈춘 것이 아니라 진행 중이라는 것을 보여주고,
    진행 경과도 표현되면 오해가 없겠다."* 지금은 단추를 누르면 흰 화면이고,
    실모델에서는 몇 분이 걸린다. 그동안 아무 반응이 없으면 고장으로 읽는 것이
    당연하다.

    **걸리는 시간을 실측 숫자로 박지 않는다.** `scripts/check_docs.py` 는
    문서만 보고 `.py` 안의 수치는 안 봐서, 재측정하면 그 줄이 오류 없이 낡는다.

    **진행률을 지어내지 않는다.** 지금 몇 번째 단계인지는 `run_pipeline` 이
    끝나야 알 수 있고, 그것을 실시간으로 받으려면 `app/pipeline.py` 를 고쳐야
    한다. 화면이 정직하게 아는 것은 **경과 시간**뿐이라 그것만 센다. 채워지는
    진행 막대를 그려 놓으면 보기에는 좋지만 아무것도 재지 않는 그림이 된다.

    열 단계 이름은 회색으로 미리 보여준다. "무엇이 남았는지"는 알 수 있고,
    실제 상태를 아는 척하지도 않는다.
    """
    steps = "".join(
        f'<li><a class="step pending">'
        f'<span class="no">{number}</span>'
        f'<span class="nm">{escape(short)}</span></a></li>'
        for number, (_key, short, _tool) in enumerate(PIPELINE_STEPS, start=1)
    )
    # 걸리는 시간은 **어림으로 적는다.** 전에는 "실측 151초" 를 그대로 박아
    # 두었는데, `scripts/check_docs.py` 는 문서만 보고 `.py` 안의 수치는 안 봐서
    # 재측정하면 이 줄이 오류 없이 낡는다. 실제로 `마스크로 자르면 9/10` 을
    # 여섯 문서가 근거로 쓰다가 같은 조건으로 다시 재니 5/10 이었던 일이 있다.
    took = (
        "VisA 실데이터입니다. 실모델이 붙어 있으면 보통 <strong>몇 분</strong> 걸립니다."
        if on_visa else
        "합성 이미지입니다. 보통 <strong>수 초</strong> 안에 끝납니다."
    )
    return f"""
    <div class="veil" id="veil">
      <div class="veil-box">
        <h2><span class="spin"></span>실행 중입니다</h2>
        <p><strong>고장이 아닙니다.</strong> 열 단계가 이어서 실행되고 있습니다.
           전부 끝나야 결과 화면이 나타납니다.</p>
        <ol class="rail">{steps}</ol>
        <p class="hint">지금 몇 번째인지는 표시하지 않습니다. 단계별 신호를
           받으려면 파이프라인을 고쳐야 하고, <strong>모르는 것을 아는 척하는
           진행 막대는 만들지 않습니다.</strong></p>
        <div class="elapsed" id="elapsed">0:00</div>
        <p>{took}</p>
        <div class="stopwarn">한 번 시작하면 중단할 수 없습니다.
           창을 닫아도 서버에서는 끝까지 실행됩니다.</div>
      </div>
    </div>
    <script>
    (function() {{
      const form = document.getElementById("runform");
      const veil = document.getElementById("veil");
      const out = document.getElementById("elapsed");
      if (!form || !veil) return;
      form.addEventListener("submit", function() {{
        veil.classList.add("on");
        const began = performance.now();
        setInterval(function() {{
          const s = Math.floor((performance.now() - began) / 1000);
          out.textContent = Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
        }}, 1000);
      }});
      // 뒤로 가기로 돌아왔을 때 덮개가 남아 있으면 화면이 잠긴다.
      window.addEventListener("pageshow", function() {{ veil.classList.remove("on"); }});
    }})();
    </script>
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
                context: dict[str, str] | None = None, on_visa: bool = False,
                threshold: str = "", factory: Any = None, decision: Any = None) -> str:
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
      <summary>{escape('이슈 접수에서 되물었습니다. 값을 채워 주세요' if asked
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
        # ── 열 단계를 canonical 순서로 훑는다 ──────────────────────────────
        #
        # 실행된 것만 그리면 "열 단계 중 어디까지 왔는가"가 화면에 없다.
        # 안 돈 칸도 자리를 지키게 두고, 멈춘 이유는 멈춘 자리에 적는다.
        views = _step_views(outcome)
        stages_html: list[str] = []
        #: 논거 블록으로 가는 자리들. (앵커, 이름)
        blocks: list[tuple[str, str]] = []
        halted = False

        def add_block(html: str, anchor: str, label: str) -> None:
            """논거 블록을 끼우고 이동 바에도 올린다.

            블록은 조건이 안 맞으면 빈 문자열을 돌려준다. 그때 이동 바에만
            남으면 눌러도 아무 데도 안 간다.
            """
            if not html.strip():
                return
            stages_html.append(html)
            blocks.append((anchor, label))

        seen_phase = None
        for number, key, short, stage, status, error in views:
            phase = _phase_of(number)
            if phase is not None and phase is not seen_phase:
                seen_phase = phase
                stages_html.append(
                    f'<div class="phase"><b>{escape(phase[1])}</b>'
                    f'<span>{escape(phase[2])}</span></div>'
                )
            if stage is None:
                # 멈춤 설명은 실행된 마지막 단계 바로 뒤, 미도달 칸이 시작되기
                # 전에 한 번만 놓는다.
                if not halted:
                    halted = True
                    halt = _halt_html(outcome, views)
                    if halt.strip():
                        stages_html.append(halt)
                        blocks.append(("block-halt", "멈춘 자리"))
                stages_html.append(_unreached_html(number, key, short, error))
                continue

            # 시뮬레이터는 섀도 단계 바로 앞에 끼운다. 숫자만 적힌 표보다
            # 무엇이 어떻게 갈렸는지가 먼저 보여야 한다.
            # 뱅크 구성은 데이터 선별 바로 뒤. "무엇을 왜 빼는가"는 계획을
            # 읽은 다음에 와야 순서가 맞다.
            if key == "curate":
                add_block(_bank_html(outcome, factory), "block-bank", "뱅크 구성")
            if key == "shadow":
                add_block(_simulator_html(outcome), "block-simulator", "가상 라인")
            stages_html.append(_stage_html(stage))
            # 진단 바로 뒤에 근거를 그린다. 문장으로만 적으면 확인할 방법이 없다.
            if key == "diagnose":
                add_block(_evidence_visual_html(outcome), "block-evidence", "진단 근거")
                # 근거 다음에 체계를 놓는다. "이 근거가 왜 이 원인이 되는가"는
                # 표를 봐야 답이 되고, 표가 앞에 오면 결론부터 읽게 된다.
                add_block(_taxonomy_html(outcome), "block-taxonomy", "원인 체계")
            if key == "evidence":
                add_block(_retrieval_html(outcome), "block-retrieval", "조회 방식")
                # 설정값도 여기 놓는다. **게이트 뒤가 아니다** — 재구성이 답이
                # 아닌 원인은 게이트까지 가지 않는데, 그런 실행에서도 판정
                # 임계값은 이미 쓰였고 그 값이 화면에 있어야 한다.
                add_block(_settings_html(outcome), "block-settings", "설정값")
            # 그래프는 인테이크 바로 뒤. "이미 답이 나온 일인가"를 묻는 자리다.
            if key == "intake":
                add_block(_ontology_html(outcome), "block-ontology", "이력 그래프")

        add_block(_decision_html(outcome, decision), "block-decision", "승인 결정")

        doc = ""
        if outcome.approval_markdown:
            # 10번 단계 이름이 "승인 요청"이라 문서 쪽은 다르게 적는다.
            blocks.append(("doc-approval", "승인 문서"))
            doc = f"""
            <div class="doc" id="doc-approval">
              <h2>승인 요청 문서</h2>
              <pre>{escape(outcome.approval_markdown)}</pre>
              <p class="note">원문: <a href="/approval">/approval</a></p>
            </div>
            """

        body = (
            _nav_html(views, blocks)
            + _driver_html(outcome)
            + '<div class="flow">'
            + "".join(stages_html)
            + "</div>"
            + doc
        )

    source_banner = _source_banner(on_visa)
    veil = _veil_html(on_visa)

    # 임계값 칸은 **실제로 쓰인 값**을 되비춘다. 비어 있으면 파이프라인의
    # 기본값이 쓰이며, 그 기본값도 코드에서 읽어 온 것이라 화면이 따로 적지 않는다.
    #
    # 추론까지 못 가고 인테이크에서 멈추면 `outcome.threshold` 가 비어 있다.
    # 그때 사람이 적어 넣은 값까지 지워 버리면 되물음에 답할 때마다 임계값을
    # 다시 타이핑해야 한다. 그래서 실행된 값이 없으면 적어 낸 값을 남긴다.
    threshold_value = (f"{outcome.threshold:g}"
                       if outcome and outcome.threshold else threshold.strip())
    run_cost = ("실데이터에 실모델이면 보통 몇 분 걸립니다."
                if on_visa else "합성 이미지라 보통 수 초입니다.")

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

  {_factory_html(factory)}

  <form method="post" action="/run" id="runform">
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
      <div>
        <label for="threshold">판정 임계값</label>
        <input id="threshold" name="threshold" inputmode="decimal"
               value="{escape(threshold_value)}"
               placeholder="비우면 기본값 {DEFAULT_THRESHOLD}">
        <span class="hint">
          이 값을 넘으면 불량입니다. <strong>내리면 미검은 줄고 과검이 늡니다.</strong>
          원인 판정까지 따라 바뀌는지는 아래 「판정 기준값과 출처」에 적어 두었습니다.
        </span>
      </div>
    </div>
    <p class="nostop">
      ⚠ 한 번 시작하면 중단할 수 없습니다. 열 단계가 끝까지 실행되고, 그동안
      화면은 기다립니다. {escape(run_cost)}
    </p>
    {supplement}
  </form>

  {body}
</div>{veil}</body></html>"""
