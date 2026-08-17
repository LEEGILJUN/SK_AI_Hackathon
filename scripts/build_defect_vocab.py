"""VisA 결함 어휘 표를 만든다 — `data/defect_vocab.json`.

    .venv/bin/python scripts/build_defect_vocab.py

**정본은 VisA 각 카테고리의 `image_anno.csv` 다.** 이 스크립트는 거기서
낱개 어휘를 뽑고 현장 한국어 별칭을 붙여 표로 만든다. 표를 손으로 고치지
말고 아래 `ALIASES` 를 고친 뒤 다시 돌린다 — 두 벌이 되면 한쪽만 는다.

── 왜 필요한가 ─────────────────────────────────────────────────────────

이슈는 한국어로 온다. 실측에서 모델이 `미세 스크래치` 를 담아 왔는데
이력의 값은 `scratch` 라 **중복 차단의 0.40 짜리 대조 축이 통째로 죽어
있었다.** 라인·품목만 겹쳐 0.60 이고 임계 0.95 를 못 넘는다. 화면은 이슈
이력 그래프를 그려 놓고 "중복을 차단합니다"라고 적는데 실제로는 안 걸렸다.

── 품목마다 쓰는 말이 다르다 ───────────────────────────────────────────

    pcb1 · pcb2 · pcb3   scratch
    fryum · macaroni     small scratches
    capsules             scratch (+ bubble · discolor · leak · misshape)

그래서 카테고리별로 묶는다. 한 덩어리로 두면 "미세 스크래치"가 pcb 에서
엉뚱한 값으로 간다.

라벨은 쉼표로 여러 개가 묶여 있고(`melt,scratch`) 낱개로 쪼갠다. `other`
는 뜻이 없어 뺀다.

**예선은 pcb1~4 만 쓴다.** 나머지 여덟은 본선에서 쓸 자리라 미리 만들어
둔다 — 그때 가서 VisA 를 다시 훑을 이유가 없다.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VISA = REPO_ROOT / "VisA_20220922"
OUT = REPO_ROOT / "data" / "defect_vocab.json"

#: 현장이 쓰는 말 → VisA 어휘. **어휘 자체를 여기서 만들지 않는다.**
#: 왼쪽은 `image_anno.csv` 에 실제로 있는 값이어야 하고, 없는 것을 적으면
#: 아래에서 "쓰이지 않는 별칭"으로 알려 준다.
ALIASES: dict[str, list[str]] = {
    "scratch": ["스크래치", "긁힘", "흠집", "스크렛치"],
    "scratches": ["스크래치", "긁힘", "흠집"],
    "small scratches": ["미세 스크래치", "잔긁힘", "미세 긁힘", "실긁힘"],
    "bent": ["휨", "굽음", "변형", "휘어짐"],
    "melt": ["녹음", "용융", "눌어붙음", "녹아내림"],
    "missing": ["누락", "빠짐", "미부착", "결품"],
    "burnt": ["탐", "그을림", "탄자국", "눌음"],
    "damage": ["손상", "파손", "깨짐"],
    "dirt": ["오염", "때", "얼룩"],
    "extra": ["이물", "이물질", "여분", "잔여물"],
    "wrong place": ["오조립", "위치불량", "오배치", "자리틀림"],
    "bubble": ["기포", "공기방울", "버블"],
    "discolor": ["변색", "색바램", "탈색"],
    "leak": ["누액", "샘", "누출"],
    "misshape": ["형상불량", "찌그러짐"],
    "small cracks": ["미세 균열", "잔균열", "실금"],
    "small holes": ["미세 구멍", "핀홀", "작은 구멍"],
    "middle breakage": ["중앙 파단", "가운데 부러짐"],
    "breakage down the middle": ["중앙 파단", "반으로 부러짐"],
    "corner missing": ["모서리 결손", "귀퉁이 빠짐"],
    "corner or edge breakage": ["모서리 파손", "가장자리 깨짐"],
    "corner and edge breakage": ["모서리 파손", "가장자리 깨짐"],
    "chip around edge and corner": ["가장자리 결손", "모서리 칩"],
    "small chip around edge": ["가장자리 미세 결손", "작은 칩"],
    "different colour spot": ["이색 반점", "다른 색 점", "이색점"],
    "different color spot": ["이색 반점", "다른 색 점"],
    "similar colour spot": ["유사색 반점", "비슷한 색 점"],
    "same colour spot": ["동색 반점", "같은 색 점"],
    "color spot similar to the object": ["유사색 반점"],
    "stuck together": ["붙음", "달라붙음", "점착"],
    "fryum stuck together": ["붙음", "달라붙음"],
    "chunk of gum missing": ["덩어리 결손", "일부 빠짐"],
    "chunk of wax missing": ["왁스 결손", "덩어리 빠짐"],
    "extra wax in candle": ["왁스 과다", "여분 왁스"],
    "wax melded out of the candle": ["왁스 흘러내림", "왁스 용융"],
    "weird candle wick": ["심지 불량", "심지 이상"],
    "foreign particals on candle": ["이물", "표면 이물"],
    "damaged corner of packaging": ["포장 모서리 손상", "포장 파손"],
}


def collect() -> dict[str, set[str]]:
    """카테고리마다 낱개 결함 어휘를 모은다."""
    per: dict[str, set[str]] = defaultdict(set)
    for folder in sorted(p for p in VISA.iterdir() if p.is_dir()):
        anno = folder / "image_anno.csv"
        if not anno.exists():
            continue
        with anno.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                label = (row.get("label") or "").strip()
                if not label or label.lower() == "normal":
                    continue
                for atom in (x.strip() for x in label.split(",")):
                    if atom and atom != "other":
                        per[folder.name].add(atom)
    return per


def main() -> int:
    if not VISA.is_dir():
        print(f"VisA 를 찾지 못했다: {VISA}")
        print("이 표는 이미 만들어져 있으므로, 어휘를 늘릴 때만 다시 돌리면 된다.")
        return 1

    per = collect()
    if not per:
        print("결함 어휘를 하나도 못 찾았다. image_anno.csv 를 확인할 것.")
        return 1

    table: dict[str, object] = {
        "_comment": (
            "VisA 12개 카테고리의 결함 어휘와 현장 한국어 별칭. 정본은 각 "
            "카테고리의 image_anno.csv 이고 이 파일은 "
            "scripts/build_defect_vocab.py 가 만든다. 손으로 고치지 말 것 — "
            "별칭을 늘리려면 그 스크립트의 ALIASES 를 고친다."
        )
    }
    used: set[str] = set()
    unnamed: set[str] = set()
    for category in sorted(per):
        terms: dict[str, list[str]] = {}
        for term in sorted(per[category]):
            terms[term] = list(ALIASES.get(term, []))
            (used if term in ALIASES else unnamed).add(term)
        table[category] = terms

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    total = sum(len(v) for v in per.values())
    print(f"카테고리 {len(per)}개 · 어휘 {total}개 → {OUT.relative_to(REPO_ROOT)}")
    if unnamed:
        print(f"\n별칭이 없는 어휘 {len(unnamed)}개 — 한국어로 오면 못 잡는다")
        for term in sorted(unnamed):
            print(f"  {term}")
    stale = sorted(set(ALIASES) - used)
    if stale:
        # 데이터에 없는 어휘에 별칭을 달아 두면, 왜 안 잡히는지 찾느라 시간을 쓴다.
        print(f"\n쓰이지 않는 별칭 {len(stale)}개 — VisA 에 없는 어휘다")
        for term in stale:
            print(f"  {term}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
