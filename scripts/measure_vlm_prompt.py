"""판별 1번의 병목이 어디인가 — 프롬프트인가 위치인가 (4090, 2026-08-16).

    python scripts/measure_vlm_prompt.py --mode ab         프롬프트 A 대 B
    python scripts/measure_vlm_prompt.py --mode control    마스크 크롭의 대조군
    python scripts/measure_vlm_prompt.py --mode vocab      품목별 정상 어휘 수집
    python scripts/measure_vlm_prompt.py --mode c          A/B + 정상 어휘

**VLM 호출 1,318건 · 5시간 25분** 짜리 측정이다. 한 번에 다 돌리지 말고
모드별로 나눠 돌린다. `--categories` 로 좁힐 수 있다.

── 무엇을 알아내려 했나 ────────────────────────────────────────────────

기판에서 판별 1번이 3/10 이었다. 원인 후보가 셋이었다.

    1. 결함이 작아서 못 본다
    2. 질문(프롬프트)이 나빠서 못 본다
    3. 자리를 잘못 짚어서 못 본다

셋 다 재서 **1·2 를 반증하고 3 만 남겼다.**

── 결과 (12 카테고리 · 각 결함 10 · 정상 10 · seed 0) ──────────────────

**크기 가설이 죽었다.**

    macaroni1    444px²   7/10   ← 가장 작은 결함
    pcb4       9,752px²   0/10   ← 가장 큰 결함

22배 차이에 상관이 없고 오히려 약한 역상관이다. 갈리는 축은 **배경
복잡도**였다 — pcb 넷 평균 0.5/10, 나머지 여덟 평균 5.5/10.

**프롬프트로도 안 움직인다.**

    A 일반 예시        검출 38.3%  과검 0.0%
    B 품목별 결함 어휘   검출 41.7%  과검 1.7%
    C ＋품목별 정상 어휘 검출 43.3%  과검 2.5%

120건 중 4~6건은 잡음이고 과검이 함께 늘어 순이득이 없다. 결함 이름을
불러줘도 1500x1000 안에서 못 찾는다 — 자리를 모르기 때문이다.

**자리를 주면 달라진다. 그리고 그것이 착시가 아니다.**

    마스크 크롭 (결함 있음)        80.8%  (97/120)
    정상 무작위 크롭 (결함 없음)   10.8%  (13/120)
    결함 비껴간 크롭 (결함 없음)   11.0%  (13/118)

**7.4배 갈린다.** 대조군이 없으면 80.8% 가 "봤다"인지 "확대된 조각이면
일단 결함이라 한다"인지 갈리지 않는다. 두 대조군이 거의 같다는 것도
중요하다 — 결함 이미지든 정상 이미지든 결함 없는 자리를 주면 같게 답한다.

그리고 이 값이 **역추적 3/10 의 뜻을 확정한다.** 자리를 놓치면 11% 만
`visible` 이므로 3/10 은 "자리를 세 번 맞혔다"이지 우연이 아니다.

── 정답 유출을 막은 것 ─────────────────────────────────────────────────

**이미지별 라벨은 쓰지 않는다.** 품목별 어휘는 `image_anno.csv` 의 label 을
카테고리 전체에서 모은 것이고, 같은 문장을 결함과 정상에 똑같이 보낸다.
이미지마다 그 이미지의 결함 이름을 주면 정답을 알려주는 것이다.

**정상 어휘도 시험용 정상 10장과 겹치지 않는 정상에서 모은다.** 겹치면
자기가 만든 답안지로 시험 보는 셈이다.

읽기만 한다. 저장소 파일을 고치지 않는다(`--mode vocab` 만 어휘 파일을 쓴다).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from agents.adapters import build_adapters  # noqa: E402
from agents.adapters.base import ChatMessage, ImagePart  # noqa: E402

CATEGORIES = ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
              "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]

#: 프롬프트 A 가 쓰는 일반 예시. `agents/vision.py` 의 판별 1번과 같은 목록이다.
GENERIC = "scratch, dent, foreign material, crack, discoloration, missing part"

#: 품목별 정상 어휘. `--mode vocab` 이 만들고 `--mode c` 가 읽는다.
VOCAB_PATH = REPO_ROOT / "data" / "normal_vocab.json"

#: 비껴간 크롭이 결함을 이만큼 넘게 담으면 다시 뽑는다.
OVERLAP_MAX = 0.02

_JSON_RULE = "Reply with JSON only. No prose, no code fences."
_VERDICTS = {"visible", "not_visible", "unknown"}


# ── 공용 ────────────────────────────────────────────────────────────────


def visa_root(args) -> Path:
    return Path(args.visa_root) if args.visa_root else REPO_ROOT / "VisA_20220922"


def defect_vocabulary(root: Path, category: str) -> list[str]:
    """카테고리 **전체**의 결함 어휘. 이미지별 라벨이 아니다.

    이미지마다 그 이미지의 결함 이름을 주면 정답을 알려주는 것이 된다.
    """
    path = root / category / "image_anno.csv"
    if not path.exists():
        return []
    words: collections.Counter = collections.Counter()
    for row in csv.DictReader(path.open(encoding="utf-8")):
        label = row.get("label", "").strip()
        if label == "normal":
            continue
        for word in label.split(","):
            word = word.strip()
            if word and word != "other":
                words[word] += 1
    return [w for w, _ in words.most_common()]


def enlarge(image: Image.Image, to: int) -> Image.Image:
    """짧은 변을 `to` 까지 키운다. 정보를 늘리지 않지만 판독이 안정된다."""
    short = min(image.size)
    if short >= to:
        return image
    scale = to / short
    return image.resize((int(image.width * scale), int(image.height * scale)),
                        Image.LANCZOS)


def mask_box(mask: np.ndarray, size: tuple[int, int], margin: int):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    width, height = size
    return (max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(width, int(xs.max()) + margin), min(height, int(ys.max()) + margin))


def ask(vlm, image, defects: str, normals: str = "") -> str:
    """판별 1번과 같은 문장. **예시 어휘만 바꿔 끼운다.**

    normals 를 주면 프롬프트 C 다 — "양품은 이렇게 생겼다"를 함께 준다.
    실제 검사 기준서에는 불량 항목만 있는 것이 아니라 양품 기준이 있다.
    """
    normal_line = (
        f"A normal, acceptable unit of this product looks like: {normals}.\n"
        if normals else ""
    )
    no_defect = (
        "you do not see one here — the unit matches the normal appearance "
        "described above" if normals else "you do not see one here"
    )
    prompt = (
        "You are inspecting a product image from a manufacturing line.\n"
        f"{normal_line}"
        "Decide whether a visible surface defect is present.\n"
        f'"visible" means you can point to an actual anomaly ({defects}).\n'
        f'"not_visible" means {no_defect}.\n'
        '"unknown" means the image is too unclear to judge.\n\n'
        '"not_visible" is about this view, not about the product. Say it when '
        "you see no defect, even if one might exist elsewhere on the item.\n\n"
        "Respond with:\n"
        '{"verdict": "visible|not_visible|unknown", "confidence": 0.0-1.0, '
        '"reason": "one short sentence naming what you saw and where"}\n'
        f"{_JSON_RULE}"
    )
    try:
        response = vlm.chat([ChatMessage.user(prompt, images=[ImagePart(image)])],
                            json_object=True)
        verdict = (response.json() or {}).get("verdict", "unknown")
        return verdict if verdict in _VERDICTS else "unknown"
    except Exception:
        return "unknown"


def pick(root: Path, category: str, count: int, rng: random.Random):
    """결함·정상을 뽑는다. **모드가 달라도 같은 난수 순서라 같은 이미지가 나온다.**

    A/B 와 C 를 다른 이미지로 재면 프롬프트 차이인지 이미지 차이인지 갈리지
    않는다.
    """
    base = root / category / "Data" / "Images"
    anomaly = sorted((base / "Anomaly").glob("*"))
    normal = sorted((base / "Normal").glob("*"))
    return (rng.sample(anomaly, min(count, len(anomaly))),
            rng.sample(normal, min(count, len(normal))))


# ── 모드 ────────────────────────────────────────────────────────────────


def mode_ab(args, vlm, root: Path, tmp: Path) -> int:
    """프롬프트 A(일반 예시) 대 B(품목별 결함 어휘). 같은 이미지에 질문만 바꾼다.

    B 에만 마스크 크롭을 넣는다 — "질문을 고치면 원본만으로 크롭에 근접하는가"
    가 이 측정의 질문이고, 그 천장은 B 조건에서 재야 뜻이 있다.
    """
    rng = random.Random(args.seed)
    started = time.time()
    rows = []

    print(f"카테고리 {len(args.categories)}개 · 결함 {args.count} · 정상 {args.count} · seed {args.seed}")
    print(f"A 일반 예시: {GENERIC}\n")
    header = (f"{'카테고리':<12} {'A결함':>6} {'A과검':>7} │ "
              f"{'B결함':>6} {'B과검':>7} {'B크롭':>7}   품목 어휘")
    print(header); print("-" * len(header))

    for category in args.categories:
        defects, normals = pick(root, category, args.count, rng)
        vocabulary = ", ".join(defect_vocabulary(root, category)) or GENERIC
        masks = root / category / "Data" / "Masks" / "Anomaly"

        a_hit = b_hit = a_fp = b_fp = crop_hit = crops = 0
        for path in defects:
            a_hit += int(ask(vlm, path, GENERIC) == "visible")
            b_hit += int(ask(vlm, path, vocabulary) == "visible")

            mask_path = masks / (path.stem + ".png")
            if not mask_path.exists():
                continue
            image = Image.open(path).convert("RGB")
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            box = mask_box(mask, image.size, args.margin)
            if box is None:
                continue
            out = tmp / f"{category}_{path.stem}.png"
            enlarge(image.crop(box), args.enlarge).save(out)
            crop_hit += int(ask(vlm, out, vocabulary) == "visible")
            crops += 1
            out.unlink(missing_ok=True)

        for path in normals:
            a_fp += int(ask(vlm, path, GENERIC) == "visible")
            b_fp += int(ask(vlm, path, vocabulary) == "visible")

        rows.append((a_hit, b_hit, a_fp, b_fp, crop_hit, crops, len(defects), len(normals)))
        short = vocabulary if len(vocabulary) <= 44 else vocabulary[:41] + "..."
        print(f"{category:<12} {a_hit:>3}/{len(defects):<2} {a_fp:>4}/{len(normals):<2} │ "
              f"{b_hit:>3}/{len(defects):<2} {b_fp:>4}/{len(normals):<2} "
              f"{crop_hit:>4}/{crops:<2}   {short}")

    print("-" * len(header))
    total_d = sum(r[6] for r in rows) or 1
    total_n = sum(r[7] for r in rows) or 1
    total_c = sum(r[5] for r in rows) or 1
    A, B = sum(r[0] for r in rows), sum(r[1] for r in rows)
    AF, BF, CR = sum(r[2] for r in rows), sum(r[3] for r in rows), sum(r[4] for r in rows)

    print(f"\n  A 일반 프롬프트   검출 {A/total_d:.1%}  과검 {AF/total_n:.1%}")
    print(f"  B 품목별 어휘     검출 {B/total_d:.1%}  과검 {BF/total_n:.1%}")
    print(f"  B 마스크 크롭     검출 {CR/total_c:.1%}   ← 천장")
    print(f"\n  검출 변화 {B-A:+d}건 · 과검 변화 {BF-AF:+d}건")
    print("  **검출만 보지 말 것.** 과검이 함께 늘면 순이득이 없다.")
    print(f"\n  {(total_d*2 + total_n*2 + total_c)}건 · {time.time()-started:.0f}초")
    return 0


def mode_control(args, vlm, root: Path, tmp: Path) -> int:
    """마스크 크롭의 대조군 — **이것이 없으면 크롭 수치를 못 쓴다.**

    마스크 크롭은 결함이 반드시 든 조각만 보여준다. 거기서 80% 가 나와도
    "봤다"인지 "확대된 조각이면 일단 결함이라 한다"인지 갈리지 않는다.

      1. 정상 무작위 크롭   정상 이미지에서 같은 크기로 아무 데나
      2. 결함 비껴간 크롭   결함 이미지에서 마스크를 피해 같은 크기로
                           → 역추적이 자리를 놓쳤을 때 무슨 답이 나오는지
    """
    rng = random.Random(args.seed)
    started = time.time()
    rows = []

    print(f"마스크 크롭 대조군 · 카테고리 {len(args.categories)}개 · 각 {args.count}건 · seed {args.seed}")
    print("프롬프트는 품목별 결함 어휘(B)와 같다\n")
    header = f"{'카테고리':<12} {'정상무작위':>12} {'결함비껴간':>12}   크기"
    print(header); print("-" * len(header))

    for category in args.categories:
        defects, normals = pick(root, category, args.count, rng)
        vocabulary = ", ".join(defect_vocabulary(root, category)) or GENERIC
        masks = root / category / "Data" / "Masks" / "Anomaly"

        sizes = []
        for path in defects:
            mask_path = masks / (path.stem + ".png")
            if not mask_path.exists():
                continue
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            box = mask_box(mask, (mask.shape[1], mask.shape[0]), args.margin)
            if box:
                sizes.append((box[2] - box[0], box[3] - box[1]))
        if not sizes:
            continue

        # 위치는 따로 고정한다 — 이미지 선택 난수와 섞이면 재현이 흔들린다.
        place = random.Random(args.seed + 1)
        fp_normal = n_normal = fp_off = n_off = 0

        for path in normals:
            image = Image.open(path).convert("RGB")
            box = _random_box(place, image.size, sizes[place.randrange(len(sizes))])
            if box is None:
                continue
            out = tmp / f"n_{category}_{path.stem}.png"
            enlarge(image.crop(box), args.enlarge).save(out)
            fp_normal += int(ask(vlm, out, vocabulary) == "visible")
            n_normal += 1
            out.unlink(missing_ok=True)

        for path in defects:
            mask_path = masks / (path.stem + ".png")
            if not mask_path.exists():
                continue
            image = Image.open(path).convert("RGB")
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            box = _random_box(place, image.size, sizes[place.randrange(len(sizes))], mask)
            if box is None:
                continue
            out = tmp / f"a_{category}_{path.stem}.png"
            enlarge(image.crop(box), args.enlarge).save(out)
            fp_off += int(ask(vlm, out, vocabulary) == "visible")
            n_off += 1
            out.unlink(missing_ok=True)

        median = (f"{int(np.median([w for w, _ in sizes]))}x"
                  f"{int(np.median([h for _, h in sizes]))}")
        rows.append((fp_normal, n_normal, fp_off, n_off))
        print(f"{category:<12} {fp_normal:>7}/{n_normal:<3} {fp_off:>8}/{n_off:<3}   {median}")

    print("-" * len(header))
    a, na = sum(r[0] for r in rows), sum(r[1] for r in rows) or 1
    b, nb = sum(r[2] for r in rows), sum(r[3] for r in rows) or 1
    print(f"{'합계':<12} {a:>7}/{na:<3} {b:>8}/{nb:<3}")
    print(f"\n  정상 무작위 크롭에서 visible   {a/na:.1%}   ({a}/{na})")
    print(f"  결함 비껴간 크롭에서 visible   {b/nb:.1%}   ({b}/{nb})")
    print("\n  **낮아야 마스크 크롭 수치를 상한선으로 읽을 수 있다.**")
    print("  높으면 '확대된 조각이면 결함이라 한다'는 뜻이라 상한선이 아니다.")
    print("  뒤는 역추적이 자리를 놓쳤을 때 무슨 답이 나오는지다.")
    print(f"\n  {na+nb}건 · {time.time()-started:.0f}초")
    return 0


def _random_box(rng, size, wh, mask=None):
    """size 안에서 wh 크기 상자를 무작위로. mask 를 주면 거의 안 겹치게."""
    width, height = size
    w, h = min(wh[0], width), min(wh[1], height)
    for _ in range(40):
        x = rng.randint(0, width - w)
        y = rng.randint(0, height - h)
        if mask is None:
            return (x, y, x + w, y + h)
        inside = mask[y:y + h, x:x + w].sum()
        if inside / max(1, int(mask.sum())) <= OVERLAP_MAX:
            return (x, y, x + w, y + h)
    return None


def mode_vocab(args, vlm, root: Path, tmp: Path) -> int:
    """품목별 정상 어휘 수집 — 모델이 쓰는 말로 "양품은 이렇게 생겼다"를 만든다.

    우리가 상상해 적는 것보다 모델이 실제로 쓰는 말이 통한다.

    **시험용 정상 10장과 겹치지 않게 뽑는다.** 겹치면 자기가 만든 답안지로
    시험 보는 셈이다.
    """
    prompt = (
        "This is a photo of a product unit on a manufacturing inspection line. "
        "This unit passed inspection — it is a normal, acceptable unit.\n"
        "Describe what a normal unit of this product looks like, in short keywords "
        "describing its expected appearance (shape, surface, colour, arrangement).\n"
        "Do not mention defects. Do not mention this specific photo.\n"
        '{"keywords": ["...", "...", "..."]}\n'
        f"{_JSON_RULE}"
    )
    rng = random.Random(args.seed)
    started = time.time()
    result: dict[str, list[str]] = {}

    print(f"품목별 정상 어휘 수집 · 카테고리 {len(args.categories)}개 · 각 {args.collect}장")
    print(f"시험용 정상 {args.count}장은 제외한다\n")

    for category in args.categories:
        _, used = pick(root, category, args.count, rng)   # 난수 순서를 맞춰 그 10장을 알아낸다
        every = sorted((root / category / "Data" / "Images" / "Normal").glob("*"))
        pool = [p for p in every if p not in set(used)][:args.collect]

        words: collections.Counter = collections.Counter()
        answered = 0
        for path in pool:
            try:
                response = vlm.chat(
                    [ChatMessage.user(prompt, images=[ImagePart(path)])], json_object=True)
                for keyword in (response.json() or {}).get("keywords") or []:
                    keyword = str(keyword).strip().lower()
                    if 2 < len(keyword) < 40:
                        words[keyword] += 1
                answered += 1
            except Exception:
                pass

        top = [w for w, _ in words.most_common(args.top)]
        result[category] = top
        print(f"{category:<12} {answered:>2}/{len(pool)} 장 · 어휘 {len(words):>2}종 → {', '.join(top)}")

    VOCAB_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(f"\n{VOCAB_PATH.relative_to(REPO_ROOT)} 에 저장 · {time.time()-started:.0f}초")
    return 0


def mode_c(args, vlm, root: Path, tmp: Path) -> int:
    """프롬프트 C — 품목별 결함 어휘 + 품목별 정상 어휘.

    **A/B 와 완전히 같은 이미지에 돌린다.** 다른 이미지로 재면 프롬프트
    차이인지 이미지 차이인지 갈리지 않는다.
    """
    if not VOCAB_PATH.exists():
        print(f"{VOCAB_PATH.relative_to(REPO_ROOT)} 이 없다. --mode vocab 을 먼저 돌려야 한다.")
        return 1
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

    rng = random.Random(args.seed)
    started = time.time()
    rows = []

    print(f"프롬프트 C · 카테고리 {len(args.categories)}개 · 결함 {args.count} · 정상 {args.count}")
    print("A/B 와 같은 이미지 · 정상 어휘는 겹치지 않는 정상에서 수집\n")
    header = f"{'카테고리':<12} {'C결함':>6} {'C과검':>7} {'C크롭':>7}   정상 어휘"
    print(header); print("-" * len(header))

    for category in args.categories:
        defects, normals = pick(root, category, args.count, rng)
        defect_words = ", ".join(defect_vocabulary(root, category)) or GENERIC
        normal_words = ", ".join(vocab.get(category, [])) or "a clean, undamaged unit"
        masks = root / category / "Data" / "Masks" / "Anomaly"

        hit = fp = crop_hit = crops = 0
        for path in defects:
            hit += int(ask(vlm, path, defect_words, normal_words) == "visible")
            mask_path = masks / (path.stem + ".png")
            if not mask_path.exists():
                continue
            image = Image.open(path).convert("RGB")
            mask = np.array(Image.open(mask_path).convert("L")) > 0
            box = mask_box(mask, image.size, args.margin)
            if box is None:
                continue
            out = tmp / f"c_{category}_{path.stem}.png"
            enlarge(image.crop(box), args.enlarge).save(out)
            crop_hit += int(ask(vlm, out, defect_words, normal_words) == "visible")
            crops += 1
            out.unlink(missing_ok=True)

        for path in normals:
            fp += int(ask(vlm, path, defect_words, normal_words) == "visible")

        rows.append((hit, fp, crop_hit, crops, len(defects), len(normals)))
        short = normal_words if len(normal_words) <= 38 else normal_words[:35] + "..."
        print(f"{category:<12} {hit:>3}/{len(defects):<2} {fp:>4}/{len(normals):<2} "
              f"{crop_hit:>4}/{crops:<2}   {short}")

    print("-" * len(header))
    H, F, C = sum(r[0] for r in rows), sum(r[1] for r in rows), sum(r[2] for r in rows)
    td = sum(r[4] for r in rows) or 1
    tn = sum(r[5] for r in rows) or 1
    tc = sum(r[3] for r in rows) or 1
    print(f"\n  C 결함 원본 검출률   {H/td:.1%}   ({H}/{td})")
    print(f"  C 정상 원본 과검률   {F/tn:.1%}   ({F}/{tn})")
    print(f"  C 마스크 크롭 검출률 {C/tc:.1%}   ({C}/{tc})")
    print(f"\n  {td+tn+tc}건 · {time.time()-started:.0f}초")
    return 0


MODES = {"ab": mode_ab, "control": mode_control, "vocab": mode_vocab, "c": mode_c}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="판별 1번 병목 측정")
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--categories", nargs="*", default=CATEGORIES)
    parser.add_argument("--visa-root", default=None,
                        help="비우면 저장소 아래 VisA_20220922")
    parser.add_argument("--count", type=int, default=10, help="카테고리당 결함·정상 장수")
    parser.add_argument("--collect", type=int, default=15, help="vocab: 수집 장수")
    parser.add_argument("--top", type=int, default=6, help="vocab: 남길 어휘 수")
    parser.add_argument("--margin", type=int, default=64, help="크롭 여유 픽셀")
    parser.add_argument("--enlarge", type=int, default=512, help="크롭 확대 목표")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = visa_root(args)
    if not root.is_dir():
        print(f"VisA 를 찾지 못했다: {root}")
        return 1

    _, vlm = build_adapters()
    if vlm.is_stub:
        print("시각 언어 모델이 연결되지 않았다. 스텁으로는 이 측정이 의미가 없다.")
        print("SHVO_VLM_PROVIDER 등을 설정하고 scripts/check_models.py 로 확인할 것.")
        return 1
    print(f"시각 언어 모델 {vlm.describe()} · 여유 {args.margin}px · 확대 {args.enlarge}px\n")

    # 크롭은 임시 폴더에만 만든다. 저장소를 어지럽히지 않는다.
    with tempfile.TemporaryDirectory(prefix="shvo_vlm_") as tmp:
        return MODES[args.mode](args, vlm, root, Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
