"""판별 5번 — 뱅크의 이 패치가 잘못 섞인 결함인가, 진짜 정상품인가.

진단이 갈리는 자리다. 결함이면 뱅크 오염, 진짜 정상품이면 정상 분포 중첩이고
**조치가 정반대다.** 이 판정이 틀리면 진단 전체가 틀린다.

양방향을 다 본다. 전부 normal 이라 답하면 뱅크 오염을 영영 못 찾고, 전부 defect 라
답하면 멀쩡한 정상 이미지를 뱅크에서 빼게 된다. 한쪽만 재면 그것을 놓친다.

── 실측 결과 (2026-08-14 · RTX 4090 Laptop · qwen3vl-8b) ──────────────────

    VisA capsules · 정상 150장 + 결함 5장 혼입 · coreset 0.01

    A (정답 defect)  6건 — 맞음 4 · 보류 1 · 틀림 1
    B (정답 normal) 15건 — 맞음 13 · 보류 2 · 틀림 0     ← 이쪽이 더 중요하다

    거짓 판정 1/21건

**B 의 틀림 0 이 실질적으로 더 중요하다.** 정상 패치를 defect 로 오판하면
멀쩡한 정상 이미지를 뱅크에서 빼게 되고, 그것은 되돌리기 어려우며 커버리지를
깎는다. 15건에서 한 번도 없었다.

**A 가 6건뿐이라 성능을 말할 크기가 아니다.** 뱅크에 남은 결함 패치가 6개뿐이라
표본이 그것으로 묶였다. 방향을 보여줄 뿐이다. 자세한 것은
`docs/실험_판별5번.md`.

── 유도 편향 — 전제를 알려주는 쪽이 더 안전했다 ────────────────────────────

judge_bank_patch() 는 "이 패치는 정상으로 등록된 것"이라는 전제를 모델에게
알려준다. 그 전제가 normal 쪽으로 답을 끌 위험이 있어 프롬프트에
"registered as normal is not evidence that it is normal" 을 넣어 두었는데,
그것이 실제로 작동하는지는 확인된 적이 없었다.

    A · 뱅크 오염 패치 6건    두 프롬프트 전부 일치.  4건을 defect 로 찾아냄
    B · 정상 패치 15건   6건 갈림 — 전제 없는 쪽이 unknown 3 · **defect 3**

**유도는 없었고, 오히려 반대쪽이 위험했다.** 전제 없이 "결함이 보이는가"라고만
물으면 모델이 무언가를 찾으려 들어 멀쩡한 정상 표면 3건에서 없는 결함을
만들어냈다. 그대로 조치했으면 정상 이미지를 뺐을 것이다. `agents/vision.py`
의 방어 문장은 고칠 이유가 없고, 그 설계가 옳았다는 근거가 나왔다.

── 정답을 어떻게 붙였는가 ────────────────────────────────────────────────

"결함 이미지에서 온 패치 = defect" 로 두면 거칠다. 결함 이미지에서 온 패치의
대부분은 그 이미지의 멀쩡한 표면이고, 그것을 defect 정답으로 놓으면 모델이
맞게 답해도 틀린 것으로 세게 된다. 그래서 VisA 정답 마스크로 한 번 더 가른다.

    A  혼입 이미지 출신 + 패치 자리가 마스크와 겹침   → 정답 defect   (채점)
    B  진짜 정상 이미지 출신                     → 정답 normal   (채점)
    C  혼입 이미지 출신 + 마스크와 안 겹침            → 정답 애매      (세기만 함)

C 를 채점에서 빼는 이유는 그것이 실제로는 정상 표면이기 때문이다. 다만 몇 개나
되는지는 보고한다 — 혼입 이미지를 통째로 빼는 조치가 얼마나 과한지 보여준다.
실측에서 C 는 398개였다(혼입 이미지가 남긴 404개 중 98.5%).

패치 자리 판정은 여유 없이(margin=0) 격자 한 칸으로 하고, 모델에게 보여줄 때만
맥락을 붙인다. 판정 기준과 제시 방식을 섞지 않기 위해서다.

── coreset 상한에 주의 ───────────────────────────────────────────────────

`--coreset-ratio 0.1` 로 돌리면 `DEFAULT_MAX_BANK_SIZE = 20,000` 상한에 걸려
실제로는 4.1% 가 된다(`bank.meta["coreset_capped"]` 가 True 로 남는다).
0.1 을 진짜로 재려면 상한을 풀어야 한다. 같은 수치를 다시 낼 때 헷갈리기 쉬운
지점이라 스크립트가 상한 여부를 출력한다.

크롭 파라미터는 판별 1번에서 검증된 값을 그대로 쓴다 — margin=64,
enlarge_to=512. 24 로 하면 전부 unknown 이 나온다
(docs/실험_역추적크롭.md).

읽기만 한다. 저장소 파일을 고치지 않는다.

    python scripts/measure_patch_judgment.py --max-normal 150 --contaminants 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.adapters import build_adapters  # noqa: E402
from agents.vision import judge_bank_patch, judge_defect_visible  # noqa: E402
from inspection import PatchEmbedder, build_bank  # noqa: E402
from inspection.crop import crop_patch, patch_box  # noqa: E402
from inspection.isolation import suspect_images  # noqa: E402

DEFAULT_MARGIN = 64   # 판별 1번에서 검증된 값. 24 로 하면 전부 unknown
ENLARGE_TO = 512


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="판별 5번 측정")
    p.add_argument("--category", default="capsules")
    p.add_argument("--visa-root", default="VisA_20220922")
    p.add_argument("--max-normal", type=int, default=150)
    p.add_argument("--contaminants", type=int, default=5, help="정상셋에 섞을 결함 이미지 수")
    p.add_argument("--samples", type=int, default=15, help="무리별로 판독할 패치 수")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    p.add_argument("--coreset-ratio", type=float, default=0.01)
    p.add_argument("--max-bank-size", type=int, default=None,
                   help="비우면 기본 상한 20,000. 높은 coreset 비율을 진짜로 재려면 올려야 한다")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cell_hits_mask(ref, grid, image_size, config, mask: np.ndarray) -> bool:
    """격자 한 칸이 결함 픽셀을 물고 있는가. 여유 없이 본다."""
    left, top, right, bottom = patch_box(ref, grid, image_size, config, margin=0)
    h, w = mask.shape
    window = mask[max(0, top):min(h, bottom), max(0, left):min(w, right)]
    return bool(window.any())


def _saw_defect(verdict: str) -> bool | None:
    """두 판별의 판정을 **공통 축**으로 옮긴다 — 결함을 봤는가.

    판별 1번은 `visible`/`not_visible`, 5번은 `defect`/`genuine_normal` 이라
    어휘가 다르다. 어느 쪽이든 "결함을 봤다/못 봤다/모르겠다" 셋으로 줄면
    견줄 수 있다. 모르겠다는 None 이고, 두 판별이 모두 None 이면 같다고 본다.
    """
    if verdict in ("defect", "visible"):
        return True
    if verdict in ("genuine_normal", "not_visible"):
        return False
    return None


def main() -> int:
    args = parse_args()
    base = REPO_ROOT / args.visa_root / args.category / "Data"
    normal_dir = base / "Images" / "Normal"
    defect_dir = base / "Images" / "Anomaly"
    mask_dir = base / "Masks" / "Anomaly"
    if not normal_dir.exists():
        print(f"VisA 를 못 찾았다: {normal_dir}")
        return 1

    normals = sorted(normal_dir.glob("*.JPG"))[: args.max_normal]
    contaminants = sorted(defect_dir.glob("*.JPG"))[: args.contaminants]

    embedder = PatchEmbedder()
    print(f"장치 {embedder.device}")
    print(f"오염된 뱅크 구성 중 — 정상 {len(normals)}장 + 결함 {len(contaminants)}장 혼입...")
    build_kwargs = {}
    if args.max_bank_size is not None:
        build_kwargs["max_bank_size"] = args.max_bank_size
    bank = build_bank(list(normals) + list(contaminants), embedder,
                      coreset_ratio=args.coreset_ratio, seed=args.seed,
                      bank_version="patch_judgment_probe", **build_kwargs)
    grid = tuple(int(v) for v in bank.meta["grid"])
    total_rows = bank.embeddings.shape[0]
    print(f"뱅크 패치 {total_rows:,}개 · 격자 {grid[0]}x{grid[1]}")
    if bank.meta.get("coreset_capped"):
        actual = total_rows / bank.meta.get("total_patches_before_coreset", total_rows)
        print(f"  ! coreset {args.coreset_ratio} 요청이 상한 "
              f"{bank.meta.get('max_bank_size'):,} 에 걸려 실제 {actual:.1%} 다. "
              f"--max-bank-size 로 상한을 올려야 요청한 비율이 나온다.")
    print()

    contaminant_names = {p.name for p in contaminants}
    masks: dict[str, np.ndarray] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for p in contaminants:
        mp = mask_dir / (p.stem + ".png")
        masks[p.name] = np.array(Image.open(mp).convert("L")) > 0
        with Image.open(p) as im:
            sizes[p.name] = im.size

    # ── 뱅크 행을 세 무리로 가른다 ──────────────────────────────────────
    group_a: list[int] = []  # 혼입 이미지 출신 + 결함 위 → defect
    group_b: list[int] = []  # 정상 출신           → normal
    group_c: list[int] = []  # 혼입 이미지 출신 + 정상부 → 애매

    for i in range(total_rows):
        ref = bank.origin_of(i)
        name = Path(ref.source_image).name
        if name in contaminant_names:
            if cell_hits_mask(ref, grid, sizes[name], embedder.config, masks[name]):
                group_a.append(i)
            else:
                group_c.append(i)
        else:
            group_b.append(i)

    print("뱅크 구성 내역")
    print(f"  A 혼입 이미지 출신 · 결함 위    {len(group_a):>5,}개   ← 정답 defect")
    print(f"  B 정상 이미지 출신         {len(group_b):>5,}개   ← 정답 normal")
    print(f"  C 혼입 이미지 출신 · 정상 부위  {len(group_c):>5,}개   ← 채점 제외")
    if not group_a:
        print("\n뱅크 오염 패치가 뱅크에 한 개도 안 남았다. contaminants 를 늘리거나 "
              "coreset-ratio 를 조정해야 한다.")
        return 1
    amp = len(group_a) / total_rows
    src = len(contaminants) / (len(normals) + len(contaminants))
    from_dirty = (len(group_a) + len(group_c)) / total_rows
    print(f"\n  혼입 이미지가 원본의 {src:.1%} 인데 그 출신 패치가 뱅크의 {from_dirty:.1%} 다 "
          f"({from_dirty / src:.1f}배).")
    print(f"  그런데 실제 결함 위 패치는 뱅크의 {amp:.1%} 뿐이다. "
          f"coreset 이 끌어올린 것은 '결함이 있는 이미지'이지 '결함 그 자체'가 아니다.")
    if group_c:
        waste = len(group_c) / (len(group_a) + len(group_c))
        print(f"  이미지 단위로 빼면 정상 표면 패치 {len(group_c):,}개({waste:.1%})를 함께 버린다.")

    rng = np.random.default_rng(args.seed)
    pick_a = rng.choice(group_a, size=min(args.samples, len(group_a)), replace=False)
    pick_b = rng.choice(group_b, size=min(args.samples, len(group_b)), replace=False)

    _, vlm = build_adapters()
    print(f"\n시각 언어 모델 {vlm.describe()} · 여유 {args.margin}px · 확대 {ENLARGE_TO}px\n")

    head = f"{'무리':<4} {'출처':<11} {'좌표':>9} {'정답':<8} {'등록전제':<9} {'전제없음':<9} {'일치':<5}"
    print(head)
    print("-" * len(head))

    stats = {"A": {"ok": 0, "held": 0, "wrong": 0, "n": 0},
             "B": {"ok": 0, "held": 0, "wrong": 0, "n": 0}}
    disagree = 0
    invented: list[str] = []   # 전제 없는 쪽이 정상 표면에서 결함을 지어낸 건

    for label, picks, truth in (("A", pick_a, "defect"), ("B", pick_b, "genuine_normal")):
        for i in picks:
            ref = bank.origin_of(int(i))
            patch = crop_patch(ref.source_image, ref, grid, embedder.config,
                               margin=args.margin, enlarge_to=ENLARGE_TO)

            bank_j = judge_bank_patch(vlm, patch)          # "정상으로 등록됨" 전제 있음
            plain_j = judge_defect_visible(vlm, patch, reported_defect="표면 결함")

            s = stats[label]
            s["n"] += 1
            if bank_j.verdict == "unknown":
                s["held"] += 1
            elif bank_j.verdict == truth:
                s["ok"] += 1
            else:
                s["wrong"] += 1

            # **두 판별은 어휘가 다르다.** 5번은 defect/genuine_normal,
            # 1번은 visible/not_visible 이라 직접 견주면 절대 같을 수
            # 없다. 실제로 그렇게 두어 '갈린 건 30/30' 이 나온 적이 있다.
            # 공통 축(결함을 봤는가)으로 옮겨 견준다.
            same = _saw_defect(bank_j.verdict) == _saw_defect(plain_j.verdict)
            if not same:
                disagree += 1
            if label == "B" and plain_j.verdict == "visible":
                invented.append(Path(ref.source_image).name)

            print(f"{label:<4} {Path(ref.source_image).name:<11} "
                  f"{f'({ref.row},{ref.col})':>9} {truth:<8} "
                  f"{bank_j.verdict:<9} {plain_j.verdict:<9} {'=' if same else 'X':<5}")

    print()
    for label, truth in (("A", "defect"), ("B", "genuine_normal")):
        s = stats[label]
        if not s["n"]:
            continue
        print(f"{label} (정답 {truth}) {s['n']}건 — 맞음 {s['ok']} · 보류 {s['held']} · 틀림 {s['wrong']}")

    wrong_total = stats["A"]["wrong"] + stats["B"]["wrong"]
    n_total = stats["A"]["n"] + stats["B"]["n"]
    print(f"\n거짓 판정 {wrong_total}/{n_total}건")
    print(f"  그중 B 의 틀림 {stats['B']['wrong']}건 — 정상 패치를 결함이라 한 것. "
          f"이쪽이 실질적으로 더 위험하다(멀쩡한 정상 이미지를 뱅크에서 빼게 된다).")
    print(f"두 프롬프트가 갈린 건 {disagree}/{n_total}건", end="  ")
    print("← 0 이면 '정상으로 등록됨' 전제가 답을 유도하지 않는다는 뜻")
    if invented:
        print(f"\n  전제 없는 judge_defect_visible() 이 정상 표면에서 결함을 지어낸 건 "
              f"{len(invented)}개: {', '.join(sorted(set(invented)))}")
        print("  전제를 빼는 쪽이 더 위험하다는 근거다. vision.py 의 방어 문장은 유지한다.")

    # ── 고립도와 겹치는가 ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("고립도가 뽑은 뱅크 오염 후보와 대조")
    print("=" * 60)
    scores = suspect_images(bank, top_n=len(contaminants) * 2)
    hit = 0
    for s in scores:
        name = Path(s.image).name
        is_cont = name in contaminant_names
        hit += int(is_cont)
        mark = "혼입 이미지" if is_cont else "정상"
        print(f"  {name:<12} z_mean {s.z_mean:>6.2f} · 패치 {s.patch_count:>4}개  [{mark}]")
    print(f"\n  후보 {len(scores)}개 중 실제 혼입 이미지 {hit}개")
    if hit == 0:
        print("  고립도가 혼입 이미지를 못 짚었다. 이미지 단위 z_mean 이 정상부 패치에 "
              "희석된 것이다 — 실측에서 확인된 한계이고, 그래서 큐레이션은 고립도를 "
              "단독 근거로 쓰지 않는다(agents/curate.py).")
    else:
        print("  고립도와 시각 판독이 같은 이미지를 지목하면 근거 두 갈래가 서로를 뒷받침한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
