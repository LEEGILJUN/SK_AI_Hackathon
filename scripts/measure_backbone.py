"""특징 추출기를 바꾸면 역추적 위치가 나아지는가 — 측정 전용 (4090, 2026-08-16).

    python scripts/measure_backbone.py --mode trace    좌표만. 시각 언어 모델을 안 쓴다
    python scripts/measure_backbone.py --mode judge    그 좌표로 잘라 실제로 물어본다

**`inspection/` 을 하나도 건드리지 않는다.** 뱅크도 임계값도 시연 경로도
그대로다. 여기서 만드는 뱅크는 이 실행 안에서만 살고 저장되지 않는다.

── 왜 이 측정인가 ──────────────────────────────────────────────────────

`docs/실험_VLM판독.md` 가 판별 1번의 병목을 **역추적 위치** 하나로 좁혔다.
프롬프트·결함 크기·판독 능력·표시 방식을 차례로 배제하고 남은 것이다.

    자리를 정확히 주면 (마스크로 잘라 줌)   80.8%
    엉뚱한 자리를 주면                      10.8% · 11.0%

7.4배 갈린다. 그러니 **자리를 짚는 쪽을 고치면 판별 1번이 산다.**

AnomalyDINO(WACV 2025, arXiv 2405.14529)가 뒤 구조를 그대로 두고 특징
추출기만 DINOv2 로 바꿔 VisA 1-shot 에서 PatchCore 79.9% → 87.4% 를 냈다.
우리와 구조가 같은 방식이라(정상 패치를 뱅크에 담고 최근접 거리로 판정)
그것이 우리 병목에 닿는지를 잰다.

    기준   wide_resnet50_2 · 512 정사각 · 격자 64x64   (지금 쓰는 것)
    후보   DINOv2 ViT-S/14 · 672 정사각 · 격자 48x48

**논문의 672 는 짧은 변 기준이고 여기서는 정사각이다.** 우리 파이프라인이
512 정사각이라 전처리를 맞춰야 특징 추출기 하나만 놓고 비교가 된다.
논문이 함께 쓰는 **배경 분리는 넣지 않았다** — 그것까지 넣으면 무엇이
들었는지 모른다. 그래서 여기 수치를 87.4% 와 나란히 놓으면 안 된다.

── 두 단계로 나눈 이유 ─────────────────────────────────────────────────

trace 는 좌표까지만 낸다. **크롭 담김이 올라도 판별 1번이 좋아지는 것은
아니다** — 결함이 크롭 안에 있어도 못 볼 수 있다. 그래서 judge 가 실제로
물어본다. 정상 이미지도 같은 수만큼 넣어 과검을 함께 잰다. 결함만 재면
"무조건 visible 이라 답하는" 경우와 구분되지 않는다.

    담김 ↑ · 검출 ↑ · 과검 그대로   →  특징 추출기 교체가 답이다
    담김 ↑ · 검출 제자리            →  크롭 안에 있어도 못 본다. 바꿔도 소용없다
    검출 ↑ · 과검도 ↑               →  visible 쪽으로 기운 것. 이득 없음

── 나와도 시연 코드는 안 바꾼다 ────────────────────────────────────────

본선까지 며칠이고 백본을 갈면 뱅크·임계값·화질 기준 분포·실측값이 전부
무효가 된다. **"다음 단계는 이것이고 근거는 이 수치다"로 쓰는 자리**이지
지금 갈아끼우자는 측정이 아니다.

── 알아 둘 것 ──────────────────────────────────────────────────────────

DINOv2 가중치는 `torch.hub` 로 처음 한 번 내려받는다(약 90MB).
**네트워크가 막힌 상태에서는 후보 쪽이 안 돈다** — 최종 통합 테스트를
차단 상태로 하므로, 그 전에 한 번 받아 두거나 이 측정을 빼야 한다.
기준 쪽(wide_resnet50_2)은 이미 받아 둔 가중치라 영향이 없다.
"""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

from agents.adapters.config import build_adapters  # noqa: E402
from inspection import FeatureConfig, PatchEmbedder, build_bank  # noqa: E402

# 프롬프트·크롭 확대·어휘·이미지 선택을 **가져다 쓴다.** 여기에 다시 적으면
# 두 벌이 되고, 한쪽만 고쳐지면 여기 수치와 판별 1번 실측을 못 견준다.
from measure_vlm_prompt import (  # noqa: E402
    ask,
    defect_vocabulary,
    enlarge,
    pick,
    visa_root,
)

DINO_SIZE = 672  # 14의 배수라야 한다. 672 = 48x48 격자
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _DinoConfig:
    """`build_bank` 가 지문에 적을 설정. `FeatureConfig` 와 같은 자리를 채운다."""

    backbone = "dinov2_vits14"
    crop = DINO_SIZE
    layers = ("x_norm_patchtokens",)

    def fingerprint(self) -> dict:
        return {"backbone": self.backbone, "crop": self.crop,
                "layers": list(self.layers), "patch": 14}


class DinoEmbedder:
    """DINOv2 패치 토큰을 `PatchEmbedder` 와 같은 모양으로 돌려준다.

    `build_bank` 가 요구하는 것은 셋뿐이다 — `embed_paths` · `device` ·
    `config.fingerprint()`. 그래서 얇은 어댑터로 끼운다.

    **가중치를 처음 한 번 내려받는다.** 네트워크가 막혀 있으면 여기서 죽는다.
    """

    def __init__(self, size: int = DINO_SIZE):
        self.config = _DinoConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14",
                                    trust_repo=True, verbose=False)
        self.model.eval().to(self.device)
        self.size = size
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    @torch.no_grad()
    def embed_paths(self, paths, batch_size: int = 8):
        chunks: list[torch.Tensor] = []
        grid: tuple[int, int] | None = None
        for i in range(0, len(paths), batch_size):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
            x = torch.stack([self.tf(im) for im in imgs]).to(self.device)
            for im in imgs:
                im.close()
            tok = self.model.forward_features(x)["x_norm_patchtokens"]
            side = int(round(tok.shape[1] ** 0.5))
            grid = grid or (side, side)
            chunks.append(tok.cpu())
        if grid is None:
            raise ValueError("이미지를 하나도 못 읽었다")
        return torch.cat(chunks, 0), grid


SETUPS = {
    "wide_resnet50_2 512": lambda: PatchEmbedder(FeatureConfig()),
    "DINOv2 ViT-S/14 672": lambda: DinoEmbedder(),
}


@torch.no_grad()
def hottest_cell(embedder, bank_t: torch.Tensor, path: Path):
    """이 이미지에서 뱅크와 가장 먼 패치의 (행, 열) 과 격자 크기.

    **`inspection` 의 역추적과 같은 계산**이다 — 최근접 거리를 패치마다 재고
    가장 큰 것을 고른다. 여기서 다시 적은 것은 두 특징 추출기를 같은 코드로
    재기 위해서다(`PatchEmbedder` 쪽 경로는 설정이 더 붙어 있다).
    """
    patches, grid = embedder.embed_paths([path], batch_size=1)
    q = patches[0].to(embedder.device)
    d = torch.cdist(q, bank_t).min(dim=1).values
    idx = int(torch.argmax(d).item())
    return idx // grid[1], idx % grid[1], grid


def cell_box(row: int, col: int, grid, h: int, w: int, margin: int):
    """격자 칸을 원본 좌표의 상자로. 여유를 둘러 자른다."""
    ch, cw = h / grid[0], w / grid[1]
    y0 = max(0, int(row * ch) - margin)
    y1 = min(h, int((row + 1) * ch) + margin)
    x0 = max(0, int(col * cw) - margin)
    x1 = min(w, int((col + 1) * cw) + margin)
    return x0, y0, x1, y1


def center_on_defect(row: int, col: int, grid, mask: np.ndarray) -> bool:
    """칸의 **중심**이 결함 위인가. 가장 엄한 기준이다."""
    h, w = mask.shape
    cy = int((row + 0.5) * h / grid[0])
    cx = int((col + 0.5) * w / grid[1])
    return bool(0 <= cy < h and 0 <= cx < w and mask[cy, cx])

def crop_captures(row: int, col: int, grid, mask: np.ndarray, margin: int) -> bool:
    """여유를 둔 크롭이 결함을 **조금이라도** 담았는가.

    중심 적중보다 느슨하다. 시각 언어 모델에 보낼 조각에 결함이 들어 있느냐가
    실제로 중요한 것이라 둘을 나눠 센다.

    **칸이 커지면 이 값도 오른다.** 격자가 64x64 에서 48x48 로 성기어지므로
    크롭도 커지는데, 실측에서 넓이 차이가 9% 뿐이라 6/10 → 10/10 을 그것으로
    설명할 수 없다. 비교할 때 이 대조를 함께 적을 것.
    """
    h, w = mask.shape
    x0, y0, x1, y1 = cell_box(row, col, grid, h, w, margin)
    return bool(mask[y0:y1, x0:x1].any())


def make_bank(paths, embedder, args, tag: str):
    return build_bank(paths, embedder, coreset_ratio=args.coreset, seed=args.seed,
                      bank_version=tag, root=REPO_ROOT)


def images_for(root: Path, category: str, args):
    """결함·정상을 뽑는다.

    `--select sorted` 는 4090 의 1단계와 같은 이미지라 그 수치를 재현한다.
    `--select random` 은 결함 쪽이 `measure_vlm_prompt` 와 같은 난수라 판별
    1번 실측(마스크 크롭 80.8% · 대조군 11%)과 **같은 결함 이미지**가 나온다.

    **정상은 어느 쪽이든 뱅크에 안 쓴 것에서만 뽑는다.** `pick` 은 정상을
    전체에서 뽑는데 그 함수를 쓰는 곳은 뱅크를 세우지 않아 상관이 없었다.
    여기서 그대로 가져다 쓰면 뱅크에 든 이미지를 과검 측정에 넣게 되고,
    자기가 만든 뱅크로 자기를 재니 과검이 실제보다 낮게 나온다.
    """
    base = root / category / "Data"
    normals = sorted((base / "Images" / "Normal").glob("*"))
    bank_paths, spare = normals[:args.bank], normals[args.bank:]
    if args.select == "sorted":
        anomalies = sorted((base / "Images" / "Anomaly").glob("*"))[:args.count]
        return bank_paths, anomalies, spare[:args.count]
    rng = random.Random(args.seed)
    anomalies, _ = pick(root, category, args.count, rng)
    held_out = rng.sample(spare, min(args.count, len(spare)))
    return bank_paths, list(anomalies), held_out


def measure(category: str, embedder, args, root: Path, tag: str, vlm, tmp: Path | None):
    """한 카테고리를 한 특징 추출기로 잰다. `vlm` 이 None 이면 좌표만."""
    bank_paths, anomalies, held_out = images_for(root, category, args)
    bank = make_bank(bank_paths, embedder, args, f"{category}-{tag}")
    bank_t = torch.from_numpy(bank.embeddings).to(embedder.device)
    masks = root / category / "Data" / "Masks" / "Anomaly"
    defects = ", ".join(defect_vocabulary(root, category))

    center = captured = hit = counted = 0
    grid_seen = None
    coords: list[str] = []
    for path in anomalies:
        mask_path = masks / (path.stem + ".png")
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        row, col, grid = hottest_cell(embedder, bank_t, path)
        grid_seen = grid
        center += int(center_on_defect(row, col, grid, mask))
        captured += int(crop_captures(row, col, grid, mask, args.margin))
        coords.append(f"({row},{col})")
        counted += 1
        if vlm is not None:
            hit += int(_judge(path, row, col, grid, args, vlm, defects, tmp) == "visible")

    false_alarm = 0
    if vlm is not None:
        for path in held_out:
            row, col, grid = hottest_cell(embedder, bank_t, path)
            false_alarm += int(_judge(path, row, col, grid, args, vlm, defects, tmp) == "visible")

    return {
        "center": center, "captured": captured, "counted": counted,
        "hit": hit, "false_alarm": false_alarm, "normals": len(held_out),
        "grid": f"{grid_seen[0]}x{grid_seen[1]}" if grid_seen else "—",
        "coords": coords,
    }


def _judge(path: Path, row: int, col: int, grid, args, vlm, defects: str, tmp: Path) -> str:
    """짚은 자리를 잘라 물어본다. **정상에는 마스크가 없으니 그냥 그 자리다.**"""
    with Image.open(path) as img:
        img = img.convert("RGB")
        box = cell_box(row, col, grid, img.height, img.width, args.margin)
        crop = enlarge(img.crop(box), args.enlarge)
    out = tmp / f"{path.stem}_{row}_{col}.png"
    crop.save(out)
    try:
        return ask(vlm, out, defects)
    finally:
        out.unlink(missing_ok=True)


def run(args, root: Path, vlm, tmp: Path | None) -> int:
    judging = vlm is not None
    print("역추적 위치 — 특징 추출기 비교")
    print(f"뱅크 정상 {args.bank}장 · 결함 {args.count}장 · coreset {args.coreset} · "
          f"seed {args.seed} · 여유 {args.margin}px · 선택 {args.select}")
    print("**inspection/ 은 건드리지 않는다. 뱅크는 저장되지 않는다.**\n")

    header = (f"{'카테고리':<12} {'특징 추출기':<22} {'격자':>7} "
              f"{'중심 적중':>9} {'크롭 담김':>9}")
    if judging:
        header += f" {'검출':>7} {'과검':>7}"
    header += "   소요"
    print(header)
    print("-" * len(header))

    results: dict[tuple[str, str], dict] = {}
    for label, make in SETUPS.items():
        embedder = make()
        for category in args.categories:
            started = time.time()
            r = measure(category, embedder, args, root, label.split()[0], vlm, tmp)
            results[(category, label)] = r
            line = (f"{category:<12} {label:<22} {r['grid']:>7} "
                    f"{r['center']:>6}/{r['counted']:<2} {r['captured']:>6}/{r['counted']:<2}")
            if judging:
                line += f" {r['hit']:>4}/{r['counted']:<2} {r['false_alarm']:>4}/{r['normals']:<2}"
            print(line + f"   {time.time() - started:.0f}초")
            if not judging:
                print(f"{'':12} {'좌표':<22} {' '.join(r['coords'])}")
        del embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("-" * len(header))
    print("\n비교")
    labels = list(SETUPS)
    for category in args.categories:
        a, b = results[(category, labels[0])], results[(category, labels[1])]
        line = (f"  {category:<12} 중심 {a['center']}/{a['counted']} → {b['center']}/{b['counted']}"
                f"   담김 {a['captured']}/{a['counted']} → {b['captured']}/{b['counted']}")
        if judging:
            line += (f"   검출 {a['hit']}/{a['counted']} → {b['hit']}/{b['counted']}"
                     f"   과검 {a['false_alarm']}/{a['normals']} → {b['false_alarm']}/{b['normals']}")
        print(line)

    if judging:
        print("\n  담김↑ 검출↑ 과검그대로 → 교체가 답이다")
        print("  담김↑ 검출제자리      → 크롭 안에 있어도 못 본다. 바꿔도 소용없다")
        print("  검출↑ 과검도↑         → visible 쪽으로 기운 것. 이득 없음")
    else:
        print("\n  좌표까지다. 판별 1번이 실제로 좋아지는지는 --mode judge 가 답한다.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="특징 추출기별 역추적 위치")
    parser.add_argument("--mode", choices=("trace", "judge"), required=True)
    parser.add_argument("--categories", nargs="*", default=["pcb1", "capsules"])
    parser.add_argument("--visa-root", default=None,
                        help="비우면 저장소 아래 VisA_20220922")
    parser.add_argument("--count", type=int, default=10, help="카테고리당 결함·정상 장수")
    parser.add_argument("--bank", type=int, default=150, help="뱅크에 넣을 정상 장수")
    parser.add_argument("--margin", type=int, default=64, help="크롭 여유 픽셀")
    parser.add_argument("--enlarge", type=int, default=512, help="크롭 확대 목표")
    parser.add_argument("--coreset", type=float, default=0.01)
    parser.add_argument("--select", choices=("sorted", "random"), default="sorted",
                        help="sorted 는 4090 1단계 재현, random 은 판별 1번 실측과 같은 이미지")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = visa_root(args)
    if not root.is_dir():
        print(f"VisA 를 찾지 못했다: {root}")
        return 1

    if args.mode == "trace":
        return run(args, root, None, None)

    _, vlm = build_adapters()
    if vlm.is_stub:
        print("시각 언어 모델이 연결되지 않았다. 스텁으로는 이 측정이 의미가 없다.")
        print("SHVO_VLM_PROVIDER 등을 설정하고 scripts/check_models.py 로 확인할 것.")
        return 1
    print(f"시각 언어 모델 {vlm.describe()}\n")
    with tempfile.TemporaryDirectory(prefix="shvo_backbone_") as tmp:
        return run(args, root, vlm, Path(tmp))


if __name__ == "__main__":
    raise SystemExit(main())
