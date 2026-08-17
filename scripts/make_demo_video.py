"""시연 화면을 스크롤 영상으로 만든다 — `docs/시연각본.md` 의 여섯 장면.

    .venv/bin/python scripts/make_demo_video.py --page <내보낸.html> --out demo.mp4

**이것은 보험이다.** 사람이 브라우저로 직접 녹화하는 편이 낫다 — 마우스가
보이고, 신구 비교 재생 애니메이션이 돌고, 실행을 기다리는 장면이 산다.
이 영상에는 그 셋이 없다. 다만 녹화가 안 되는 사정이 있을 때(시연 장비가
화면 캡처에 워터마크를 박는다) 손 대지 않고 나오는 결과물이 필요하다.

── 어떻게 만드나 ───────────────────────────────────────────────────────

내보낸 화면을 헤드리스 브라우저로 한 장의 긴 그림으로 그린 뒤, 그 위를
창 크기만큼 훑어 내려가며 프레임을 만든다. 장면마다 머물 자리와 자막은
각본에서 가져온다.

**자막이 화면에 없는 말을 하지 않는다.** 각본이 지키는 원칙이 그대로
적용된다. 숫자는 화면 집계에서 옮긴 것이고 여기서 계산하지 않는다.

── 왜 브라우저를 다시 부르나 ───────────────────────────────────────────

이미 그려 둔 그림이 있어도, 블록이 어디쯤 있는지를 알아야 장면마다 멈출
자리를 정한다. 좌표는 브라우저가 알려 준다. 그림만 있으면 눈으로 찍어야
하고, 화면이 바뀔 때마다 다시 찍어야 한다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]

#: 장면마다 (블록 id, 자막 여러 줄, 머무는 초).
#:
#: **`docs/시연각본.md` 와 같은 문안이다.** 두 벌이 되면 한쪽만 고쳐진다.
#: 각본을 고치면 여기도 고친다.
SCENES: list[tuple[str, list[str], float]] = [
    ("runform", [
        "현장에서 올라온 이슈는 이미지가 아니라 제품명과 로트입니다.",
        "자연어 한 줄이면 됩니다. 라인과 품목은 언어 모델이 여기서 뽑습니다.",
        "정보가 모자라면 추측하지 않고 되묻습니다.",
    ], 13),
    ("cap-rail", [
        "접수부터 승인 요청까지 열 단계입니다.",
        "각 단계는 도구 하나이고, 어떤 순서로 부를지는 언어 모델이 정합니다.",
        "순서를 어기면 도구가 거부합니다.",
    ], 16),
    ("block-evidence", [
        "못 잡은 이미지가 뱅크의 어느 정상 패치와 가까웠는지 되짚을 수 있습니다.",
        "판단 근거가 모델 안에 이미 있습니다.",
        "결함이면 뱅크 오염 하나로 정해집니다.",
        "진짜 정상품이면 셋으로 나뉘고, 판별 6번과 임계값 스윕이 그것을 가릅니다.",
    ], 22),
    ("block-bank", [
        "뱅크 118장 가운데 맨 앞 두 장이 뺄 것입니다.",
        "정상만 들어가야 하는데 결함이 섞여 있었습니다.",
        "섞인 것이 같은 유형의 불량을 정상 쪽으로 끌어당겨 못 잡게 만들었습니다.",
    ], 18),
    ("block-simulator", [
        "새 뱅크를 실제 판정에 쓰지 않습니다.",
        "같은 이미지에 두 뱅크를 나란히 돌려 판정이 서로 다른 것만 뽑습니다.",
        "14장 중 1장만 사람이 확인하면 됩니다.",
    ], 18),
    ("stage-release", [
        "배포 패키지와 승인 요청 문서까지 만듭니다.",
        "배포는 실행되지 않습니다. 배포 승인: 아니오. 사람이 결정합니다.",
        "품질 검사 설비라 의도적으로 뺀 경계입니다.",
    ], 20),
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if pathlib.Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def render(page: pathlib.Path, work: pathlib.Path, width: int) -> tuple[Image.Image, dict]:
    """화면을 한 장의 긴 그림으로 그리고 블록 좌표를 받는다."""
    ids = [scene[0] for scene in SCENES]
    js = ("<script>(function(){var out={};" + json.dumps(ids) +
          ".forEach(function(id){var el=document.getElementById(id);if(!el){return;}"
          "var r=el.getBoundingClientRect();"
          "out[id]=[Math.round(r.top+scrollY),Math.round(r.bottom+scrollY)];});"
          'document.title="BOX"+JSON.stringify(out);})();</script>')
    text = page.read_text(encoding="utf-8")
    text = text.replace("max-height:460px}", "max-height:none}")
    text = text.replace("<details", "<details open").replace("<details open open", "<details open")
    text = text.replace('<ol class="rail">', '<ol class="rail" id="cap-rail">', 1)
    prepared = work / "_page.html"
    prepared.write_text(text.replace("</body>", js + "</body>"), encoding="utf-8")

    shot = work / "_page.png"
    base = [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
            f"--window-size={width},34000"]
    subprocess.run(base + [f"--screenshot={shot}", f"file://{prepared}"],
                   capture_output=True, timeout=600)
    dom = subprocess.run(base + ["--virtual-time-budget=7000", "--dump-dom",
                                 f"file://{prepared}"],
                         capture_output=True, text=True, timeout=600).stdout
    match = re.search(r"<title>BOX(.*?)</title>", dom, re.S)
    boxes = json.loads(match.group(1)) if match else {}
    return Image.open(shot).convert("RGB"), boxes


def caption(frame: Image.Image, lines: list[str], font, small) -> Image.Image:
    """자막을 아래쪽에 얹는다. 화면을 가리지 않게 반투명 띠를 깐다."""
    out = frame.convert("RGBA")
    band = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(band)
    pad, gap = 28, 8
    heights = [draw.textbbox((0, 0), s, font=font)[3] for s in lines]
    height = sum(heights) + gap * (len(lines) - 1) + pad * 2
    top = out.height - height
    draw.rectangle([0, top, out.width, out.height], fill=(16, 22, 30, 216))
    y = top + pad
    for line, h in zip(lines, heights):
        draw.text((pad + 8, y), line, font=font, fill=(240, 244, 248, 255))
        y += h + gap
    return Image.alpha_composite(out, band).convert("RGB")


def main() -> int:
    parser = argparse.ArgumentParser(description="시연 화면을 스크롤 영상으로")
    parser.add_argument("--page", required=True, help="내보낸 화면 HTML")
    parser.add_argument("--out", default="demo.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    page = pathlib.Path(args.page)
    if not page.exists():
        print(f"화면 파일을 찾지 못했다: {page}")
        return 1
    if not pathlib.Path(CHROME).exists():
        print("크롬을 찾지 못했다. 이 스크립트는 맥에서만 돈다.")
        return 1

    work = pathlib.Path(tempfile.mkdtemp(prefix="shvo_video_"))
    try:
        full, boxes = render(page, work, args.width)
        missing = [s[0] for s in SCENES if s[0] not in boxes]
        if missing:
            # 조용히 건너뛰면 짧은 영상이 나오고 왜 짧은지 모른다.
            print(f"화면에 없는 블록: {', '.join(missing)}")
        scale = args.width / full.width
        page_img = full.resize((args.width, round(full.height * scale)), Image.LANCZOS)

        font, small = find_font(27), find_font(21)
        writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (args.width, args.height))
        if not writer.isOpened():
            print("영상 파일을 열지 못했다.")
            return 1

        limit = max(0, page_img.height - args.height)
        position = 0.0
        total = 0
        for block, lines, seconds in SCENES:
            if block not in boxes:
                continue
            top = round(boxes[block][0] * scale) - 60
            target = float(min(max(0, top), limit))
            # 다음 자리까지 흘러간 뒤 머문다. 뚝 끊기면 어디로 갔는지 모른다.
            glide = max(1, int(args.fps * 1.6))
            hold = max(1, int(args.fps * (seconds - 1.6)))
            for i in range(glide):
                y = position + (target - position) * (i + 1) / glide
                frame = page_img.crop((0, round(y), args.width, round(y) + args.height))
                writer.write(np.array(caption(frame, lines, font, small))[:, :, ::-1])
                total += 1
            position = target
            frame = page_img.crop((0, round(position), args.width,
                                   round(position) + args.height))
            painted = np.array(caption(frame, lines, font, small))[:, :, ::-1]
            for _ in range(hold):
                writer.write(painted)
                total += 1
        writer.release()
        print(f"{args.out}  {total / args.fps:.0f}초 · {total}프레임 · "
              f"{pathlib.Path(args.out).stat().st_size // 1024}KB")
        print("**사람이 브라우저로 찍은 것이 더 낫다.** 이것은 보험이다 — "
              "마우스도, 재생 애니메이션도, 실행을 기다리는 장면도 없다.")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
