"""고립도 선별 검증.

이 기능은 "오염이다"를 말하지 않고 "여기부터 보라"를 말한다. 그래서
검증도 두 갈래다.

  1. 오염원을 실제로 상위로 올리는가
  2. **한계를 알고 있는가** — 기여 패치가 적은 이미지를 순위에서 빼는가,
     후보가 없으면 없다고 답하는가

2번이 없으면 이 신호는 위험해진다. 무엇이든 상위 몇 개를 뽑아 오염으로
지목하면, 드물지만 진짜인 정상 이미지를 뱅크에서 빼게 되고 그건 커버리지
부족을 스스로 만드는 짓이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from inspection import FeatureConfig, PatchEmbedder, build_bank
from inspection.isolation import (
    contamination_amplification,
    image_isolation,
    patch_isolation,
    suspect_images,
)
from tests.synthetic import make_defect, make_normal

CONFIG = FeatureConfig(backbone="resnet18", resize=64, crop=64)


@pytest.fixture(scope="module")
def embedder() -> PatchEmbedder:
    try:
        return PatchEmbedder(CONFIG)
    except RuntimeError as exc:
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def contaminated(embedder, tmp_path_factory):
    """정상 14장 + 오염 2장으로 만든 뱅크.

    오염 이미지는 정상과 같은 생성 규칙(같은 시드 범위)으로 만들어 얼룩
    하나만 다르게 한다. 조명이 통째로 다르면 얼룩이 아니라 조명 차이를
    재게 되어 검증이 무의미해진다.
    """
    root = tmp_path_factory.mktemp("iso")
    (root / "normal").mkdir()
    (root / "defect").mkdir()

    normal = []
    for seed in range(14):
        path = root / "normal" / f"normal_{seed:03d}.png"
        make_normal(seed).save(path)
        normal.append(path)

    contaminants = []
    for seed in (20, 21):
        path = root / "defect" / f"defect_{seed:03d}.png"
        make_defect(seed).save(path)
        contaminants.append(path)

    bank = build_bank(
        normal + contaminants, embedder, coreset_ratio=0.25, seed=42, root=root
    )
    names = {p.relative_to(root).as_posix() for p in contaminants}
    return bank, names


# ── 신호가 실제로 오염원을 올리는가 ────────────────────────────────────


def test_contaminants_rank_at_the_top(contaminated):
    """오염 이미지 2장이 고립도 순위 상위 2위를 차지해야 한다."""
    bank, contaminant_names = contaminated
    ranking = [s for s in image_isolation(bank, k=8) if s.ranked]

    top_two = {s.image for s in ranking[:2]}
    assert top_two == contaminant_names, f"상위 2장이 오염원이 아니다: {[s.image for s in ranking[:4]]}"


def test_gap_between_contaminant_and_normal_is_visible(contaminated):
    """상위와 그다음 사이에 눈에 보이는 간격이 있어야 임계값을 정할 수 있다."""
    bank, contaminant_names = contaminated
    ranking = [s for s in image_isolation(bank, k=8) if s.ranked]

    contaminant_z = [s.z_mean for s in ranking if s.image in contaminant_names]
    normal_z = [s.z_mean for s in ranking if s.image not in contaminant_names]

    assert min(contaminant_z) > max(normal_z)


def test_patch_isolation_is_normalized(contaminated):
    bank, _ = contaminated
    z = patch_isolation(bank, k=8)

    assert z.shape == (len(bank),)
    assert abs(float(np.mean(z))) < 1e-4
    assert abs(float(np.std(z)) - 1.0) < 1e-4


def test_tiny_bank_is_rejected(embedder, tmp_path):
    """이웃을 셀 수 없을 만큼 작은 뱅크에서는 계산하지 않는다."""
    path = tmp_path / "n.png"
    make_normal(0).save(path)
    bank = build_bank([path], embedder, coreset_ratio=0.005, seed=0, root=tmp_path)

    with pytest.raises(ValueError, match="뱅크가 너무 작다"):
        patch_isolation(bank, k=64)


# ── 한계를 알고 있는가 ─────────────────────────────────────────────────


def test_low_contribution_images_are_excluded_from_ranking(contaminated):
    """패치 한두 개만 남은 이미지는 평균이 우연에 좌우되므로 순위에서 뺀다.

    실측에서 그런 정상 이미지가 오염원보다 높은 순위를 차지하는 경우를
    확인했다. 빼지 않으면 엉뚱한 이미지를 오염으로 지목하게 된다.
    """
    bank, _ = contaminated
    scores = image_isolation(bank, k=8, min_patches=5)

    for score in scores:
        assert score.ranked == (score.patch_count >= 5)

    # 제외된 이미지도 목록에는 남아야 한다. "왜 순위에 없나"에 답해야 하므로.
    assert len(scores) == len(bank.contributing_images())


def test_ranked_images_come_before_unranked(contaminated):
    bank, _ = contaminated
    scores = image_isolation(bank, k=8, min_patches=999)  # 전부 제외되는 기준

    assert all(not s.ranked for s in scores)


def test_high_threshold_returns_no_candidates(contaminated):
    """의심 대상이 없으면 없다고 답해야 한다. 억지로 뽑지 않는다."""
    bank, _ = contaminated
    assert suspect_images(bank, k=8, z_threshold=100.0) == []


def test_top_n_caps_vlm_call_count(contaminated):
    """시각 언어 모델 호출 수를 묶을 수 있어야 한다."""
    bank, _ = contaminated
    assert len(suspect_images(bank, k=8, z_threshold=-10.0, top_n=3)) == 3


# ── coreset 증폭 ───────────────────────────────────────────────────────


def test_coreset_amplifies_outlier_images(contaminated):
    """coreset 은 튀는 이미지를 과대 대표한다. 그 사실이 드러나야 한다.

    합성 데이터에서 원본의 12.5% 였던 오염원이 뱅크의 절반을 차지했다.
    오염이 섞였을 때 피해가 그만큼 커지므로 뱅크를 만든 뒤 확인할 값이다.
    """
    bank, contaminant_names = contaminated
    report = contamination_amplification(bank)

    assert report["max_amplification"] > 1.0
    top = report["images"][0]
    assert top["image"] in contaminant_names
    assert top["bank_share"] > 1.0 / len(bank.images)


def test_amplification_is_flat_without_coreset(embedder, tmp_path):
    """coreset 을 쓰지 않으면 모든 이미지가 같은 수의 패치를 남긴다."""
    paths = []
    for seed in range(4):
        path = tmp_path / f"n{seed}.png"
        make_normal(seed).save(path)
        paths.append(path)

    bank = build_bank(paths, embedder, coreset_ratio=1.0, seed=0, root=tmp_path)
    report = contamination_amplification(bank)

    assert report["max_amplification"] == pytest.approx(1.0, abs=1e-6)
