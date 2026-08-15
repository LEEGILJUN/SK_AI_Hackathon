"""추론과 역추적 검증.

핵심은 마지막 테스트다. 정상 학습셋에 결함이 섞이면 같은 종류의 결함이
정상으로 판정되는데, 그때 최근접 패치를 되짚으면 **섞여 들어간 그 이미지가
지목되는가**를 확인한다. 이 과제의 가치가 성립하는지를 재는 자리다.

백본은 resnet18 을 쓴다. 실제 뱅크는 wide_resnet50_2 이지만, 테스트는
가중치 캐시만으로 빠르게 돌아야 하므로 가벼운 쪽을 택했다. 검증하는 성질은
백본과 무관하다.
"""

from __future__ import annotations

import pytest
import torch

from inspection import (
    FeatureConfig,
    MemoryBank,
    PatchEmbedder,
    bank_contribution,
    build_bank,
    score_image,
)
from tests.synthetic import write_set

# 테스트용 경량 설정. 격자 8×8 = 패치 64개라 몇 초 안에 끝난다.
TEST_CONFIG = FeatureConfig(
    backbone="resnet18",
    layers=("layer2", "layer3"),
    weights="IMAGENET1K_V1",
    crop=64,
)


@pytest.fixture(scope="module")
def embedder() -> PatchEmbedder:
    try:
        return PatchEmbedder(TEST_CONFIG)
    except RuntimeError as exc:  # 가중치 캐시도 네트워크도 없는 환경
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def images(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("factory")
    return {
        "root": root,
        "normal": write_set(root / "normal", 12, "normal", seed_offset=0),
        "defect": write_set(root / "defect", 4, "defect", seed_offset=500),
    }


@pytest.fixture(scope="module")
def clean_bank(embedder, images) -> MemoryBank:
    """정상 이미지만으로 만든 뱅크."""
    return build_bank(
        images["normal"],
        embedder,
        coreset_ratio=0.25,
        seed=42,
        bank_version="clean-v1",
        root=images["root"],
    )


# ── 역추적 대장이 맞게 만들어지는가 ────────────────────────────────────


def test_provenance_points_back_to_source(clean_bank):
    """뱅크 행 번호로 원본 이미지와 격자 좌표를 되짚을 수 있어야 한다.

    이게 깨지면 진단 전체가 성립하지 않으므로 가장 먼저 확인한다.
    """
    assert len(clean_bank) > 0

    grid_h, grid_w = clean_bank.grid
    for row_index in range(0, len(clean_bank), max(1, len(clean_bank) // 20)):
        ref = clean_bank.origin_of(row_index)
        assert ref.source_image in clean_bank.images
        assert 0 <= ref.row < grid_h
        assert 0 <= ref.col < grid_w
        # 평탄 인덱스와 좌표가 어긋나면 순서가 뒤섞인 것이다.
        assert ref.patch_index == ref.row * grid_w + ref.col


def test_coreset_reduces_and_is_deterministic(embedder, images):
    """같은 입력과 시드면 같은 뱅크가 나와야 한다. 재현성 목표의 전제다."""
    kwargs = dict(coreset_ratio=0.25, seed=7, root=images["root"])
    first = build_bank(images["normal"], embedder, **kwargs)
    second = build_bank(images["normal"], embedder, **kwargs)

    assert (first.origins == second.origins).all()
    assert torch.allclose(
        torch.from_numpy(first.embeddings), torch.from_numpy(second.embeddings)
    )

    # 전체를 다 담지 않고 실제로 솎아야 한다.
    assert len(first) < first.meta["total_patches_before_coreset"]


def test_contributing_images_covers_every_source(clean_bank):
    """뱅크 구성 이력 조회 — 판별 항목 6번의 재료."""
    counts = clean_bank.contributing_images()
    assert set(counts) == set(clean_bank.images)
    assert sum(counts.values()) == len(clean_bank)


def test_bank_save_load_roundtrip(clean_bank, tmp_path):
    """뱅크 파일은 커밋하지 않으므로 저장·복원이 정확해야 한다."""
    saved = clean_bank.save(tmp_path / "bank")
    assert (saved / "bank_meta.json").exists()

    loaded = MemoryBank.load(saved)
    assert loaded.images == clean_bank.images
    assert loaded.version == clean_bank.version
    assert (loaded.origins == clean_bank.origins).all()
    assert loaded.origin_of(0) == clean_bank.origin_of(0)


def test_incompatible_config_is_rejected(clean_bank):
    """설정이 다르면 점수가 조용히 틀리므로 멈춰야 한다."""
    with pytest.raises(ValueError, match="뱅크와 추론 설정이 다르다"):
        clean_bank.assert_compatible(FeatureConfig(backbone="resnet34", crop=64))


# ── 추론이 결함을 가려내는가 ───────────────────────────────────────────


def test_defect_scores_higher_than_normal(embedder, images, clean_bank):
    """깨끗한 뱅크라면 결함이 정상보다 높은 점수를 받아야 한다."""
    normal_score = score_image(
        images["normal"][0], clean_bank, embedder, root=images["root"]
    ).score
    defect_score = score_image(
        images["defect"][0], clean_bank, embedder, root=images["root"]
    ).score

    assert defect_score > normal_score


def test_result_shape_and_score_position(embedder, images, clean_bank):
    """진단 에이전트에 넘길 구조가 갖춰졌는지."""
    result = score_image(images["defect"][0], clean_bank, embedder, root=images["root"], top_k=3)

    grid_h, grid_w = clean_bank.grid
    assert (result.grid_h, result.grid_w) == (grid_h, grid_w)
    assert len(result.patch_distances) == grid_h
    assert len(result.matches) == 3
    assert result.bank_version == clean_bank.version

    # 거리 내림차순이어야 matches[0] 이 판정을 좌우한 자리가 된다.
    distances = [m.distance for m in result.matches]
    assert distances == sorted(distances, reverse=True)

    # 판별 항목 3번 — 임계값 대비 위치
    assert result.score_position(threshold=result.score * 0.5) == "above"
    assert result.score_position(threshold=result.score * 10) == "below"

    assert "matches" in result.to_dict()


# ── 뱅크 오염: 이 과제의 핵심 시나리오 ─────────────────────────────────


def test_contaminated_bank_misses_defect_and_traces_to_contaminant(embedder, images):
    """정상셋에 결함이 섞이면 같은 결함을 놓치고, 역추적이 그 원인을 지목한다.

    시연에서 보여줄 장면 그대로다.
      1. 뱅크에 결함 이미지가 섞여 들어간다
      2. 같은 종류의 결함이 정상으로 판정된다 (점수가 떨어진다)
      3. 최근접 패치를 되짚으면 섞여 들어간 그 이미지가 나온다

    3번이 되지 않으면 원인을 사람이 찾아야 하고, 이 서비스의 값어치가 없다.
    """
    contaminants = images["defect"][:2]
    query = images["defect"][3]  # 뱅크에 넣지 않은 별개의 결함

    clean = build_bank(
        images["normal"], embedder, coreset_ratio=0.25, seed=42,
        bank_version="clean-v1", root=images["root"],
    )
    contaminated = build_bank(
        list(images["normal"]) + list(contaminants),
        embedder, coreset_ratio=0.25, seed=42,
        bank_version="contaminated-v2", root=images["root"],
    )

    clean_result = score_image(query, clean, embedder, root=images["root"])
    dirty_result = score_image(query, contaminated, embedder, root=images["root"])

    # 2. 오염된 뱅크에서 점수가 떨어진다 = 놓치기 시작한다
    assert dirty_result.score < clean_result.score, (
        f"오염이 점수를 낮추지 못했다. clean={clean_result.score:.4f} "
        f"contaminated={dirty_result.score:.4f}"
    )

    # 3. 최근접 패치가 섞여 들어간 이미지를 지목한다
    top = dirty_result.top_match
    assert top is not None
    contaminant_names = {p.relative_to(images["root"]).as_posix() for p in contaminants}
    assert top.bank.source_image in contaminant_names, (
        f"역추적이 오염원을 지목하지 못했다. 지목={top.bank.source_image} "
        f"오염원={contaminant_names}"
    )

    # 되짚은 좌표가 실제 격자 안에 있어야 한다
    grid_h, grid_w = contaminated.grid
    assert 0 <= top.bank.row < grid_h and 0 <= top.bank.col < grid_w


def test_bank_contribution_ranks_the_contaminant(embedder, images):
    """여러 미검출 건이 같은 뱅크 이미지를 가리키면 그게 오염 후보다.

    한 장은 우연일 수 있으므로 반복이 근거가 된다.
    """
    contaminants = images["defect"][:2]
    contaminated = build_bank(
        list(images["normal"]) + list(contaminants),
        embedder, coreset_ratio=0.25, seed=42,
        bank_version="contaminated-v2", root=images["root"],
    )

    results = [
        score_image(p, contaminated, embedder, root=images["root"])
        for p in images["defect"][2:]
    ]
    ranking = bank_contribution(results)

    assert ranking, "최근접 출처 집계가 비었다"
    contaminant_names = {p.relative_to(images["root"]).as_posix() for p in contaminants}
    top_source = next(iter(ranking))
    assert top_source in contaminant_names, f"가장 많이 지목된 출처가 오염원이 아니다: {ranking}"


# ── 정사각 리사이즈 ─────────────────────────────────────────────────────
#
# 중앙 크롭을 하면 VisA 는 가로가 길어 양옆이 잘린다. 정답 마스크로 세니
# pcb3 40건 중 12건 일부 잘림, capsules 1건 완전 소실이었다. 사라진 건
# 어떤 모델로도 검출할 수 없다.
#
# 카테고리마다 다르게 맞추지 않아도 되는 것이 더 중요하다. VisA 는
# 카테고리가 많고, 이 과제는 전처리 성능 비교가 목적이 아니다.


def test_a_wide_image_is_not_clipped():
    """가로가 긴 이미지도 통째로 들어간다."""
    import numpy as np
    from PIL import Image

    from inspection import FeatureConfig, PatchEmbedder

    wide = Image.fromarray((np.random.rand(1000, 1500, 3) * 255).astype("uint8"))
    embedder = PatchEmbedder(FeatureConfig())
    assert tuple(embedder.transform(wide).shape) == (3, 512, 512)


def test_there_is_only_one_size_to_configure():
    """맞출 값이 `crop` 하나다.

    짧은 변 기준 리사이즈 값과 크롭 값을 따로 두면 카테고리마다 두 값을
    맞춰야 하고, **어긋나면 조용히 잘린다.**
    """
    from dataclasses import fields

    from inspection import FeatureConfig

    names = {f.name for f in fields(FeatureConfig)}
    assert "resize" not in names, "크기를 정하는 값이 둘이면 어긋난다"
    assert "crop" in names


def test_a_bank_from_a_removed_setting_is_refused():
    """**없어진 설정도 잡아야 한다.**

    전처리를 정사각 리사이즈로 바꾸며 `resize` 필드를 없앴다. 현재 키만
    훑으면 예전 뱅크의 `resize` 를 안 보고, 나머지 값이 같아서 **중앙 크롭
    시절 뱅크가 그대로 통과한다.** 거리 척도가 다른데 오류가 안 난다.
    """
    import pytest

    from inspection import FeatureConfig
    from inspection.bank import MemoryBank

    bank = MemoryBank.__new__(MemoryBank)
    bank.meta = {"feature_config": dict(FeatureConfig().fingerprint(), resize=512)}

    with pytest.raises(ValueError, match="resize"):
        bank.assert_compatible(FeatureConfig())
