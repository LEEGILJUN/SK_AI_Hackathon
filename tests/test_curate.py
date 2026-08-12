"""큐레이션과 재구성 검증 — 작업 15·16.

제일 중요한 것은 **하지 말아야 할 때 안 하는가**다. 여섯 원인 중 넷은
뱅크 재구성이 답이 아니고, 그 넷에서 재구성이 실행되면 정량 목표
"재구성이 답이 아닌 케이스 전건 차단"이 깨진다.

두 번째는 **근거가 겹치는지 보는가**다. 오염 후보를 지목할 때 역추적과
고립도가 함께 가리키면 확신이 올라가고, 한 갈래뿐이면 사람 확인이 붙어야
한다. 정상 이미지를 잘못 빼면 커버리지 부족을 스스로 만드는 셈이다.
"""

from __future__ import annotations

import pytest

from agents.curate import CurationPlan, plan_curation
from agents.diagnose import DiagnosisResult, Evidence
from agents.rebuild import DirectoryImageSource, compare_banks, execute_rebuild
from inspection import FeatureConfig, PatchEmbedder, build_bank
from inspection.types import InferenceResult, NearestMatch, PatchRef
from tests.synthetic import write_set

CONFIG = FeatureConfig(backbone="resnet18", resize=64, crop=64)


# ── helper ─────────────────────────────────────────────────────────────


def diagnosis(cause, *, nearest_image=None, patch_verdict=None, rebuild=None):
    from agents.diagnose import REBUILD_REQUIRED

    evidence = []
    if nearest_image:
        evidence.append(
            Evidence(
                item_no=4, name="nearest_patch",
                value={"source_image": nearest_image, "row": 1, "col": 2, "bank_row_index": 5},
                source="trace", usable=True, detail=f"{nearest_image} 격자(1,2)",
            )
        )
    if patch_verdict:
        evidence.append(
            Evidence(item_no=5, name="nearest_patch_is_defect", value=patch_verdict,
                     source="vlm", usable=True, detail="판독함")
        )
    return DiagnosisResult(
        cause=cause,
        requires_bank_rebuild=REBUILD_REQUIRED.get(cause) if rebuild is None else rebuild,
        confidence="high",
        needs_human=False,
        evidence=evidence,
    )


def missed(bank_image: str) -> InferenceResult:
    match = NearestMatch(
        query=PatchRef("q.png", 1, 1, 9),
        bank=PatchRef(bank_image, 2, 2, 18),
        distance=2.0,
        bank_row_index=7,
    )
    return InferenceResult(
        image="q.png", score=1.5, max_patch_distance=1.5,
        grid_h=8, grid_w=8, matches=[match], bank_version="v3",
    )


# ── 하지 말아야 할 때 멈추는가 ─────────────────────────────────────────


@pytest.mark.parametrize("cause", ["threshold", "normal_overlap", "equipment_optics", "criteria"])
def test_rebuild_is_blocked_for_four_causes(cause):
    """재구성이 답이 아닌 넷에서는 계획 자체가 뱅크를 건드리지 않는다."""
    plan = plan_curation(diagnosis(cause))

    assert plan.touches_bank is False
    assert plan.is_empty
    assert "답이 아니다" in plan.reason
    assert plan.alternative_actions, "대신 무엇을 해야 하는지 제시해야 한다"


def test_withheld_diagnosis_produces_no_plan():
    """진단이 보류되면 조치도 계획하지 않는다."""
    withheld = DiagnosisResult(
        cause=None, requires_bank_rebuild=None, confidence="none", needs_human=True,
        blocking_reason="판별 5번을 얻지 못했다.",
    )
    plan = plan_curation(withheld)

    assert plan.touches_bank is False
    assert plan.needs_human is True
    assert "판별 5번" in plan.reason


def test_blocked_plan_is_refused_by_rebuild(tmp_path):
    """계획이 막았으면 실행 단계에서도 거부되어야 한다.

    언어 모델이 재구성을 부르더라도 여기서 멈춘다. 안전 장치가 두 겹이다.
    """
    plan = plan_curation(diagnosis("normal_overlap"))
    embedder = PatchEmbedder(CONFIG)
    paths = write_set(tmp_path / "n", 6, "normal", 0)
    bank = build_bank(paths, embedder, coreset_ratio=0.3, seed=0, root=tmp_path)

    result = execute_rebuild(plan, bank, DirectoryImageSource(tmp_path), embedder)

    assert result.executed is False
    assert result.bank is None
    assert "건드리지 않기로" in result.reason


# ── 뱅크 오염 — 무엇을 뺄 것인가 ───────────────────────────────────────


def test_contamination_plan_names_the_traced_image():
    """역추적이 지목한 이미지가 제거 후보에 들어가야 한다."""
    plan = plan_curation(
        diagnosis("bank_contamination", nearest_image="defect/d0.png", patch_verdict="defect"),
        missed_results=[missed("defect/d0.png"), missed("defect/d0.png")],
    )

    assert plan.touches_bank is True
    assert plan.remove
    names = [c.image for c in plan.remove]
    assert "defect/d0.png" in names

    candidate = next(c for c in plan.remove if c.image == "defect/d0.png")
    assert candidate.traced_hits >= 2, "반복 지목이 횟수로 세어져야 한다"
    assert candidate.confirmed_by_vlm is True


def test_repeated_hits_rank_higher():
    """여러 건이 가리킨 이미지가 위로 와야 한다. 한 장은 우연일 수 있다."""
    plan = plan_curation(
        diagnosis("bank_contamination", nearest_image="defect/often.png", patch_verdict="defect"),
        missed_results=[missed("defect/often.png")] * 3 + [missed("normal/rare.png")],
    )
    assert plan.remove[0].image == "defect/often.png"


def test_single_source_evidence_requires_human():
    """근거가 한 갈래뿐이면 사람 확인이 붙는다.

    정상 이미지를 잘못 빼면 커버리지 부족을 스스로 만든다.
    """
    plan = plan_curation(
        diagnosis("bank_contamination", nearest_image="normal/n0.png"),
        missed_results=[missed("normal/n0.png")],
    )
    assert plan.needs_human is True
    assert plan.remove[0].evidence_count == 1


def test_contamination_without_candidates_stops():
    """오염으로 진단됐어도 뺄 대상을 못 찾으면 실행하지 않는다."""
    plan = plan_curation(diagnosis("bank_contamination"), missed_results=[])

    assert plan.touches_bank is False
    assert plan.needs_human is True
    assert "특정하지 못했다" in plan.reason


# ── 커버리지 부족 — 무엇을 채울 것인가 ─────────────────────────────────


def test_coverage_plan_lists_missing_conditions():
    plan = plan_curation(
        diagnosis("coverage_gap"),
        missing_conditions={"date": "2026-06-20", "shift": "night"},
    )

    assert plan.touches_bank is True
    assert len(plan.add) == 2
    keys = {a.condition_key for a in plan.add}
    assert keys == {"date", "shift"}
    assert not plan.remove, "커버리지 부족에서는 빼지 않고 채운다"


def test_coverage_without_known_gap_stops():
    plan = plan_curation(diagnosis("coverage_gap"), missing_conditions=None)
    assert plan.touches_bank is False
    assert "어느 조건이 비었는지" in plan.reason


# ── 실제 재구성 ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder():
    try:
        return PatchEmbedder(CONFIG)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_rebuild_removes_and_records(embedder, tmp_path):
    """제거 계획이 실제 뱅크 구성에 반영되고 기록이 남아야 한다."""
    normal = write_set(tmp_path / "normal", 8, "normal", 0)
    defect = write_set(tmp_path / "defect", 2, "defect", 500)
    composition = list(normal) + list(defect)

    bank = build_bank(composition, embedder, coreset_ratio=0.3, seed=7,
                      bank_version="v3", root=tmp_path)
    contaminant = defect[0].relative_to(tmp_path).as_posix()

    plan = plan_curation(
        diagnosis("bank_contamination", nearest_image=contaminant, patch_verdict="defect"),
        missed_results=[missed(contaminant)] * 2,
    )
    result = execute_rebuild(
        plan, bank, DirectoryImageSource(tmp_path), embedder,
        triggered_by="2라인 뱅크 다시 만들어줘",
    )

    assert result.executed is True
    assert result.bank is not None
    assert contaminant not in result.bank.images
    assert len(result.bank.images) == len(bank.images) - 1

    record = result.record
    assert record is not None
    assert record.from_version == "v3" and record.to_version == "v4"
    assert contaminant in record.removed
    assert record.triggered_by == "2라인 뱅크 다시 만들어줘"
    assert record.approved_for_deploy is False, "재구성이 배포를 뜻하지 않는다"


def test_rebuild_keeps_settings_so_only_composition_changes(embedder, tmp_path):
    """시드와 설정이 유지되어야 성능 차이를 구성 변화로 해석할 수 있다."""
    paths = write_set(tmp_path / "n", 8, "normal", 0)
    bank = build_bank(paths, embedder, coreset_ratio=0.3, seed=7, root=tmp_path)

    from agents.curate import RemovalCandidate

    plan = CurationPlan(
        touches_bank=True,
        cause="bank_contamination",
        remove=[
            RemovalCandidate(
                image=paths[0].relative_to(tmp_path).as_posix(),
                reason="설정 유지 확인용",
                traced_hits=2,
            )
        ],
    )
    result = execute_rebuild(plan, bank, DirectoryImageSource(tmp_path), embedder)

    assert result.executed
    assert result.bank.meta["seed"] == bank.meta["seed"]
    assert result.bank.meta["coreset_ratio"] == bank.meta["coreset_ratio"]
    assert result.bank.meta["feature_config"] == bank.meta["feature_config"]


def test_compare_banks_shows_difference(embedder, tmp_path):
    paths = write_set(tmp_path / "n", 8, "normal", 0)
    before = build_bank(paths, embedder, coreset_ratio=0.3, seed=0, root=tmp_path)
    after = build_bank(paths[:-2], embedder, coreset_ratio=0.3, seed=0, root=tmp_path)

    diff = compare_banks(before, after)

    assert diff["images_before"] == 8
    assert diff["images_after"] == 6
    assert len(diff["removed"]) == 2
    assert diff["added"] == []


def test_rebuild_refuses_to_empty_the_bank(embedder, tmp_path):
    """전부 빼는 계획은 실행하지 않는다."""
    from agents.curate import RemovalCandidate

    paths = write_set(tmp_path / "n", 4, "normal", 0)
    bank = build_bank(paths, embedder, coreset_ratio=0.5, seed=0, root=tmp_path)

    plan = CurationPlan(
        touches_bank=True, cause="bank_contamination",
        remove=[RemovalCandidate(image=i, reason="전부") for i in bank.images],
    )
    result = execute_rebuild(plan, bank, DirectoryImageSource(tmp_path), embedder)

    assert result.executed is False
    assert "남는 이미지가 없다" in result.reason
