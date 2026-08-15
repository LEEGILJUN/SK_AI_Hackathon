"""평가 게이트·섀도 비교·릴리즈 검증 — 작업 17·18·20.

세 가지를 본다.

  1. 게이트가 나쁜 후보를 실제로 막는가
  2. 섀도가 **새로 놓치는 건**을 잡아내는가 — 개선인 줄 알고 배포하는 것을 막는다
  3. 릴리즈가 배포하지 않고 승인 요청까지만 만드는가

3번은 이 과제가 스스로 그은 경계다. 배포하는 코드가 생기면 그 경계가
무너지므로 테스트로 고정한다.
"""

from __future__ import annotations

import pytest

from agents.curate import CurationPlan, RemovalCandidate
from agents.diagnose import DiagnosisResult, Evidence
from agents.gate import GateCriteria, check_reproducibility, evaluate_gate
from agents.rebuild import RebuildRecord
from agents.release import prepare_release
from inspection import FeatureConfig, PatchEmbedder, build_bank
from inspection.shadow import Disagreement, ShadowReport, shadow_compare
from inspection.sweep import sweep_thresholds
from tests.synthetic import write_set

CONFIG = FeatureConfig(backbone="resnet18", crop=64)


# ── 평가 게이트 ────────────────────────────────────────────────────────


def test_gate_passes_a_good_candidate():
    result = evaluate_gate(
        normal_scores=[0.1, 0.2, 0.15, 0.18, 0.12],
        defect_scores=[0.9, 0.85, 0.95],
        threshold=0.5,
    )
    assert result.passed is True
    assert result.failures == []
    assert "모든 항목을 통과" in result.reason


def test_gate_blocks_low_detection():
    result = evaluate_gate(
        normal_scores=[0.1, 0.2, 0.15],
        defect_scores=[0.9, 0.3, 0.2],   # 셋 중 하나만 임계값 위
        threshold=0.5,
    )
    assert result.passed is False
    assert any(c.name == "detection_rate" for c in result.failures)


def test_gate_blocks_high_false_positive():
    result = evaluate_gate(
        normal_scores=[0.9, 0.8, 0.7],   # 양품이 전부 임계값 위
        defect_scores=[0.95, 0.92, 0.99],
        threshold=0.5,
    )
    assert result.passed is False
    assert any(c.name == "false_positive_rate" for c in result.failures)


def test_gate_blocks_regression_from_shadow():
    """섀도에서 새로 놓치는 건이 있으면 막는다.

    고치려던 문제가 나아져도 다른 것을 잃으면 개선이 아니다.
    """
    shadow = ShadowReport(
        total=100, agreed=97,
        disagreements=[
            Disagreement("a.png", "newly_missed", 0.9, 0.3, "defect", "pass"),
            Disagreement("b.png", "newly_detected", 0.3, 0.9, "pass", "defect"),
            Disagreement("c.png", "newly_detected", 0.2, 0.8, "pass", "defect"),
        ],
    )
    result = evaluate_gate(
        normal_scores=[0.1, 0.2, 0.15, 0.18],
        defect_scores=[0.9, 0.85, 0.95],
        threshold=0.5,
        shadow=shadow,
    )
    assert result.passed is False
    failed = next(c for c in result.failures if c.name == "newly_missed")
    assert failed.value == 1
    assert "개선이 아니다" in failed.detail


def test_gate_requires_improvement_over_baseline():
    """이전보다 나빠졌으면 막는다."""
    baseline = sweep_thresholds([0.1, 0.2], [0.9, 0.95])       # AUROC 1.0
    result = evaluate_gate(
        normal_scores=[0.1, 0.6, 0.2, 0.15],
        defect_scores=[0.9, 0.5, 0.95],                        # 겹침
        threshold=0.45,
        baseline_curve=baseline,
        criteria=GateCriteria(min_detection_rate=0.5, min_auroc=0.0, max_false_positive_rate=1.0),
    )
    assert result.passed is False
    assert any(c.name == "improvement" for c in result.failures)


def test_criteria_are_configurable():
    """통과 기준은 도메인이 정한다. 코드에 박혀 있으면 안 된다."""
    scores = dict(normal_scores=[0.1, 0.2, 0.15], defect_scores=[0.9, 0.3, 0.2], threshold=0.5)

    strict = evaluate_gate(**scores)
    lenient = evaluate_gate(**scores, criteria=GateCriteria(min_detection_rate=0.3, min_auroc=0.0))

    assert strict.passed is False
    assert lenient.passed is True


# ── 재현성 ─────────────────────────────────────────────────────────────


def test_reproducibility_detects_identical_results():
    result = check_reproducibility(lambda: {"cause": "bank_contamination"}, runs=10)
    assert result.identical is True
    assert result.runs == 10


def test_reproducibility_detects_drift():
    """흔들리면 잡아내야 한다. 게이트를 통과할 때까지 재시도하는 일을 막는다."""
    counter = {"n": 0}

    def flaky():
        counter["n"] += 1
        return {"cause": "bank_contamination" if counter["n"] % 2 else "normal_overlap"}

    result = check_reproducibility(flaky, runs=10)
    assert result.identical is False
    assert len(result.distinct_outcomes) == 2
    assert "시드로 묶어야" in result.detail


def test_reproducibility_needs_at_least_two_runs():
    with pytest.raises(ValueError, match="2회 이상"):
        check_reproducibility(lambda: 1, runs=1)


# ── 섀도 비교 ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def embedder():
    try:
        return PatchEmbedder(CONFIG)
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_shadow_finds_only_disagreements(embedder, tmp_path):
    """판정이 갈린 건만 뽑아야 한다. 사람은 그것만 보면 된다."""
    normal = write_set(tmp_path / "n", 10, "normal", 0)
    defect = write_set(tmp_path / "d", 4, "defect", 500)

    clean = build_bank(normal, embedder, coreset_ratio=0.3, seed=0,
                       bank_version="v3", root=tmp_path)
    dirty = build_bank(list(normal) + list(defect[:2]), embedder, coreset_ratio=0.3,
                       seed=0, bank_version="v4", root=tmp_path)

    queries = list(normal[:3]) + list(defect[2:])
    report = shadow_compare(
        queries, clean, dirty, current_threshold=2.0, embedder=embedder, root=tmp_path
    )

    assert report.total == len(queries)
    assert report.agreed + report.review_count == report.total
    assert report.current_version == "v3" and report.candidate_version == "v4"
    for d in report.disagreements:
        assert d.current_verdict != d.candidate_verdict
    assert "사람은 이" in report.summary()


def test_shadow_separates_newly_missed_from_newly_detected():
    report = ShadowReport(
        total=10, agreed=8,
        disagreements=[
            Disagreement("a.png", "newly_missed", 0.9, 0.3, "defect", "pass"),
            Disagreement("b.png", "newly_detected", 0.3, 0.9, "pass", "defect"),
        ],
    )
    assert len(report.newly_missed) == 1
    assert len(report.newly_detected) == 1
    assert report.review_count == 2
    assert report.agreement_rate == 0.8


def test_shadow_rejects_empty_input(embedder, tmp_path):
    paths = write_set(tmp_path / "n", 4, "normal", 0)
    bank = build_bank(paths, embedder, coreset_ratio=0.5, seed=0, root=tmp_path)
    with pytest.raises(ValueError, match="이미지가 없다"):
        shadow_compare([], bank, bank, 1.0)


# ── 릴리즈 ─────────────────────────────────────────────────────────────


def _fixtures(embedder, tmp_path):
    paths = write_set(tmp_path / "n", 6, "normal", 0)
    bank = build_bank(paths, embedder, coreset_ratio=0.3, seed=0,
                      bank_version="v4", root=tmp_path)
    diagnosis = DiagnosisResult(
        cause="bank_contamination", requires_bank_rebuild=True,
        confidence="high", needs_human=False,
        reasoning="최근접 패치가 결함이었다.",
        evidence=[
            Evidence(4, "nearest_patch", {"source_image": "d/defect_000.png"}, "trace", True, "역추적"),
            Evidence(5, "nearest_patch_is_defect", "defect", "vlm", True, "결함으로 판독"),
            Evidence(2, "quality_within_baseline", None, "compute", False, "재지 않음"),
        ],
    )
    plan = CurationPlan(
        touches_bank=True, cause="bank_contamination",
        remove=[RemovalCandidate("d/defect_000.png", "역추적이 지목", traced_hits=3)],
        reason="오염 1장 제거",
    )
    record = RebuildRecord(
        from_version="v3", to_version="v4", cause="bank_contamination",
        removed=["d/defect_000.png"], kept_count=6,
        reason="1장 제거 후 재구성", triggered_by="2라인 뱅크 다시 만들어줘",
    )
    return bank, diagnosis, plan, record


def test_release_never_marks_itself_approved(embedder, tmp_path):
    """배포 승인은 사람이 한다. 이 경계가 무너지면 안 된다."""
    bank, diagnosis, plan, record = _fixtures(embedder, tmp_path)
    gate = evaluate_gate([0.1, 0.2, 0.15], [0.9, 0.85, 0.95], threshold=0.5)

    package = prepare_release(
        tmp_path / "release", bank=bank, record=record,
        diagnosis=diagnosis, plan=plan, gate=gate,
    )

    assert package.approved is False
    assert record.approved_for_deploy is False
    assert package.ready_for_review is True


def test_release_has_no_deploy_function():
    """배포 함수가 생기면 이 테스트가 깨진다. 의도적으로 만들지 않는다."""
    import agents.release as module

    suspicious = [
        name for name in dir(module)
        if any(word in name.lower() for word in ("deploy", "apply", "install", "push_to"))
    ]
    assert suspicious == [], f"배포로 보이는 함수가 생겼다: {suspicious}"


def test_release_package_contains_everything_for_a_decision(embedder, tmp_path):
    bank, diagnosis, plan, record = _fixtures(embedder, tmp_path)
    gate = evaluate_gate([0.1, 0.2, 0.15], [0.9, 0.85, 0.95], threshold=0.5)
    shadow = ShadowReport(
        total=50, agreed=48,
        disagreements=[
            Disagreement("x.png", "newly_detected", 0.3, 0.9, "pass", "defect"),
            Disagreement("y.png", "newly_missed", 0.9, 0.3, "defect", "pass"),
        ],
    )
    reproducibility = check_reproducibility(lambda: "same", runs=10)

    package = prepare_release(
        tmp_path / "release", bank=bank, record=record, diagnosis=diagnosis,
        plan=plan, gate=gate, shadow=shadow, reproducibility=reproducibility,
        issue_text="2라인 캡슐 찍힘이 계속 빠집니다",
    )

    for name in ("bank_meta.json", "rebuild_record.json", "diagnosis.json",
                 "curation_plan.json", "gate.json", "shadow.json", "승인요청.md"):
        assert (package.directory / name).exists(), f"{name} 이 없다"

    text = package.approval_document.read_text(encoding="utf-8")
    assert "배포는 실행되지 않았습니다" in text
    assert "2라인 캡슐 찍힘이 계속 빠집니다" in text
    assert "뱅크 오염" in text
    assert "새로 놓치는 건 1건" in text, "불리한 내용도 실려야 한다"
    assert "확인하지 못한 판별 항목" in text, "근거를 못 얻은 항목도 밝혀야 한다"
    assert "승인자" in text


def test_failed_gate_is_surfaced_not_hidden(embedder, tmp_path):
    """게이트에서 떨어졌으면 문서에 그대로 드러나야 한다."""
    bank, diagnosis, plan, record = _fixtures(embedder, tmp_path)
    gate = evaluate_gate([0.1, 0.2, 0.15], [0.9, 0.3, 0.2], threshold=0.5)
    assert gate.passed is False

    package = prepare_release(
        tmp_path / "release", bank=bank, record=record,
        diagnosis=diagnosis, plan=plan, gate=gate,
    )
    text = package.approval_document.read_text(encoding="utf-8")

    assert "**미통과**" in text
    assert "게이트를 통과하지 못했습니다" in text
    assert package.blocking_reasons, "승인을 막는 사유가 남아야 한다"


# ── 통과 기준은 파일에서 온다 ──────────────────────────────────────────


def test_the_criteria_come_from_the_file_not_the_code(tmp_path):
    """**기준값을 코드에 박지 않는다.**

    상황에 따라 바뀌는 값이라 코드에 있으면 바꿀 때마다 커밋이 필요하고,
    나중에 왜 그 값인지 아무도 답하지 못한다.
    """
    path = tmp_path / "gate.yaml"
    path.write_text(
        "defaults:\n"
        "  min_detection_rate: 0.7\n"
        "  max_newly_missed: 2\n",
        encoding="utf-8",
    )

    loaded = GateCriteria.load(path=path)
    assert loaded.min_detection_rate == 0.7
    assert loaded.max_newly_missed == 2
    assert loaded.min_auroc == GateCriteria().min_auroc, "안 적은 항목은 기본값이다"


def test_a_line_can_tighten_one_value_without_repeating_the_rest(tmp_path):
    """라인마다 과검 한 건의 무게가 다르다. 적은 항목만 덮어쓴다."""
    path = tmp_path / "gate.yaml"
    path.write_text(
        "defaults:\n"
        "  max_false_positive_rate: 0.05\n"
        "  min_auroc: 0.85\n"
        "lines:\n"
        "  line_01:\n"
        "    max_false_positive_rate: 0.03\n",
        encoding="utf-8",
    )

    tight = GateCriteria.load("line_01", path=path)
    loose = GateCriteria.load("line_02", path=path)

    assert tight.max_false_positive_rate == 0.03
    assert loose.max_false_positive_rate == 0.05
    assert tight.min_auroc == 0.85, "덮어쓰지 않은 항목은 기본값을 그대로 쓴다"


def test_a_broken_file_falls_back_instead_of_stopping_the_gate(tmp_path):
    """**파일이 깨져도 게이트는 선다.**

    게이트가 못 서는 것보다 기본값으로라도 도는 편이 낫다. 시연 중에는
    오타 하나를 고칠 시간이 없다.
    """
    broken = tmp_path / "gate.yaml"
    broken.write_text("defaults: [이건 표가 아니다\n", encoding="utf-8")

    assert GateCriteria.load(path=broken) == GateCriteria()
    assert GateCriteria.load(path=tmp_path / "없는파일.yaml") == GateCriteria()


def test_an_unknown_key_is_dropped_rather_than_crashing(tmp_path):
    """오타 하나로 게이트가 안 서면 시연 중에 고칠 수 없다."""
    path = tmp_path / "gate.yaml"
    path.write_text(
        "defaults:\n"
        "  min_detection_rate: 0.8\n"
        "  min_detecion_rate: 0.1\n",   # 오타
        encoding="utf-8",
    )

    assert GateCriteria.load(path=path).min_detection_rate == 0.8


def test_the_shipped_file_parses_and_carries_reasons():
    """저장소에 든 파일이 실제로 읽히는지. 근거 없는 숫자는 못 지킨다."""
    from agents.gate import CRITERIA_PATH

    assert CRITERIA_PATH.exists(), f"{CRITERIA_PATH} 가 없다"
    loaded = GateCriteria.load()
    assert loaded.max_newly_missed == 0, "새로 놓치는 것은 한 건도 감수하지 않는다"

    text = CRITERIA_PATH.read_text(encoding="utf-8")
    for name in ("min_detection_rate", "max_false_positive_rate",
                 "min_auroc", "max_newly_missed"):
        assert name in text
    assert "근거" in text or "이유" in text or "때문" in text, "값만 있고 근거가 없다"
