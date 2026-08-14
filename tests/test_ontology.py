"""진단 온톨로지 — 모델이 읽되 정하지는 못하는가.

여기서 지키려는 것은 둘이다.

  1. **표가 두 벌이 되지 않는가.** 재구성 필요 여부·권고 조치·금지 조치는
     diagnose.py 가 정답이다. 온톨로지가 같은 값을 따로 적으면 한쪽만 고쳐지고,
     그때 모델은 틀린 쪽을 읽는다.
  2. **온톨로지가 판정하지 않는가.** 원인은 판별 7항목으로 decide() 가 낸다.
     조회 결과가 그럴듯하다고 원인이 정해지면 진단이 유사도 맞히기가 된다.
"""

from __future__ import annotations

import pytest

from agents import ontology
from agents.diagnose import (
    CAUSE_LABEL_KO,
    FORBIDDEN_ACTIONS,
    REBUILD_REQUIRED,
    RECOMMENDED_ACTIONS,
)


# ── 표가 두 벌이 되지 않는가 ────────────────────────────────────────────


def test_every_cause_in_diagnose_has_a_definition():
    """원인이 늘거나 줄면 여기서 걸린다. 정의 없는 원인이 모델에 노출되면 안 된다."""
    assert set(ontology.CAUSES) == set(REBUILD_REQUIRED)
    assert set(ontology.CAUSES) == set(CAUSE_LABEL_KO)


@pytest.mark.parametrize("cause", sorted(REBUILD_REQUIRED))
def test_the_tables_are_not_copied_but_borrowed(cause):
    """조회 결과가 diagnose 의 표와 글자 그대로 같아야 한다."""
    found = ontology.describe_cause(cause)

    assert found["requires_bank_rebuild"] == REBUILD_REQUIRED[cause]
    assert found["recommended_actions"] == RECOMMENDED_ACTIONS[cause]
    assert found["forbidden_actions"] == FORBIDDEN_ACTIONS[cause]
    assert found["label"] == CAUSE_LABEL_KO[cause]


def test_the_summary_counts_rebuild_causes_from_the_table():
    """"여섯 중 넷은 재구성이 답이 아니다"를 손으로 적지 않는다."""
    summary = ontology.overview()
    expected = [c for c, needed in REBUILD_REQUIRED.items() if needed]

    assert summary["rebuild_is_the_answer_for"] == expected
    assert str(len(expected)) in summary["note"]


def test_every_action_the_rules_can_emit_has_a_korean_label():
    """화면과 승인 문서가 조치 id 를 그대로 보여주지 않게."""
    emitted = {a for actions in RECOMMENDED_ACTIONS.values() for a in actions}
    emitted |= {a for actions in FORBIDDEN_ACTIONS.values() for a in actions}
    # 진단이 판정 없이 돌려주는 조치도 포함한다.
    emitted |= {"review_past_issue", "request_correct_image"}

    missing = sorted(a for a in emitted if a not in ontology.ACTION_LABEL_KO)
    assert not missing, f"사람 말 이름이 없는 조치: {missing}"


# ── 판별 7항목 ──────────────────────────────────────────────────────────


def test_seven_checks_numbered_one_through_seven():
    assert [c.item_no for c in ontology.CHECKS] == [1, 2, 3, 4, 5, 6, 7]


def test_only_two_checks_come_from_a_vision_model():
    """일곱 중 둘만 시각 언어 모델이다. 이 비중이 뒤집히면 진단이 인상 평가가 된다."""
    vlm = [c.item_no for c in ontology.CHECKS if c.source == "vlm"]
    assert vlm == [1, 5]


def test_the_two_causes_that_split_on_check_five_say_so():
    """뱅크 오염과 정상 분포 중첩은 판별 5번으로 갈린다. 조치가 정반대다."""
    contamination = ontology.describe_cause("bank_contamination")
    overlap = ontology.describe_cause("normal_overlap")

    assert 5 in contamination["decided_by"]
    assert 5 in overlap["decided_by"]
    assert "normal_overlap" in contamination["confused_with"]
    assert "bank_contamination" in overlap["confused_with"]
    assert contamination["requires_bank_rebuild"] is True
    assert overlap["requires_bank_rebuild"] is False


# ── 판정하지 않는가 ─────────────────────────────────────────────────────


def test_the_module_has_no_way_to_decide_a_cause():
    """온톨로지에 판정 함수가 생기면 여기서 걸린다."""
    for forbidden in ("decide", "diagnose", "classify", "infer_cause"):
        assert not hasattr(ontology, forbidden), (
            f"온톨로지에 {forbidden} 가 생겼습니다. 원인 판정은 diagnose.decide() 하나입니다."
        )


@pytest.mark.parametrize("kwargs", [
    {},
    {"cause": "bank_contamination"},
    {"check_item": 5},
    {"cause": "없는원인"},
    {"check_item": 99},
])
def test_every_answer_carries_the_disclaimer(kwargs):
    """조회 결과만 보고 원인을 정하지 말라는 문구가 매번 함께 나간다."""
    assert "disclaimer" in ontology.lookup_ontology(**kwargs)


def test_unknown_names_answer_instead_of_raising():
    """모델이 고쳐 부를 수 있어야 한다. 조회 실패로 진행이 멈추면 안 된다."""
    missing_cause = ontology.lookup_ontology(cause="뱅크가_이상함")
    assert "error" in missing_cause
    assert missing_cause["available_causes"] == list(REBUILD_REQUIRED)

    missing_check = ontology.lookup_ontology(check_item=0)
    assert "error" in missing_check
    assert missing_check["available_checks"] == [1, 2, 3, 4, 5, 6, 7]


def test_the_summary_is_returned_when_nothing_is_asked():
    summary = ontology.lookup_ontology()
    assert len(summary["causes"]) == len(REBUILD_REQUIRED)
    assert len(summary["checks"]) == 7
