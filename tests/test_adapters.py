"""어댑터 계층과 시각 판별 검증.

모델 서버 없이 도는 테스트다. 실제 모델 응답을 흉내 내는 것이 목적이 아니라,
**응답이 없거나 깨졌을 때 판정을 지어내지 않는지**를 확인하는 것이 목적이다.

지어낸 판정이 근거로 올라가면 진단 정확도 수치 자체가 거짓이 된다.
"""

from __future__ import annotations

import json

import pytest

from agents.adapters import (
    ChatMessage,
    ImagePart,
    ModelConfig,
    StubAdapter,
    ToolCall,
    ToolSpec,
    build_adapter,
    load_config,
    parse_json_object,
)
from agents.vision import cause_from_patch_judgment, judge_bank_patch, judge_defect_visible
from tests.synthetic import make_defect, make_normal


# ── JSON 파싱: 모델은 형식을 자주 어긴다 ───────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"verdict": "defect"}', {"verdict": "defect"}),
        ('```json\n{"verdict": "normal"}\n```', {"verdict": "normal"}),
        ('설명입니다. {"verdict": "defect", "confidence": 0.8} 끝.',
         {"verdict": "defect", "confidence": 0.8}),
        ("전혀 JSON 이 아님", {}),
        ("", {}),
        ("[1, 2, 3]", {}),  # 객체가 아니면 받지 않는다
    ],
)
def test_json_parsing_survives_common_deviations(text, expected):
    assert parse_json_object(text) == expected


# ── 스텁: 기본값은 판단하지 않음 ───────────────────────────────────────


def test_stub_defaults_to_no_judgment():
    """설정 없이 만든 스텁은 판정을 내리지 않아야 한다."""
    adapter = StubAdapter()
    response = adapter.chat([ChatMessage.user("아무거나")])

    assert response.is_stub is True
    assert response.json()["verdict"] == "unknown"
    assert response.json()["confidence"] == 0.0


def test_stub_marks_itself_in_description():
    """리포트에 실렸을 때 사람이 알아볼 수 있어야 한다."""
    assert "스텁" in StubAdapter().describe()


def test_stub_returns_scripted_answers_in_order():
    adapter = StubAdapter(scripted=[{"verdict": "defect"}, {"verdict": "normal"}])

    assert adapter.chat([ChatMessage.user("1")]).json()["verdict"] == "defect"
    assert adapter.chat([ChatMessage.user("2")]).json()["verdict"] == "normal"
    # 대본이 떨어지면 마지막 응답을 반복한다
    assert adapter.chat([ChatMessage.user("3")]).json()["verdict"] == "normal"


def test_stub_can_script_tool_calls():
    """오케스트레이션 검증용. 도구 호출이 그대로 전달되는지."""
    adapter = StubAdapter(
        tool_calls=[[ToolCall(id="c1", name="lookup_threshold", arguments={"line": "line_02"})]]
    )
    response = adapter.chat([ChatMessage.user("임계값 알려줘")])

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "lookup_threshold"
    assert response.tool_calls[0].arguments == {"line": "line_02"}


def test_stub_records_what_it_received():
    """프롬프트가 의도대로 조립됐는지 확인할 수 있어야 한다."""
    adapter = StubAdapter()
    judge_defect_visible(adapter, make_defect(1), reported_defect="스크래치")

    assert len(adapter.calls) == 1
    sent = adapter.calls[0][0]
    assert "스크래치" in sent.content
    assert len(sent.images) == 1


# ── 설정: 환경 변수로 교체된다 ─────────────────────────────────────────


def test_default_config_is_stub(monkeypatch):
    """아무 설정이 없으면 스텁으로 떨어져 흐름은 돌아간다."""
    for key in list(os_environ_keys()):
        monkeypatch.delenv(key, raising=False)

    config = load_config()
    assert config.llm.provider == "stub"
    assert config.vlm.provider == "stub"
    assert build_adapter(config.llm).is_stub is True


def os_environ_keys():
    import os

    return [k for k in os.environ if k.startswith("SHVO_")]


def test_env_switches_provider_and_fills_local_base_url(monkeypatch):
    """설정으로 교체된다 — 폐쇄망 대응의 전제."""
    monkeypatch.setenv("SHVO_VLM_PROVIDER", "openai_compat")
    monkeypatch.setenv("SHVO_VLM_MODEL", "some-vision-model")

    config = load_config()
    assert config.vlm.provider == "openai_compat"
    assert config.vlm.model == "some-vision-model"
    # 주소를 안 주면 로컬 기본값이 들어가야 한다
    assert config.vlm.base_url.startswith("http://localhost")
    # 언어 모델 쪽은 건드리지 않았으므로 그대로
    assert config.llm.provider == "stub"


def test_bad_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("SHVO_LLM_PROVIDER", "gpt-something")
    with pytest.raises(ValueError, match="stub 또는 openai_compat"):
        load_config()


def test_openai_compat_requires_model_name():
    with pytest.raises(ValueError, match="모델 이름이 필요하다"):
        build_adapter(ModelConfig(provider="openai_compat", model=""))


# ── 시각 판별: 지어내지 않는다 ─────────────────────────────────────────


def test_stub_judgment_is_not_usable_as_evidence():
    """스텁 응답은 근거로 쓸 수 없다고 표시되어야 한다."""
    judgment = judge_defect_visible(StubAdapter(), make_defect(1))

    assert judgment.verdict == "unknown"
    assert judgment.is_stub is True
    assert judgment.usable is False
    assert cause_from_patch_judgment(judgment) is None


def test_scripted_stub_is_still_not_usable():
    """대본으로 그럴듯한 답을 넣어도 스텁이면 근거가 될 수 없다.

    테스트 편의로 넣은 값이 진단 근거로 새어 들어가는 것을 막는다.
    """
    adapter = StubAdapter(scripted=[{"verdict": "defect", "confidence": 0.99, "reason": "찍힘"}])
    judgment = judge_bank_patch(adapter, make_normal(1))

    assert judgment.verdict == "defect"
    assert judgment.usable is False  # is_stub 이 True 이므로


def test_broken_response_becomes_unknown():
    """모델이 형식을 어기면 unknown 이어야 한다. 추측하면 안 된다."""
    adapter = StubAdapter(scripted=["모르겠는데요 아마 정상 같습니다"])
    judgment = judge_defect_visible(adapter, make_normal(1))

    assert judgment.verdict == "unknown"
    assert judgment.confidence == 0.0


def test_out_of_range_verdict_becomes_unknown():
    """허용하지 않은 판정값은 받지 않는다."""
    adapter = StubAdapter(scripted=[{"verdict": "maybe_defect", "confidence": 0.9}])
    assert judge_bank_patch(adapter, make_normal(1)).verdict == "unknown"


def test_call_failure_becomes_unknown_not_exception():
    """모델 호출이 실패해도 진단이 멈추면 안 된다."""

    class Broken(StubAdapter):
        def chat(self, *args, **kwargs):
            raise ConnectionError("서버 없음")

    judgment = judge_defect_visible(Broken(), make_normal(1))
    assert judgment.verdict == "unknown"
    assert "실패" in judgment.reason


def test_confidence_is_clamped():
    adapter = StubAdapter(scripted=[{"verdict": "defect", "confidence": 3.7}])
    assert judge_bank_patch(adapter, make_normal(1)).confidence == 1.0


# ── 판별 5번이 원인 두 갈래로 이어지는가 ───────────────────────────────


def test_patch_judgment_maps_to_opposite_causes():
    """결함이면 뱅크 오염, 진짜 정상품이면 정상 분포 중첩.

    조치가 정반대인 두 원인이 이 한 판정에서 갈린다.
    """
    from agents.vision import VisionJudgment

    contaminated = VisionJudgment("defect", 0.9, "찍힘 보임", "real-model", is_stub=False)
    genuine = VisionJudgment("normal", 0.9, "정상 표면", "real-model", is_stub=False)

    assert cause_from_patch_judgment(contaminated) == "bank_contamination"
    assert cause_from_patch_judgment(genuine) == "normal_overlap"


# ── 이미지 전달 ────────────────────────────────────────────────────────


def test_image_becomes_inline_data_url():
    """이미지는 data URL 로 실려 나간다. 외부 주소를 만들지 않는다."""
    url = ImagePart(make_normal(0)).to_data_url()

    assert url.startswith("data:image/png;base64,")
    assert len(url) > 100


def test_missing_image_path_raises():
    with pytest.raises(FileNotFoundError):
        ImagePart("없는파일.png").to_data_url()


def test_tool_spec_converts_to_openai_shape():
    spec = ToolSpec(
        name="lookup_threshold",
        description="라인의 현재 임계값을 조회한다",
        parameters={"type": "object", "properties": {"line": {"type": "string"}}},
    )
    payload = spec.to_openai()

    assert payload["type"] == "function"
    assert payload["function"]["name"] == "lookup_threshold"
    assert json.dumps(payload)  # 직렬화가 되어야 전송된다
