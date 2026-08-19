

def test_보충할_이미지를_못_찾으면_성공이라_하지_않는다():
    """**"보충하겠다"고 말하고 0장을 넣은 뒤 성공으로 끝나던 자리다.**

    `app/pipeline.py` 가 `DirectoryImageSource` 에 조건을 안 넘겨
    `images_for` 가 언제나 빈 목록이었다. 그런데 `composition = kept` 라
    재구성이 성공으로 끝났다. **뱅크가 안 바뀌는데 화면은 성공으로 보인다.**

    뱅크 오염 시연에서는 제거만 하므로 안 드러났다. 커버리지 부족을
    시연하려면 여기가 이어져 있어야 한다.
    """
    from agents.curate import AdditionRequest, CurationPlan
    from agents.rebuild import DirectoryImageSource, execute_rebuild

    class _Bank:
        version = "pcb1-01-v1"
        images = ["a.png", "b.png"]
        meta = {"coreset_ratio": 0.5, "seed": 0, "projection_dim": None,
                "max_bank_size": None}

    plan = CurationPlan(
        touches_bank=True, cause="coverage_gap",
        add=[AdditionRequest(condition_key="lot", condition_value="LOT-X",
                             reason="그 조건의 정상 패치가 없다")],
        reason="커버리지 부족",
    )
    # 조건을 안 준 원본. 예전 동작이다.
    result = execute_rebuild(plan, _Bank(), DirectoryImageSource("."), embedder=None)
    assert result.executed is False, "0장 보충을 성공이라 하면 안 된다"
    assert "LOT-X" in result.reason, "어느 조건을 못 찾았는지 알려야 한다"


def test_조건을_주면_보충이_실제로_일어난다(tmp_path):
    """조건 표가 이어져 있으면 그 조건의 이미지가 실제로 들어간다."""
    from agents.curate import AdditionRequest, CurationPlan
    from agents.rebuild import DirectoryImageSource

    source = DirectoryImageSource(tmp_path, {"lot": {"LOT-X": ["n1.png", "n2.png"]}})
    assert source.images_for("lot", "LOT-X") == ["n1.png", "n2.png"]
    assert source.images_for("lot", "LOT-없음") == []
