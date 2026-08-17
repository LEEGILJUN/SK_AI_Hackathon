"""인테이크 — 자연어에서 무엇을 뽑고, 못 뽑으면 어떻게 되는가.

**제품명(`product_id`)이 MES 조회의 열쇠다.** 이슈는 보통 이미지가 아니라
"이 제품이 계속 빠진다"로 오고, 이것이 비면 접수가 되묻고 거기서 멈춘다.

접수 판정과 중복 차단은 `tests/test_app_pipeline.py` 와
`tests/test_issue_history.py` 가 본다. 여기는 추출만 본다.
"""

# ── 제품명 추출 ─────────────────────────────────────────────────────────


def test_the_model_s_product_id_is_actually_read():
    """모델이 뽑은 제품명을 버리지 않는다.

    **`extract()` 가 `product_id` 를 JSON 에서 읽지도 않고 있었다.** 모델이
    제대로 뽑아 줘도 버려졌고, 4090 실측에서 "gemma 가 제품명을 못 뽑는다"로
    보고됐다. 모델 문제가 아니라 코드가 안 읽은 것이었다.

    제품명은 MES 조회의 열쇠라 이게 비면 접수가 되묻고 거기서 멈춘다.
    """
    from agents.adapters.base import ChatResponse, ModelAdapter
    from agents.intake import extract

    class Extractor(ModelAdapter):
        is_stub = False

        def describe(self):
            return "test"

        def chat(self, messages, tools=None, **kwargs):
            return ChatResponse(
                text='{"line": "line_01", "object_name": "pcb1", '
                     '"defect_type": "scratch", "product_id": "PCB1-LOT-AAJ-img_0087", '
                     '"lot": null, "area_hint": null, "observed_from": null}',
                is_stub=False, model="test",
            )

    report = extract("아무 텍스트", Extractor())
    assert report.product_id == "PCB1-LOT-AAJ-img_0087"


def test_the_product_id_is_picked_up_from_the_text_when_the_model_misses_it():
    """모델이 놓치면 원문에서 줍는다. 다만 지어내지는 않는다."""
    from agents.intake import find_product_id

    assert find_product_id(
        "1라인 PCB 기판에 스크래치. 제품 PCB1-LOT-AAJ-img_0087 건입니다."
    ) == "PCB1-LOT-AAJ-img_0087"

    # 숫자가 없으면 제품 코드로 보지 않는다. "PCB-기판" 같은 말이 걸리면
    # 엉뚱한 것을 MES 에 물어보게 된다.
    assert find_product_id("PCB 기판에서 미검이 납니다.") is None
    assert find_product_id("라인에서 문제가 있습니다.") is None
    assert find_product_id("") is None


def test_a_lot_is_not_mistaken_for_a_product():
    """`A-217 로트가 …` 의 A-217 을 제품명으로 넣지 않는다.

    로트는 로트 칸에 들어가야 조인이 맞는다. 제품명 자리에 넣으면
    `find_images(product_id=...)` 가 **조용히 빈손이 된다.**

    한글 조사 때문에 낱말 경계(`\\b`)가 성립하지 않아 이 규칙이 한 번
    통째로 안 먹었다.
    """
    from agents.intake import find_product_id

    assert find_product_id("A-217 로트가 계속 빠집니다.") is None
    assert find_product_id("A-217 랏에서 미검") is None
    # 로트와 제품이 함께 있으면 제품 쪽을 고른다
    assert find_product_id(
        "A-217 로트의 PCB1-LOT-AAJ-img_0087 이 빠집니다."
    ) == "PCB1-LOT-AAJ-img_0087"


def test_모델이_자연어를_담아와도_식별자로_맞춘다():
    """**모델은 명세를 안 지킨다.** 도구 명세에 "라인 ID(예: line_01). 1라인
    같은 말이 아니다" 라고 적어 두었는데도 `1라인`·`PCB 기판` 을 담아 왔다.

    그대로 두면 뱅크 조회가 정확한 문자열 일치라 `None` 이 되고 2단계에서
    멈춘다. 실측에서 그랬다. 지금까지 완주한 것은 모델이 도구 인자에는
    정규 ID 를 넘겼기 때문이지 이 값이 맞아서가 아니었다.
    """
    from agents.intake import normalize, IssueReport
    from lookup.base import ImageRecord

    class _Lookup:
        def find_images(self, line=None, object_name=None, lot=None,
                        product_id=None, limit=50):
            table = {"line_01": "pcb1", "line_02": "pcb2"}
            if line not in table:
                return []
            return [ImageRecord(product_id="P-1", line=line,
                                object_name=table[line], path="x.png")]

    lookup = _Lookup()
    for raw_line, raw_object, line, obj in [
        ("1라인", "PCB 기판", "line_01", "pcb1"),
        ("1번 라인", "기판", "line_01", "pcb1"),
        ("line 1", None, "line_01", "pcb1"),
        ("2라인", "PCB", "line_02", "pcb2"),
        ("line_01", "pcb1", "line_01", "pcb1"),      # 이미 맞으면 그대로
    ]:
        report = IssueReport(raw_text="")
        report.line, report.object_name = raw_line, raw_object
        normalize(report, lookup)
        assert report.line == line, f"{raw_line} → {report.line}"
        assert report.object_name == obj, f"{raw_object} → {report.object_name}"


def test_사람이_말한_품목을_라인으로_덮어쓰지_않는다():
    """**"2라인 캡슐" 을 pcb2 로 바꿔 놓고 있었다.**

    라인으로 품목을 되찾는 것은 "PCB 기판" 처럼 같은 것을 다르게 부른
    경우를 위한 것인데, 2라인의 품목이 pcb2 라는 이유로 캡슐을 pcb2 로
    바꿔 첫 카드에 찍었다. **사용자가 말한 적 없는 값이 화면에 나타나면
    "모델이 뽑은 것"으로 오해한다.**
    """
    from agents.intake import normalize, IssueReport
    from lookup.base import ImageRecord

    class _Lookup:
        def find_images(self, line=None, object_name=None, lot=None,
                        product_id=None, limit=50):
            table = {"line_01": "pcb1", "line_02": "pcb2"}
            if line not in table:
                return []
            return [ImageRecord(product_id="P-1", line=line,
                                object_name=table[line], path="x.png")]

    lookup = _Lookup()
    for raw_line, spoken, expected in [
        ("1라인", "PCB 기판", "pcb1"),      # 같은 것을 다르게 부른 것 → 되찾는다
        ("2라인", "기판", "pcb2"),
        ("2라인", "캡슐", None),            # 다른 품목 → 덮어쓰지 않는다
        ("9라인", "PCB 기판", None),        # 그 라인에 아무것도 없다
    ]:
        report = IssueReport(raw_text="")
        report.line, report.object_name = raw_line, spoken
        normalize(report, lookup)
        assert report.object_name == expected, f"{raw_line}/{spoken} → {report.object_name}"


def test_결함_유형이_품목마다_다른_어휘로_맞는다():
    """**품목마다 쓰는 말이 다르다.** pcb 는 scratch, fryum 은 small scratches.

    실측에서 모델이 `미세 스크래치` 를 담아 왔는데 이력의 값은 `scratch` 라
    중복 차단의 0.40 짜리 축이 통째로 죽어 있었다.
    """
    from agents.intake import normalize_defect_type, defect_vocabulary

    pcb = defect_vocabulary("pcb1")
    assert normalize_defect_type("미세 스크래치", pcb) == "scratch"
    assert normalize_defect_type("긁힘", pcb) == "scratch"
    assert normalize_defect_type("휨 불량", pcb) == "bent"
    assert normalize_defect_type("기포", pcb) is None, "pcb 에 없는 어휘다"

    fryum = defect_vocabulary("fryum")
    assert normalize_defect_type("미세 스크래치", fryum) == "small scratches", (
        "같은 말이 품목에 따라 다른 값으로 가야 한다"
    )

    capsules = defect_vocabulary("capsules")
    assert normalize_defect_type("기포", capsules) == "bubble"
    assert normalize_defect_type("실선 자국", capsules) is None, "모르면 비운다"


def test_어휘표가_열두_카테고리를_담는다():
    """예선은 pcb 넷만 쓰지만 본선에서 나머지를 쓴다."""
    from agents.intake import defect_vocabulary
    import json, pathlib

    table = json.loads((pathlib.Path("data") / "defect_vocab.json").read_text(encoding="utf-8"))
    table.pop("_comment", None)
    assert len(table) == 12, f"카테고리 {len(table)}개"
    for category, terms in table.items():
        assert terms, f"{category} 어휘가 비었다"
        assert all(aliases for aliases in terms.values()), (
            f"{category} 에 별칭 없는 어휘가 있다 — 한국어로 오면 못 잡는다"
        )


def test_라인을_못_정하면_비우고_되묻는다():
    """**추측하지 않는다.** 번호가 없으면 어느 라인인지 정할 근거가 없다."""
    from agents.intake import normalize, IssueReport

    report = IssueReport(raw_text="")
    report.line, report.object_name = "스크래치 라인", "PCB"
    normalize(report, None)
    assert report.line is None
    assert report.object_name is None
