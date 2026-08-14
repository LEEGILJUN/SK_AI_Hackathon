"""진단 온톨로지 — 원인·판별·조치의 관계를 언어 모델이 읽을 수 있게.

지금까지 원인 6종·판별 7항목·조치 체계는 `diagnose.py` 의 dict 안에만 있었다.
`decide()` 는 그것을 쓰지만 **언어 모델은 못 본다.** 모델이 읽는 구조화된
지식은 도구 명세 열 개와 시스템 프롬프트 몇 줄이 전부였고, 그래서 프롬프트에
"여섯 중 넷은 재구성이 답이 아니다" 같은 문장을 손으로 적어 넣어야 했다.
그 문장은 표와 어긋나도 아무도 모른다.

여기서 하는 일은 하나다. **이미 있는 판정 테이블을 모델이 조회할 수 있는
형태로 노출한다.** 새 지식을 만들지 않는다 — 재구성 필요 여부·권고 조치·금지
조치는 전부 `diagnose.py` 에서 그대로 가져오고, 이 파일이 더하는 것은 사람이
읽을 정의와 **무엇으로 그 원인이 갈리는가**뿐이다.

**이 온톨로지는 판정하지 않는다.**

    조회한다        원인이 무엇을 뜻하는가, 무엇으로 갈리는가, 무엇이 금지인가
    판정하지 않는다  이번 이슈의 원인이 무엇인가

모델이 "뱅크 오염 같으니 그것으로 하자"고 답해도 진단 결과는 바뀌지 않는다.
원인은 `decide()` 가 판별 7항목으로 매번 새로 낸다. 이 경계가 무너지면 진단이
인상 평가가 된다 — 이슈 이력 그래프에 "그래프는 원인을 정하지 않는다"를 못
박아 둔 것과 같은 이유다.

**조회 계층(`lookup/`)에 두지 않은 이유가 있다.** 저기 있는 여덟 함수는 공장
데이터를 조인·집계·그래프 탐색으로 가져오는 것이고, 이동현의 실구현으로
갈아끼워질 자리다. 이 파일은 데이터를 찾지 않는다. 우리 판정 규칙의 스키마이며
갈아끼울 대상이 아니다. `RETRIEVAL_KIND` 에 넣지 않은 것도 같은 이유다 —
넣으면 "조회 아홉 개 중 하나"로 세어져 조회 방식 비중이 흐려진다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from .diagnose import (
    CAUSE_LABEL_KO,
    FORBIDDEN_ACTIONS,
    REBUILD_REQUIRED,
    RECOMMENDED_ACTIONS,
)


@dataclass(frozen=True)
class CauseNode:
    """원인 하나의 정의.

    definition
        무엇을 뜻하는가. 한 문장.
    decided_by
        이 원인을 확정하는 판별 항목 번호. 진단이 실제로 보는 값이다.
    confused_with
        {혼동하기 쉬운 원인: 무엇으로 갈리는가}. **이 항목이 이 파일의 값어치다** —
        원인 목록만 나열하면 모델이 비슷한 둘을 임의로 고르게 된다.
    rebuild_note
        재구성이 답인지 아닌지의 근거. 여섯 중 넷은 아니다.
    """

    cause: str
    definition: str
    decided_by: tuple[int, ...]
    confused_with: dict[str, str]
    rebuild_note: str

    @property
    def label(self) -> str:
        return CAUSE_LABEL_KO[self.cause]

    def to_dict(self) -> dict[str, Any]:
        # 재구성 여부·조치·금지는 여기서 다시 적지 않고 diagnose 에서 가져온다.
        # 두 벌이 되면 한쪽만 고쳐지고, 그때 모델은 틀린 쪽을 읽는다.
        return asdict(self) | {
            "label": self.label,
            "requires_bank_rebuild": REBUILD_REQUIRED[self.cause],
            "rebuild_note": self.rebuild_note,
            "recommended_actions": list(RECOMMENDED_ACTIONS[self.cause]),
            "forbidden_actions": list(FORBIDDEN_ACTIONS[self.cause]),
            "decided_by": list(self.decided_by),
        }


@dataclass(frozen=True)
class CheckItem:
    """판별 항목 하나의 정의.

    source 가 중요하다. 일곱 중 둘만 시각 언어 모델이고 나머지는 조회와
    계산이다. 이 비중이 진단의 신뢰도가 어디에서 나오는지를 말해 준다.
    """

    item_no: int
    name: str
    question: str
    source: str          # vlm | lookup | compute | trace
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: 원인 6종. 재구성 여부·조치·금지는 diagnose.py 가 정답이고 여기 없다.
CAUSES: dict[str, CauseNode] = {
    "threshold": CauseNode(
        cause="threshold",
        definition=(
            "이상 점수는 충분히 높은데 임계값 아래에 있어 검출되지 않았다. "
            "모델은 차이를 보고 있고 자르는 선만 잘못 놓인 상태다."
        ),
        decided_by=(3, 5),
        confused_with={
            "normal_overlap": (
                "임계값 스윕으로 갈린다. 전건 검출이 감당할 과검률로 가능하면 임계값 "
                "문제이고, 어떤 값으로도 안 되면 정상 분포와 겹친 것이다. "
                "스윕이 없으면 점수 위치로 가르되 확신도를 낮춘다."
            ),
        },
        rebuild_note=(
            "뱅크는 멀쩡하다. 다시 만들어도 점수 분포가 그대로라 같은 자리에서 또 놓친다."
        ),
    ),
    "bank_contamination": CauseNode(
        cause="bank_contamination",
        definition=(
            "정상으로 등록되어 뱅크에 들어간 결함이 같은 유형의 불량을 정상으로 "
            "끌어당기고 있다. 최근접 정상 패치를 되짚었더니 실제로는 결함이었다."
        ),
        decided_by=(4, 5),
        confused_with={
            "normal_overlap": (
                "**갈림의 전부가 판별 5번이다.** 되짚은 패치가 잘못 섞인 결함이면 오염, "
                "진짜 정상품이면 중첩이다. 조치가 정반대라 5번을 얻지 못하면 찍지 않고 "
                "사람에게 넘긴다."
            ),
        },
        rebuild_note="오염 샘플을 빼고 다시 만드는 것이 답이다. 여섯 중 재구성이 답인 둘 가운데 하나다.",
    ),
    "coverage_gap": CauseNode(
        cause="coverage_gap",
        definition=(
            "지금 조건(로트·일자·설비·교대조 등)의 정상 패치가 뱅크 구성에 없다. "
            "비교할 기준 자체가 없으니 거리로 가릴 수 없다."
        ),
        decided_by=(5, 6),
        confused_with={
            "bank_contamination": (
                "판별 5번이 먼저다. 되짚은 패치가 결함이면 오염이고, 진짜 정상품인데 "
                "현재 조건이 뱅크 구성에 없으면 커버리지 부족이다."
            ),
        },
        rebuild_note=(
            "빠진 조건의 정상 이미지를 채워 다시 만드는 것이 답이다. "
            "**무엇이 빠졌는지 모른 채 다시 만들면 같은 구멍이 그대로 남는다.**"
        ),
    ),
    "normal_overlap": CauseNode(
        cause="normal_overlap",
        definition=(
            "되짚은 패치가 진짜 정상품인데 형상이 불량과 유사하다. 정상 분포와 겹쳐 "
            "거리 기반으로는 구분되지 않는다."
        ),
        decided_by=(4, 5),
        confused_with={
            "bank_contamination": "판별 5번으로 갈린다. 진짜 정상품이면 여기다.",
            "threshold": (
                "임계값 스윕이 가른다. 어떤 값으로도 전건 검출이 안 되면 중첩이다. "
                "**주의 — 오염된 뱅크에서도 스윕은 같은 '해결 불가'를 낸다.** "
                "스윕만으로는 오염과 중첩을 못 가르고, 역추적이 가른다."
            ),
        },
        rebuild_note=(
            "재구성으로 해결되지 않는다. 정상 패치를 더 넣어도 형상이 겹친다는 사실은 "
            "그대로다. 기준 재정의·전용 판별 로직·촬영 개선이 답이다."
        ),
    ),
    "equipment_optics": CauseNode(
        cause="equipment_optics",
        definition=(
            "화질 지표가 기준 분포를 벗어났다. 조명 열화·초점 이탈·오염 등 설비 쪽 문제이며 "
            "모델 문제가 아니다."
        ),
        decided_by=(2,),
        confused_with={},
        rebuild_note=(
            "재구성을 **차단한다.** 화질이 나간 상태에서는 이상 점수도 최근접 패치도 전부 "
            "오염된 값이라, 그 값으로 만든 뱅크는 설비 이상을 정상으로 학습한다."
        ),
    ),
    "criteria": CauseNode(
        cause="criteria",
        definition=(
            "모델은 검출했는데 판정 기준에서 양품으로 분류됐다. 모델이 놓친 것이 아니라 "
            "기준이 그렇게 정해져 있는 것이다."
        ),
        decided_by=(3, 7),
        confused_with={
            "threshold": (
                "판별 7번이 가른다. 점수가 임계값을 넘었는데 기준 대조에서 양품으로 흘렀으면 "
                "기준 문제이고, 점수가 임계값에 못 미쳤으면 모델 쪽 문제다."
            ),
        },
        rebuild_note="재구성을 차단한다. 모델은 이미 잡았다. 건드릴 곳은 판정 기준이다.",
    ),
}


#: 판별 7항목. 어느 것이 모델이고 어느 것이 조회인지가 함께 있어야 한다.
CHECKS: tuple[CheckItem, ...] = (
    CheckItem(1, "defect_visible", "결함이 이미지에 실제로 보이는가", "vlm",
              "안 보이면 접수 오류일 수 있어 진단을 멈춘다. 전체 이미지로는 놓치므로 "
              "역추적이 가리킨 자리를 잘라 확대해서 묻는다."),
    CheckItem(2, "quality_within_baseline", "화질 지표가 기준 분포 안에 있는가", "compute",
              "여기가 무너지면 3·4·5번이 전부 오염된 값이 된다. 그래서 제일 먼저 본다."),
    CheckItem(3, "score_position", "이상 점수가 임계값 대비 어느 위치인가", "lookup",
              "above | near | below. 임계값은 뱅크가 아니라 운영 설정에서 조회한다."),
    CheckItem(4, "nearest_patch", "가장 가까웠던 정상 패치가 무엇인가", "trace",
              "PatchCore 라서 되짚을 수 있다. 판단 근거가 모델 안에 이미 있다."),
    CheckItem(5, "nearest_patch_is_defect", "그 패치가 잘못 섞인 결함인가 진짜 정상품인가", "vlm",
              "**진단이 갈리는 지점이다.** 결함이면 뱅크 오염, 정상품이면 정상 분포 중첩이고 "
              "조치가 정반대다. 얻지 못하면 판정하지 않는다."),
    CheckItem(6, "coverage_present", "현재 조건의 정상 패치가 뱅크 구성에 있는가", "lookup",
              "조건 축은 여럿이다. 어느 축이 비었는지가 보충할 이미지를 정한다."),
    CheckItem(7, "criteria_verdict", "판정 기준상 불량이 맞는가", "lookup",
              "그 시점에 유효했던 기준으로 본다. 지금 기준으로 과거를 재면 기준 문제를 "
              "영영 못 찾는다."),
)

#: 조치 id 를 사람 말로. 화면과 승인 문서가 같은 이름을 쓰게 한다.
ACTION_LABEL_KO: dict[str, str] = {
    "adjust_threshold": "임계값 재조정",
    "lower_threshold": "임계값 낮추기",
    "remove_contaminated_samples": "오염 샘플 제거",
    "rebuild_bank": "뱅크 재구성",
    "add_normal_images_for_condition": "빠진 조건의 정상 이미지 보충",
    "redefine_criteria": "판정 기준 재정의 요청",
    "add_dedicated_detector": "전용 판별 로직 추가",
    "improve_imaging": "촬영 조건 개선",
    "request_equipment_check": "설비 점검 요청",
    "shadow_compare": "섀도 평가",
    "review_past_issue": "과거 이슈 확인",
    "request_correct_image": "올바른 이미지 재요청",
}

#: 모델이 이 온톨로지를 원인 결정에 쓰지 못하게 하는 문구. 응답마다 함께 나간다.
DISCLAIMER = (
    "이 조회는 원인을 정하지 않는다. 이번 이슈의 원인은 판별 7항목을 모아 "
    "diagnose_issue 도구가 규칙으로 낸다. 여기서 읽은 정의가 그럴듯하다는 이유로 "
    "원인을 고르지 말 것."
)


def action_label(action: str) -> str:
    """조치 id 를 사람 말로. 모르는 id 는 그대로 돌려준다."""
    return ACTION_LABEL_KO.get(action, action)


def cause_names() -> list[str]:
    """원인 6종의 id. diagnose 의 표를 기준으로 한다."""
    return list(REBUILD_REQUIRED)


def describe_cause(cause: str) -> dict[str, Any] | None:
    """원인 하나의 정의·갈림·조치. 없는 원인이면 None."""
    node = CAUSES.get(cause)
    if node is None:
        return None
    return node.to_dict()


def describe_check(item_no: int) -> dict[str, Any] | None:
    """판별 항목 하나. 없는 번호면 None."""
    for item in CHECKS:
        if item.item_no == item_no:
            return item.to_dict()
    return None


def overview() -> dict[str, Any]:
    """원인 6종과 판별 7항목의 요약. 표 전체를 한 번에 준다."""
    rebuild_needed = [c for c in cause_names() if REBUILD_REQUIRED[c]]
    return {
        "causes": [
            {
                "cause": c,
                "label": CAUSE_LABEL_KO[c],
                "definition": CAUSES[c].definition,
                "requires_bank_rebuild": REBUILD_REQUIRED[c],
                "decided_by": list(CAUSES[c].decided_by),
            }
            for c in cause_names()
        ],
        "checks": [item.to_dict() for item in CHECKS],
        "rebuild_is_the_answer_for": rebuild_needed,
        "note": (
            f"원인 {len(cause_names())}종 중 뱅크 재구성이 답인 것은 "
            f"{len(rebuild_needed)}종뿐이다({', '.join(CAUSE_LABEL_KO[c] for c in rebuild_needed)}). "
            f"나머지는 재구성해도 해결되지 않거나 오히려 나빠진다."
        ),
        "disclaimer": DISCLAIMER,
    }


def lookup_ontology(cause: str = "", check_item: int | None = None) -> dict[str, Any]:
    """도구 진입점 — 원인·판별 체계를 조회한다. **판정하지 않는다.**

    인자를 주지 않으면 전체 요약을, 원인을 주면 그 원인의 정의와 갈림을,
    판별 번호를 주면 그 항목의 정의를 돌려준다.

    없는 이름을 물으면 예외를 내지 않고 무엇이 있는지 알려 준다. 모델이
    고쳐 부를 수 있어야 하고, 조회 실패로 진행이 멈추면 안 된다.
    """
    answer: dict[str, Any] = {"disclaimer": DISCLAIMER}

    if not cause and check_item is None:
        return overview() | answer

    if cause:
        found = describe_cause(cause)
        if found is None:
            return answer | {
                "error": f"'{cause}' 는 원인 6종에 없다.",
                "available_causes": cause_names(),
            }
        answer["cause"] = found

    if check_item is not None:
        found_check = describe_check(check_item)
        if found_check is None:
            return answer | {
                "error": f"판별 항목 {check_item} 번은 없다. 1~{len(CHECKS)} 중에서 고른다.",
                "available_checks": [item.item_no for item in CHECKS],
            }
        answer["check"] = found_check

    return answer
