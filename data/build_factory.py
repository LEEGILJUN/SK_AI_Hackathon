# 이 스크립트는 현재 스크립트 파일 기준 상대경로를 사용해 공장 데이터셋을 생성합니다.
import csv
import hashlib
import os
import random
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import time
from datetime import timedelta
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import yaml

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RANDOM_SEED = 20260814
#: VisA 원본은 저장소 루트에 푼다. `app/pipeline.py` 와 같은 규약이라
#: `SHVO_VISA_ROOT` 로 옮길 수 있다 — 두 군데가 다른 자리를 보면
#: 화면과 공장 데이터가 서로 다른 원본으로 서게 된다.
SOURCE_ROOT = Path(os.environ.get("SHVO_VISA_ROOT") or REPO_ROOT / "VisA_20220922")
SCENARIO_PATH = BASE_DIR / "scenarios.yaml"
FACTORY_ROOT = BASE_DIR / "factory"
MANIFEST_PATH = BASE_DIR / "manifest.csv"
MES_PATH = BASE_DIR / "mes.csv"
SUMMARY_PATH = BASE_DIR / "factory_summary.txt"

FULL_FREE_SPACE_THRESHOLD_GB = 10
NORMAL_IMAGES_PER_LOT = 900
ANOMALY_IMAGES_PER_LOT = 100
MAX_WORKERS = 8

VALID_LINES = {
    "line_01": "pcb1",
    "line_02": "pcb2",
    "line_03": "pcb3",
    "line_04": "pcb4",
}

LOT_CODE_BY_START_DATE = {
    date(2026, 6, 5): "AAE",
    date(2026, 6, 8): "AAH",
    date(2026, 6, 10): "AAJ",
    date(2026, 6, 12): "AAL",
    date(2026, 6, 14): "AAN",
    date(2026, 6, 16): "AAP",
    date(2026, 6, 18): "AAR",
    date(2026, 6, 19): "AAS",
    date(2026, 6, 22): "AAV",
    date(2026, 6, 24): "AAX",
    date(2026, 6, 25): "AAY",
    date(2026, 6, 26): "AAZ",
    date(2026, 6, 28): "ABB",
    date(2026, 6, 30): "ABD",
    date(2026, 7, 1): "ABE",
    date(2026, 7, 2): "ABF",
    date(2026, 7, 3): "ABG",
    date(2026, 7, 4): "ABH",
    date(2026, 7, 9): "ABM",
    date(2026, 7, 10): "ABN",
    date(2026, 7, 15): "ABS",
    date(2026, 7, 18): "ABV",
    date(2026, 7, 22): "ABZ",
    date(2026, 7, 25): "ACC",
}

# 이길준 수정 (2026-08-14): "dirt" 를 뺐다. VisA pcb1~4 결함 어휘 어디에도
# 없는 이름이라 ESA3 은 한 번도 붙은 적이 없다. 코드는 이어 붙이지 않고
# 비워 둔다 — 뒤를 당기면 이미 나간 값의 뜻이 바뀐다.
ERROR_CODE_BY_DEFECT_NAME = {
    "normal": "0000",
    "bent": "ESA0",
    "burnt": "ESA1",
    "damage": "ESA2",
    "extra": "ESA4",
    "melt": "ESA5",
    "missing": "ESA6",
    "scratch": "ESA7",
    "wrong place": "ESA8",
}

# 이길준 수정 (2026-08-14): 이물·표면을 없는 결함(dirt)에 물려 두어 두 낱말이
# 조용히 무시되고 있었다. pcb 어휘 안으로 옮긴다. "오염"은 대응할 결함이
# 정말 없어서 뺐다 — 억지로 물리면 엉뚱한 이미지가 붙는다.
ISSUE_KEYWORD_TO_DEFECT = {
    "크랙": "damage",
    "파손": "damage",
    "이물": "extra",
    "휘": "bent",
    "굽": "bent",
    "소손": "burnt",
    "탄화": "burnt",
    "스크래치": "scratch",
    "긁": "scratch",
    "용융": "melt",
    "녹": "melt",
    "미삽": "missing",
    "누락": "missing",
    "위치 오류": "wrong place",
    "오삽": "wrong place",
    "잉여물": "extra",
    "돌기": "extra",
    "배선": "scratch",
    "표면": "scratch",
}


@dataclass
class SourceSample:
    object_name: str
    image_rel: str
    label_raw: str
    mask_rel: str
    primary_defect: str


@dataclass
class ScenarioInfo:
    scenario_id: str
    title: str
    cause_group: str
    line: str
    object_name: str
    lot_ids: List[str]
    date_start: date
    date_end: date
    issue_text: str
    attachments: List[str]
    #: 이길준 추가 (2026-08-14): injection 절. 장영진이 뱅크에 넣을 오염
    #: 이미지를 여기에 정확히 지정해 뒀는데 읽지 않고 있었다.
    injection_method: str = ""
    contaminated_count: int = 0
    contaminated_images: List[str] = None


@dataclass
class ManifestRow:
    image_path: str
    line: str
    object_name: str
    date_str: str
    lot_id: str
    equipment_id: str
    split: str
    label: str
    mask_path: str
    visa_source: str
    defect_name: str
    ercd: str


@dataclass
class CopyTask:
    source_rel: str
    destination_path: Path


def make_seed(*parts: str) -> int:
    joined = "||".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def daterange(start_date: date, end_date: date) -> List[date]:
    result = []
    current = start_date
    while current <= end_date:
        result.append(current)
        current = current + timedelta(days=1)
    return result


def load_scenarios(path: Path) -> Tuple[List[ScenarioInfo], date, date, Dict[str, int]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenario_rows = []
    cause_counter = defaultdict(int)
    min_date_value = None
    max_date_value = None
    for item in payload.get("scenarios", []):
        target = item.get("target", {})
        line = target.get("line")
        object_name = target.get("object")
        date_range_value = target.get("date_range", [])
        if line not in VALID_LINES:
            continue
        if object_name != VALID_LINES[line]:
            continue
        if len(date_range_value) != 2:
            continue
        injection = item.get("injection", {}) or {}
        injection_params = injection.get("params", {}) or {}
        scenario = ScenarioInfo(
            injection_method=str(injection.get("method", "")),
            contaminated_count=int(injection_params.get("contaminated_count", 0) or 0),
            contaminated_images=[
                str(value).replace("\\", "/")
                for value in (injection_params.get("contaminated_images") or [])
            ],
            scenario_id=str(item.get("id", "")),
            title=str(item.get("title", "")),
            cause_group=str(item.get("cause_group", "")),
            line=str(line),
            object_name=str(object_name),
            lot_ids=[str(lot_id) for lot_id in target.get("lot_ids", [])],
            date_start=datetime.fromisoformat(str(date_range_value[0])).date(),
            date_end=datetime.fromisoformat(str(date_range_value[1])).date(),
            issue_text=str(item.get("input", {}).get("issue_text", "")),
            attachments=[str(value).replace("\\", "/") for value in item.get("input", {}).get("attachments", [])],
        )
        scenario_rows.append(scenario)
        cause_counter[scenario.cause_group] += 1
        min_date_value = scenario.date_start if min_date_value is None else min(min_date_value, scenario.date_start)
        max_date_value = scenario.date_end if max_date_value is None else max(max_date_value, scenario.date_end)
    if min_date_value is None or max_date_value is None:
        raise ValueError("유효한 PCB 시나리오를 scenarios.yaml에서 찾지 못했습니다.")
    return scenario_rows, min_date_value, max_date_value, dict(cause_counter)


def load_source_samples() -> Tuple[Dict[str, List[SourceSample]], Dict[str, List[SourceSample]]]:
    normal_by_object = defaultdict(list)
    anomaly_by_object = defaultdict(list)
    for object_name in VALID_LINES.values():
        anno_path = SOURCE_ROOT / object_name / "image_anno.csv"
        with anno_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                image_rel = str(row.get("image", "")).strip()
                label_raw = str(row.get("label", "")).strip()
                mask_rel = str(row.get("mask", "")).strip()
                primary_defect = label_raw.split(",")[0].strip() if label_raw and label_raw != "normal" else "normal"
                sample = SourceSample(
                    object_name=object_name,
                    image_rel=image_rel,
                    label_raw=label_raw,
                    mask_rel=mask_rel,
                    primary_defect=primary_defect,
                )
                if label_raw == "normal":
                    normal_by_object[object_name].append(sample)
                else:
                    anomaly_by_object[object_name].append(sample)
    return dict(normal_by_object), dict(anomaly_by_object)


def determine_lot_code(target_date: date) -> str:
    eligible_dates = [start_date for start_date in LOT_CODE_BY_START_DATE if start_date <= target_date]
    if not eligible_dates:
        raise ValueError(f"LOT 코드 규칙보다 이른 날짜가 입력되었습니다: {target_date}")
    matched_start = max(eligible_dates)
    return LOT_CODE_BY_START_DATE[matched_start]


def pick_split(target_date: date, line: str, index_within_day: int, label: str) -> str:
    """이 이미지가 뱅크용인가 운영용인가 홀드아웃인가.

    **이길준 수정 (2026-08-14): label 을 보고 정하게 바꿨다.**

    전에는 split 해시와 label 해시가 서로 독립이라 결함 이미지가 뱅크로 그냥
    들어갔다. 로트 하나를 재현해 세어 보니 `split="bank"` 안에 결함이 9.1%
    (633 정상 / 63 결함) 였고, **모든 라인 · 모든 로트에서 그렇게 된다.**

    두 가지가 깨진다.

    1. 우리 설계는 한 품목만 오염시킨다. 넷이 동시에 오염되면 "이 라인만
       문제다"를 보여줄 수 없고, 진단 원인 여섯 중 뱅크 오염이 언제나 참이
       되어 나머지 다섯을 가를 수가 없다.
    2. 오염률을 해시가 정한다. 통제가 안 되니 "이 뱅크는 오염 3.2%" 라고 쓸
       수 없다. 실측값은 조건을 맞춰 잰 것인데 여기엔 조건 자체가 없다.

    그래서 **뱅크에는 정상만 담는다.** 오염은 우연이 아니라 시나리오가
    `injection.params.contaminated_*` 로 지정한 것만 들어간다
    (`apply_bank_contamination`). 결함은 운영·홀드아웃으로만 가며, 둘 사이
    비율(20:10)은 정상 쪽과 같게 유지한다.
    """
    selector = make_seed(target_date.isoformat(), line, str(index_within_day), "split") % 100
    if label == "defect":
        return "operation" if selector < 67 else "holdout"
    if selector < 70:
        return "bank"
    if selector < 90:
        return "operation"
    return "holdout"


def error_code_for(defect_name: str) -> str:
    """결함 이름 → 설비 오류 코드.

    **이길준 추가 (2026-08-14): 모르는 결함에 "0000"(정상)을 주지 않는다.**

    전에는 `ERROR_CODE_BY_DEFECT_NAME.get(name, "0000")` 이었다. 손으로 적은
    표라서 카테고리를 늘리면 표에 없는 이름이 반드시 나오는데, 그때 정상
    코드가 붙어 **MES 집계에서 불량이 통째로 사라진다.** 표에 없으면 이름에서
    결정론적으로 만든다 — ESA 는 손으로 정한 것, ESB 는 파생된 것이다.
    """
    if defect_name == "normal":
        return "0000"
    known = ERROR_CODE_BY_DEFECT_NAME.get(defect_name)
    if known:
        return known
    return f"ESB{make_seed(defect_name, 'ercd') % 100:02d}"


def build_equipment_id(line: str, lot_id: str) -> str:
    line_number = line.split("_")[-1]
    equipment_suffixes = ["A", "B", "C", "D"]
    suffix_index = make_seed(line, lot_id, "equipment") % len(equipment_suffixes)
    return f"EQ-{line_number}-{equipment_suffixes[suffix_index]}"


def choose_defect_from_issue(issue_text: str, object_name: str, anomaly_pool: Dict[str, List[SourceSample]]) -> str:
    normalized_text = issue_text.strip()
    for keyword, defect_name in ISSUE_KEYWORD_TO_DEFECT.items():
        if keyword in normalized_text:
            available_defects = {sample.primary_defect for sample in anomaly_pool[object_name]}
            if defect_name in available_defects:
                return defect_name
    defect_counts = defaultdict(int)
    for sample in anomaly_pool[object_name]:
        defect_counts[sample.primary_defect] += 1
    if not defect_counts:
        raise ValueError(f"불량 샘플이 존재하지 않습니다: {object_name}")
    return sorted(defect_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def choose_anomaly_sample(scenario: ScenarioInfo, attachment_path: str, anomaly_pool: Dict[str, List[SourceSample]]) -> SourceSample:
    desired_defect = choose_defect_from_issue(scenario.issue_text, scenario.object_name, anomaly_pool)
    candidates = [sample for sample in anomaly_pool[scenario.object_name] if sample.primary_defect == desired_defect]
    if not candidates:
        candidates = list(anomaly_pool[scenario.object_name])
    ordered = sorted(candidates, key=lambda sample: sample.image_rel)
    randomizer = random.Random(make_seed(scenario.scenario_id, attachment_path, desired_defect))
    return ordered[randomizer.randrange(len(ordered))]


def get_free_space_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def select_execution_scenarios(scenarios: List[ScenarioInfo], free_space_gb: float) -> Tuple[str, List[ScenarioInfo]]:
    if free_space_gb >= FULL_FREE_SPACE_THRESHOLD_GB:
        return "full", scenarios
    line_01_scenarios = [scenario for scenario in scenarios if scenario.line == "line_01"]
    line_03_scenarios = [scenario for scenario in scenarios if scenario.line == "line_03"]
    if not line_01_scenarios or not line_03_scenarios:
        raise ValueError("간이 실행용 line_01 또는 line_03 시나리오를 찾지 못했습니다.")
    earliest_line_01 = sorted(line_01_scenarios, key=lambda scenario: (scenario.date_start, scenario.scenario_id))[0]
    latest_line_03 = sorted(line_03_scenarios, key=lambda scenario: (scenario.date_end, scenario.scenario_id), reverse=True)[0]
    selected = [earliest_line_01]
    if latest_line_03.scenario_id != earliest_line_01.scenario_id:
        selected.append(latest_line_03)
    return "light", selected


def build_copy_task(source_rel: str, destination_path: Path) -> CopyTask:
    return CopyTask(source_rel=source_rel, destination_path=destination_path)


def execute_copy_tasks(copy_tasks: List[CopyTask]) -> None:
    unique_tasks = {}
    for task in copy_tasks:
        unique_tasks[str(task.destination_path)] = task
    tasks = list(unique_tasks.values())
    def _copy(task: CopyTask) -> None:
        source_path = SOURCE_ROOT / Path(task.source_rel)
        task.destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, task.destination_path)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(_copy, tasks))


def collect_target_lots(selected_scenarios: List[ScenarioInfo]) -> List[Tuple[str, str, str]]:
    lot_keys = set()
    for scenario in selected_scenarios:
        for attachment_path in scenario.attachments:
            attachment = Path(attachment_path)
            lot_keys.add((scenario.line, attachment.parts[1], attachment.parts[2]))
    return sorted(lot_keys)


def plan_lot_generation(selected_scenarios: List[ScenarioInfo]) -> Dict[Tuple[str, str, str], Set[int]]:
    lot_image_indexes = {}
    for line, date_str, lot_id in collect_target_lots(selected_scenarios):
        indexes = set()
        rng = random.Random(make_seed(line, date_str, lot_id, "base-lot-population"))
        while len(indexes) < NORMAL_IMAGES_PER_LOT + ANOMALY_IMAGES_PER_LOT:
            indexes.add(rng.randint(1, 9999))
        lot_image_indexes[(line, date_str, lot_id)] = indexes
    return lot_image_indexes


def choose_background_anomaly_sample(object_name: str, anomaly_pool: Dict[str, List[SourceSample]], line: str, date_str: str, lot_id: str, image_index: int) -> SourceSample:
    ordered = sorted(anomaly_pool[object_name], key=lambda sample: sample.image_rel)
    if not ordered:
        raise ValueError(f"불량 샘플이 존재하지 않습니다: {object_name}")
    rng = random.Random(make_seed(line, object_name, date_str, lot_id, str(image_index), "background-anomaly"))
    return ordered[rng.randrange(len(ordered))]


def choose_normal_sample(normal_pool: Dict[str, List[SourceSample]], object_name: str, line: str, date_str: str, lot_id: str, image_index: int) -> SourceSample:
    ordered = sorted(normal_pool[object_name], key=lambda sample: sample.image_rel)
    if not ordered:
        raise ValueError(f"정상 샘플이 존재하지 않습니다: {object_name}")
    rng = random.Random(make_seed(line, object_name, date_str, lot_id, str(image_index), "normal-selection"))
    return ordered[rng.randrange(len(ordered))]


def choose_background_anomaly_indexes(sorted_indexes: List[int], line: str, date_str: str, lot_id: str) -> Set[int]:
    if len(sorted_indexes) < ANOMALY_IMAGES_PER_LOT:
        raise ValueError(f"배경 불량 인덱스가 부족합니다: {line}/{date_str}/{lot_id}")
    rng = random.Random(make_seed(line, date_str, lot_id, "background-anomaly-indexes"))
    shuffled_indexes = list(sorted_indexes)
    rng.shuffle(shuffled_indexes)
    return set(shuffled_indexes[:ANOMALY_IMAGES_PER_LOT])


def build_base_factory(normal_pool: Dict[str, List[SourceSample]], anomaly_pool: Dict[str, List[SourceSample]], selected_scenarios: List[ScenarioInfo]) -> List[ManifestRow]:
    if FACTORY_ROOT.exists():
        shutil.rmtree(FACTORY_ROOT)
    FACTORY_ROOT.mkdir(parents=True, exist_ok=True)
    lot_image_indexes = plan_lot_generation(selected_scenarios)
    manifest_rows = []
    copy_tasks = []
    for (line, date_str, lot_id), image_indexes in sorted(lot_image_indexes.items()):
        object_name = VALID_LINES[line]
        sorted_indexes = sorted(image_indexes)
        background_anomaly_indexes = choose_background_anomaly_indexes(sorted_indexes, line, date_str, lot_id)
        for image_index in sorted_indexes:
            image_filename = f"img_{image_index:04d}.png"
            relative_image_path = Path(line) / date_str / lot_id / image_filename
            relative_image_text = relative_image_path.as_posix()
            if image_index in background_anomaly_indexes:
                sample = choose_background_anomaly_sample(object_name, anomaly_pool, line, date_str, lot_id, image_index)
                label = "defect"
                defect_name = sample.primary_defect
                mask_path = sample.mask_rel if sample.mask_rel else ""
            else:
                sample = choose_normal_sample(normal_pool, object_name, line, date_str, lot_id, image_index)
                label = "normal"
                defect_name = "normal"
                mask_path = ""
            destination_image_path = FACTORY_ROOT / relative_image_path
            copy_tasks.append(build_copy_task(sample.image_rel, destination_image_path))
            manifest_rows.append(
                ManifestRow(
                    image_path=relative_image_text,
                    line=line,
                    object_name=object_name,
                    date_str=date_str,
                    lot_id=lot_id,
                    equipment_id=build_equipment_id(line, lot_id),
                    split=pick_split(date.fromisoformat(date_str), line, image_index, label),
                    label=label,
                    mask_path=mask_path,
                    visa_source=sample.image_rel,
                    defect_name=defect_name,
                    ercd=error_code_for(defect_name),
                )
            )
    execute_copy_tasks(copy_tasks)
    return manifest_rows


def apply_scenarios_to_factory(manifest_rows: List[ManifestRow], anomaly_pool: Dict[str, List[SourceSample]], selected_scenarios: List[ScenarioInfo]) -> List[ManifestRow]:
    manifest_by_path = {row.image_path: row for row in manifest_rows}
    copy_tasks = []
    for scenario in selected_scenarios:
        for attachment_path in scenario.attachments:
            attachment = Path(attachment_path)
            line = scenario.line
            date_str = attachment.parts[1]
            lot_id = attachment.parts[2]
            object_name = VALID_LINES[line]
            image_index = int(attachment.name.replace("img_", "").replace(".png", ""))
            sample = choose_anomaly_sample(scenario, attachment_path, anomaly_pool)
            replacement_row = ManifestRow(
                image_path=attachment_path,
                line=line,
                object_name=object_name,
                date_str=date_str,
                lot_id=lot_id,
                equipment_id=build_equipment_id(line, lot_id),
                split=pick_split(date.fromisoformat(date_str), line, image_index, "defect"),
                label="defect",
                mask_path=sample.mask_rel if sample.mask_rel else "",
                visa_source=sample.image_rel,
                defect_name=sample.primary_defect,
                ercd=error_code_for(sample.primary_defect),
            )
            manifest_by_path[attachment_path] = replacement_row
            copy_tasks.append(build_copy_task(sample.image_rel, FACTORY_ROOT / attachment))
    execute_copy_tasks(copy_tasks)
    return [manifest_by_path[path] for path in sorted(manifest_by_path)]


#: 오염 이미지가 쓰는 인덱스 대역. 기본 로트는 1~9999 를 쓴다.
#: 대역을 갈라 두면 manifest 만 보고도 "이건 일부러 넣은 오염"임이 드러난다.
CONTAMINANT_INDEX_BASE = 10000

#: 이 아래 오염률이면 경고를 남긴다. VisA 실측에서 오염 이미지가 3.2% 일 때
#: 결함 **위** 패치는 뱅크의 0.1%(6/4,861)뿐이었다. coreset 이 끌어올리는 것은
#: "결함이 있는 이미지"이지 "결함 그 자체"가 아니라서, 오염률이 낮으면 결함
#: 패치가 한 장도 안 남는다. 그러면 역추적이 짚을 것이 없다.
MIN_DETECTABLE_CONTAMINATION_PCT = 1.0


def apply_bank_contamination(
    manifest_rows: List[ManifestRow],
    anomaly_pool: Dict[str, List[SourceSample]],
    selected_scenarios: List[ScenarioInfo],
) -> Tuple[List[ManifestRow], List[str]]:
    """뱅크 오염을 시나리오가 지정한 만큼만 넣는다.

    **이길준 추가 (2026-08-14).**

    `pick_split` 이 결함을 뱅크에서 막게 되면서 오염이 들어올 길이 없어졌다.
    그런데 없어도 되는 게 아니다 — 시나리오 다섯 건이 뱅크 오염을 정답으로
    걸고 있다. 우연히 섞이던 것을 **의도한 것만 들어오게** 바꾼 것이지
    없앤 것이 아니다.

    장영진이 `injection.params` 에 이미 지정해 뒀다.

        method: bank_contamination
        params:
          contaminated_count: 3
          contaminated_images: [pcb1/Data/Images/Anomaly/004.JPG]

    `contaminated_images` 는 이름을 밝힌 것이고 `contaminated_count` 가
    전체 장수다. 밝힌 것을 먼저 넣고 모자라면 같은 품목 결함에서 결정론적으로
    채운다. **정답 파일은 읽기만 한다** — 이 값이 채점 기준이므로 여기서
    바꾸면 측정이 무의미해진다.
    """
    notes = ["[뱅크 오염 주입]"]
    contaminated = [s for s in selected_scenarios if s.injection_method == "bank_contamination"]
    if not contaminated:
        notes.append("- 이번 실행 시나리오에 bank_contamination 이 없습니다.")
        return manifest_rows, notes

    added: List[ManifestRow] = []
    copy_tasks: List[CopyTask] = []
    for scenario in contaminated:
        pool = sorted(anomaly_pool.get(scenario.object_name, []), key=lambda s: s.image_rel)
        if not pool:
            raise ValueError(f"오염에 쓸 결함 샘플이 없습니다: {scenario.object_name}")
        by_rel = {s.image_rel: s for s in pool}

        chosen: List[SourceSample] = []
        for named in scenario.contaminated_images:
            sample = by_rel.get(named)
            if sample is None:
                raise ValueError(
                    f"{scenario.scenario_id}: contaminated_images 가 가리키는 원본이 "
                    f"VisA 에 없습니다: {named}"
                )
            chosen.append(sample)

        # 밝히지 않은 나머지는 결정론적으로 채운다. 같은 시나리오면 같은 것이 나온다.
        rng = random.Random(make_seed(scenario.scenario_id, "bank-contamination"))
        remaining = [s for s in pool if s.image_rel not in {c.image_rel for c in chosen}]
        rng.shuffle(remaining)
        while len(chosen) < scenario.contaminated_count and remaining:
            chosen.append(remaining.pop())

        if len(chosen) < scenario.contaminated_count:
            raise ValueError(
                f"{scenario.scenario_id}: 오염 {scenario.contaminated_count}장을 채우지 "
                f"못했습니다 ({scenario.object_name} 결함 {len(pool)}장)"
            )

        # 시나리오가 가리키는 첫 로트에 얹는다. 뱅크는 라인 단위라 어느 로트에
        # 넣든 같은 뱅크로 들어가지만, 자리를 정해 두어야 재현된다.
        target_lots = collect_target_lots([scenario])
        if not target_lots:
            raise ValueError(f"{scenario.scenario_id}: 오염을 넣을 로트를 찾지 못했습니다.")
        line, date_str, lot_id = target_lots[0]

        for offset, sample in enumerate(chosen):
            image_index = CONTAMINANT_INDEX_BASE + offset
            relative_image_path = Path(line) / date_str / lot_id / f"img_{image_index:05d}.png"
            added.append(
                ManifestRow(
                    image_path=relative_image_path.as_posix(),
                    line=line,
                    object_name=scenario.object_name,
                    date_str=date_str,
                    lot_id=lot_id,
                    equipment_id=build_equipment_id(line, lot_id),
                    # **여기가 핵심이다.** 결함인데 split 이 bank 다. 우연이
                    # 아니라 시나리오가 그렇게 지정했기 때문에 들어간다.
                    split="bank",
                    label="defect",
                    mask_path=sample.mask_rel if sample.mask_rel else "",
                    visa_source=sample.image_rel,
                    defect_name=sample.primary_defect,
                    ercd=error_code_for(sample.primary_defect),
                )
            )
            copy_tasks.append(build_copy_task(sample.image_rel, FACTORY_ROOT / relative_image_path))

        notes.append(
            f"- {scenario.scenario_id} | {line}/{scenario.object_name} | "
            f"오염 {len(chosen)}장 (지정 {len(scenario.contaminated_images)} + "
            f"채움 {len(chosen) - len(scenario.contaminated_images)}) → {lot_id}"
        )

    execute_copy_tasks(copy_tasks)
    merged = {row.image_path: row for row in manifest_rows}
    for row in added:
        merged[row.image_path] = row

    # 라인별로 뱅크 오염률을 적어 둔다. 실측과 나란히 놓을 수 있어야 한다.
    bank_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"normal": 0, "defect": 0})
    for row in merged.values():
        if row.split == "bank":
            bank_stats[row.line][row.label] += 1
    notes.append("- 라인별 뱅크 구성:")
    for line in sorted(bank_stats):
        stat = bank_stats[line]
        total = stat["normal"] + stat["defect"]
        share = stat["defect"] / total * 100 if total else 0.0
        flag = "  ← 검출 한계 아래일 수 있음" if 0 < share < MIN_DETECTABLE_CONTAMINATION_PCT else ""
        notes.append(
            f"  - {line}: 정상 {stat['normal']:,} / 오염 {stat['defect']} "
            f"= 오염률 {share:.2f}%{flag}"
        )
    if any(0 < (s["defect"] / (s["normal"] + s["defect"]) * 100) < MIN_DETECTABLE_CONTAMINATION_PCT
           for s in bank_stats.values() if s["normal"] + s["defect"]):
        notes.append(
            "- 주의: VisA 실측에서 오염 3.2% 일 때 결함 위 패치는 뱅크의 0.1% 였다"
            f"(6/4,861). 오염률이 {MIN_DETECTABLE_CONTAMINATION_PCT}% 아래면 coreset 을"
            " 거친 뒤 결함 패치가 한 장도 안 남을 수 있고, 그러면 역추적이 오염원을"
            " 짚지 못해 bank_contamination 시나리오가 재현되지 않는다."
        )
        notes.append(
            "  contaminated_count 는 채점 기준이라 여기서 고치지 않는다."
            " 장영진 확인이 필요하다."
        )
    return [merged[path] for path in sorted(merged)], notes


def verify_scenario_integration(selected_scenarios: List[ScenarioInfo], manifest_rows: List[ManifestRow], mes_rows: List[Dict[str, str]]) -> List[str]:
    manifest_by_path = {row.image_path: row for row in manifest_rows}
    mes_lookup = defaultdict(list)
    for row in mes_rows:
        mes_lookup[(row["lot_id"], row["line"], row["date"], row["ERCD"])].append(row)
    verification_lines = []
    verification_lines.append("[시나리오 반영 검증]")
    for scenario in selected_scenarios:
        verification_lines.append(f"- {scenario.scenario_id} 검증 시작")
        if not scenario.attachments:
            raise ValueError(f"시나리오 attachment가 없습니다: {scenario.scenario_id}")
        for attachment_path in scenario.attachments:
            manifest_row = manifest_by_path.get(attachment_path)
            if manifest_row is None:
                raise ValueError(f"시나리오 attachment가 manifest에 없습니다: {scenario.scenario_id} | {attachment_path}")
            if manifest_row.label != "defect":
                raise ValueError(f"시나리오 attachment가 defect로 반영되지 않았습니다: {scenario.scenario_id} | {attachment_path}")
            if not manifest_row.mask_path:
                raise ValueError(f"시나리오 attachment의 mask_path가 비어 있습니다: {scenario.scenario_id} | {attachment_path}")
            factory_image_path = FACTORY_ROOT / Path(attachment_path)
            if not factory_image_path.exists():
                raise ValueError(f"시나리오 attachment 이미지가 factory에 없습니다: {scenario.scenario_id} | {factory_image_path}")
            mes_matches = mes_lookup.get((manifest_row.lot_id, manifest_row.line, manifest_row.date_str, manifest_row.ercd), [])
            if not mes_matches:
                raise ValueError(f"시나리오 attachment와 연결되는 MES 행을 찾지 못했습니다: {scenario.scenario_id} | {attachment_path} | ERCD={manifest_row.ercd}")
            verification_lines.append(f"  - OK | attachment={attachment_path} | lot={manifest_row.lot_id} | ercd={manifest_row.ercd} | visa={manifest_row.visa_source}")
    return verification_lines


def compute_shift(started_at_value: datetime) -> str:
    """교대 이름.

    **이길준 수정 (2026-08-14): 이름이 한 칸씩 밀려 있었다.** 자정~08시가
    "day", 08~16시가 "swing" 이었다. 통상 주간이 08–16, 스윙이 16–24,
    야간이 00–08 이다. 기능에는 영향이 없지만 심사에서 눈에 띌 자리다.
    """
    current_time = started_at_value.time()
    if time(0, 0, 0) <= current_time < time(8, 0, 0):
        return "night"
    if time(8, 0, 0) <= current_time < time(16, 0, 0):
        return "day"
    return "swing"


def build_mes_rows(manifest_rows: List[ManifestRow]) -> List[Dict[str, str]]:
    """이미지마다 MES 한 행.

    **이길준 수정 (2026-08-14): inspected_count · defect_count 가 개수가
    아니라 일련번호였다.**

    전에는 결함 행마다 1, 2, 3… 이 들어가고 정상 행은 0 이었다. 열 이름이
    count 인데 뜻은 "이 로트의 n번째 결함" 이라, 합계를 내면 개수가 아니라
    삼각수(1+2+3+…)가 나온다. 승인 문서에 "이 로트 불량 N건"을 적는 자리에서
    그대로 틀린다.

    이제 **로트 총량**을 넣는다. 같은 로트의 모든 행이 같은 값을 갖는다 —
    이미지 단위 표에서 로트 집계를 얻으려고 max 나 last 를 취해야 하는 것보다,
    조인해서 바로 읽히는 편이 조회 계층(`lookup.defect_distribution`)에 맞다.
    """
    sorted_rows = sorted(manifest_rows, key=lambda row: (row.line, row.date_str, row.lot_id, row.image_path))
    lot_totals: Dict[Tuple[str, str, str], Dict[str, int]] = defaultdict(lambda: {"inspected": 0, "defect": 0})
    for row in sorted_rows:
        totals = lot_totals[(row.line, row.date_str, row.lot_id)]
        totals["inspected"] += 1
        if row.label == "defect":
            totals["defect"] += 1
    previous_end_by_line = {}
    mes_rows = []
    for row in sorted_rows:
        previous_end = previous_end_by_line.get(row.line)
        if previous_end is None:
            started_at_value = datetime.fromisoformat(f"{row.date_str}T00:00:01")
        else:
            gap_seconds = 5 + random.Random(make_seed(row.line, row.date_str, row.lot_id, row.image_path, "gap")).randint(-2, 2)
            started_at_value = previous_end + timedelta(seconds=gap_seconds)
            minimum_start = datetime.fromisoformat(f"{row.date_str}T00:00:01")
            if started_at_value < minimum_start:
                started_at_value = minimum_start
        duration_seconds = 60 + random.Random(make_seed(row.line, row.date_str, row.lot_id, row.image_path, "duration")).randint(-10, 10)
        end_at_value = started_at_value + timedelta(seconds=duration_seconds)
        previous_end_by_line[row.line] = end_at_value
        totals = lot_totals[(row.line, row.date_str, row.lot_id)]
        inspected_count_value = totals["inspected"]
        defect_count_value = totals["defect"]
        image_filename = Path(row.image_path).name
        cell_id_value = f"{row.lot_id}_{row.line}_{row.date_str}_{image_filename}"
        mes_rows.append(
            {
                "lot_id": row.lot_id,
                "cell_id": cell_id_value,
                "line": row.line,
                "object": row.object_name,
                "date": row.date_str,
                "started_at": started_at_value.isoformat(timespec="seconds"),
                "end_at": end_at_value.isoformat(timespec="seconds"),
                "equipment_id": row.equipment_id,
                "inspected_count": str(inspected_count_value),
                "defect_count": str(defect_count_value),
                "operator_shift": compute_shift(started_at_value),
                "ERCD": row.ercd,
            }
        )
    return mes_rows


def write_manifest_csv(manifest_rows: List[ManifestRow]) -> None:
    try:
        with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["image_path", "line", "object", "date", "lot_id", "equipment_id", "split", "label", "mask_path", "visa_source"])
            for row in sorted(manifest_rows, key=lambda item: item.image_path):
                writer.writerow([row.image_path, row.line, row.object_name, row.date_str, row.lot_id, row.equipment_id, row.split, row.label, row.mask_path, row.visa_source])
    except PermissionError as error:
        raise PermissionError(f"manifest.csv 파일이 열려 있어 저장할 수 없습니다: {MANIFEST_PATH}") from error


def write_mes_csv(mes_rows: List[Dict[str, str]]) -> None:
    try:
        with MES_PATH.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["lot_id", "cell_id", "line", "object", "date", "started_at", "end_at", "equipment_id", "inspected_count", "defect_count", "operator_shift", "ERCD"])
            writer.writeheader()
            for row in mes_rows:
                writer.writerow(row)
    except PermissionError as error:
        raise PermissionError(f"mes.csv 파일이 열려 있어 저장할 수 없습니다: {MES_PATH}") from error


def write_summary(mode: str, free_space_gb: float, scenarios: List[ScenarioInfo], selected_scenarios: List[ScenarioInfo], min_date_value: date, max_date_value: date, cause_counts: Dict[str, int], manifest_rows: List[ManifestRow], mes_rows: List[Dict[str, str]], verification_lines: List[str]) -> None:
    lines = []
    lines.append("[실행 정보]")
    lines.append(f"- 실행 모드: {mode}")
    lines.append(f"- 시작 시 여유 용량(GB): {free_space_gb:.2f}")
    lines.append(f"- 전체 시나리오 수: {len(scenarios)}")
    lines.append(f"- 이번 실행 시나리오 수: {len(selected_scenarios)}")
    lines.append("[시나리오 분석 요약]")
    lines.append(f"- 이벤트 최소 일자: {min_date_value.isoformat()}")
    lines.append(f"- 이벤트 최대 일자: {max_date_value.isoformat()}")
    lines.append("- 원인 그룹별 건수:")
    for cause_group in sorted(cause_counts):
        lines.append(f"  - {cause_group}: {cause_counts[cause_group]}")
    lines.append("- 실행 시나리오 상세:")
    for scenario in selected_scenarios:
        lines.append(f"  - {scenario.scenario_id} | {scenario.cause_group} | {scenario.line}/{scenario.object_name} | LOT={','.join(scenario.lot_ids)} | RANGE={scenario.date_start.isoformat()}~{scenario.date_end.isoformat()} | ATTACH={len(scenario.attachments)}")
    total_defects = sum(1 for row in manifest_rows if row.label == "defect")
    total_normals = sum(1 for row in manifest_rows if row.label == "normal")
    lines.append("[생성 결과 요약]")
    lines.append(f"- manifest 행 수: {len(manifest_rows)}")
    lines.append(f"- normal 이미지 수: {total_normals}")
    lines.append(f"- defect 이미지 수: {total_defects}")
    lines.append(f"- mes 행 수: {len(mes_rows)}")
    lines.extend(verification_lines)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    random.seed(RANDOM_SEED)
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"원본 데이터셋 폴더가 없습니다: {SOURCE_ROOT}")
    if not SCENARIO_PATH.exists():
        raise FileNotFoundError(f"시나리오 파일이 없습니다: {SCENARIO_PATH}")
    scenarios, min_date_value, max_date_value, cause_counts = load_scenarios(SCENARIO_PATH)
    free_space_gb = get_free_space_gb(BASE_DIR)
    mode, selected_scenarios = select_execution_scenarios(scenarios, free_space_gb)
    normal_pool, anomaly_pool = load_source_samples()
    manifest_rows = build_base_factory(normal_pool, anomaly_pool, selected_scenarios)
    manifest_rows = apply_scenarios_to_factory(manifest_rows, anomaly_pool, selected_scenarios)
    manifest_rows, contamination_lines = apply_bank_contamination(
        manifest_rows, anomaly_pool, selected_scenarios
    )
    mes_rows = build_mes_rows(manifest_rows)
    verification_lines = contamination_lines + verify_scenario_integration(
        selected_scenarios, manifest_rows, mes_rows
    )
    write_manifest_csv(manifest_rows)
    write_mes_csv(mes_rows)
    write_summary(mode, free_space_gb, scenarios, selected_scenarios, min_date_value, max_date_value, cause_counts, manifest_rows, mes_rows, verification_lines)
    print(f"mode={mode}")
    print(f"free_space_gb={free_space_gb:.2f}")
    print(f"selected_scenarios={len(selected_scenarios)}")
    print(f"manifest_rows={len(manifest_rows)}")
    print(f"mes_rows={len(mes_rows)}")
    print(f"factory_root={FACTORY_ROOT}")
    print(f"manifest_path={MANIFEST_PATH}")
    print(f"mes_path={MES_PATH}")
    print(f"summary_path={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
