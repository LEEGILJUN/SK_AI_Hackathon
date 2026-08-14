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

ERROR_CODE_BY_DEFECT_NAME = {
    "normal": "0000",
    "bent": "ESA0",
    "burnt": "ESA1",
    "damage": "ESA2",
    "dirt": "ESA3",
    "extra": "ESA4",
    "melt": "ESA5",
    "missing": "ESA6",
    "scratch": "ESA7",
    "wrong place": "ESA8",
}

ISSUE_KEYWORD_TO_DEFECT = {
    "크랙": "damage",
    "파손": "damage",
    "이물": "dirt",
    "오염": "dirt",
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
    "표면": "dirt",
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
        scenario = ScenarioInfo(
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


def pick_split(target_date: date, line: str, index_within_day: int) -> str:
    selector = make_seed(target_date.isoformat(), line, str(index_within_day), "split") % 100
    if selector < 70:
        return "bank"
    if selector < 90:
        return "operation"
    return "holdout"


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
                    split=pick_split(date.fromisoformat(date_str), line, image_index),
                    label=label,
                    mask_path=mask_path,
                    visa_source=sample.image_rel,
                    defect_name=defect_name,
                    ercd=ERROR_CODE_BY_DEFECT_NAME.get(defect_name, "0000"),
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
                split=pick_split(date.fromisoformat(date_str), line, image_index),
                label="defect",
                mask_path=sample.mask_rel if sample.mask_rel else "",
                visa_source=sample.image_rel,
                defect_name=sample.primary_defect,
                ercd=ERROR_CODE_BY_DEFECT_NAME.get(sample.primary_defect, "0000"),
            )
            manifest_by_path[attachment_path] = replacement_row
            copy_tasks.append(build_copy_task(sample.image_rel, FACTORY_ROOT / attachment))
    execute_copy_tasks(copy_tasks)
    return [manifest_by_path[path] for path in sorted(manifest_by_path)]


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
    current_time = started_at_value.time()
    if time(0, 0, 0) <= current_time < time(8, 0, 0):
        return "day"
    if time(8, 0, 0) <= current_time < time(16, 0, 0):
        return "swing"
    return "night"


def build_mes_rows(manifest_rows: List[ManifestRow]) -> List[Dict[str, str]]:
    sorted_rows = sorted(manifest_rows, key=lambda row: (row.line, row.date_str, row.lot_id, row.image_path))
    daily_counters = defaultdict(lambda: {"inspected": 1, "defect": 1})
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
        counters = daily_counters[(row.line, row.date_str)]
        inspected_count_value = counters["inspected"]
        if row.label == "defect":
            defect_count_value = counters["defect"]
            counters["defect"] += 1
        else:
            defect_count_value = 0
        counters["inspected"] += 1
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
    mes_rows = build_mes_rows(manifest_rows)
    verification_lines = verify_scenario_integration(selected_scenarios, manifest_rows, mes_rows)
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
