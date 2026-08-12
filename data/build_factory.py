#!/usr/bin/env python3
"""
===============================================================================
[SK AI Hackathon] 가상 공장 검사 이미지 데이터베이스 & MES/Manifest 데이터 생성기
담당자: 이동현 (eundong)
Base Dataset: Amazon VisA Dataset (pcb1, pcb2, pcb3, pcb4)

설명:
  본 스크립트는 아마존 VisA 데이터셋을 기반으로 가상의 4개 공정 라인의 이미지 데이터베이스
  (data/factory/ 및 pcb_factory_dataset/), 통합 MES 데이터(data/mes.csv),
  이미지 매니페스트(data/manifest.csv), 불량 에러코드 매핑표(data/error_code_mapping.csv),
  비정상 이벤트 설명서(docs/event_scenarios_guide.md)를 'LLM 연동 없이 순수 파이썬 코드'로
  자동 생성합니다.

사용 방법:
  python3 data/build_factory.py
===============================================================================
"""

import os
import sys
import shutil
import random
import datetime
import numpy as np
import pandas as pd

# ==============================================================================
# 1. 사용자 설정 옵션 (CONFIG)
# ==============================================================================
CONFIG = {
    # --------------------------------------------------------------------------
    # [추출 방식 설정]
    # "WITHOUT_REPLACEMENT": 비복원 추출 (원본 이미지 중복 사용 안 함 - 약 10~11일분 약 4,200건 생산)
    # "WITH_REPLACEMENT"   : 복원 추출 (원본 이미지 재사용/중복 허용 - 지정 일수 전체 생산)
    # --------------------------------------------------------------------------
    "SAMPLING_MODE": "WITHOUT_REPLACEMENT",

    # --------------------------------------------------------------------------
    # [디렉토리 및 파일 경로 설정]
    # --------------------------------------------------------------------------
    "ROOT_DIR": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "ARCHIVE_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive"),
    "FACTORY_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "factory"),
    "OUTPUT_DIR": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pcb_factory_dataset"),

    "MES_CSV_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "mes.csv"),
    "ROOT_MES_CSV_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mes_data.csv"),
    "MANIFEST_CSV_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "manifest.csv"),
    "ERROR_MAPPING_CSV_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "error_code_mapping.csv"),
    "EVENT_GUIDE_MD_PATH": os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "event_scenarios_guide.md"),

    # --------------------------------------------------------------------------
    # [라인별 VisA 데이터셋 카테고리 매핑]
    # --------------------------------------------------------------------------
    "LINE_CATEGORY_MAP": {
        "01": "pcb1",
        "02": "pcb2",
        "03": "pcb3",
        "04": "pcb4",
    },

    # --------------------------------------------------------------------------
    # [생산 일정 및 수량 설정]
    # --------------------------------------------------------------------------
    "START_DATE": "2026-08-01",      # 디폴트 시작일: 2026년 8월 1일
    "NUM_DAYS": 30,                 # 최대 목표 일수 (비복원 추출시 이미지 소진 시 자동 조기 종료)
    "DAILY_BASE_QTY_PER_LINE": 100, # 일일 라인당 평균 목표 생산량 (~100개/일)
    "DAILY_QTY_STD": 10,            # 일일 생산량의 가우시안 변동폭 (표준편차)
    "LOT_CELL_CAPACITY": 200,       # 랏(LOT) 당 셀 수량 (200개 도달 시 AAA -> AAB 로 랏 교체)

    # --------------------------------------------------------------------------
    # [불량률 및 과검/미검 설정]
    # --------------------------------------------------------------------------
    "BASE_DEFECT_RATE": 0.08,       # 기본 실불량률 (~8%)
    "NORMAL_OVERKILL_RATE": 0.05,   # 정상품 중 과검(가성불량) 발생 비율 (~5%)
    "NORMAL_OVERKILL_STD": 0.005,   # 과검 비율 일별 변동폭 (±0.5%)

    # --------------------------------------------------------------------------
    # [AI Anomaly Score (PatchCore) 판정 기준 구간 설정]
    # --------------------------------------------------------------------------
    "RE_NG_RATIO": 0.9,             # AI가 불량 판정 시 RE(재검사) 대 NG(폐기) 비율 (9:1)
    "AD_SCORE_PARAMS": {
        "OK": {"mean": 0.84, "std": 0.04, "min": 0.80, "max": 1.00},
        "RE": {"mean": 1.25, "std": 0.10, "min": 1.01, "max": 1.50},
        "NG": {"mean": 1.80, "std": 0.10, "min": 1.51, "max": 2.00},
    },

    # --------------------------------------------------------------------------
    # [VisA 원본 불량 라벨 ↔ MES 에러코드 매핑]
    # --------------------------------------------------------------------------
    "DEFECT_CODE_MAP": {
        "bent": {"code": "ESA0", "desc": "휘어짐 / 굽힘 (Bent)"},
        "burnt": {"code": "ESA1", "desc": "소손 / 탄화 (Burnt)"},
        "damage": {"code": "ESA2", "desc": "파손 / 크랙 (Damage)"},
        "dirt": {"code": "ESA3", "desc": "이물 / 오염 (Dirt)"},
        "extra": {"code": "ESA4", "desc": "잉여물 / 동박 돌기 (Extra material)"},
        "melt": {"code": "ESA5", "desc": "용융 / 녹음 (Melt)"},
        "missing": {"code": "ESA6", "desc": "미삽 / 부품 누락 (Missing component)"},
        "scratch": {"code": "ESA7", "desc": "스크래치 / 긁힘 (Scratch)"},
        "wrong place": {"code": "ESA8", "desc": "오삽 / 위치 오류 (Wrong place)"},
    },
    "NORMAL_ERROR_CODE": "0000",
    "NORMAL_ERROR_DESC": "정상품 (Normal / Good)",

    # --------------------------------------------------------------------------
    # [전공정 변수 (PRE_PROCESS_A, PRE_PROCESS_B) 설정]
    # --------------------------------------------------------------------------
    "PRE_PROCESS_A_RATIO": 0.5,
    "PRE_PROCESS_B_PARAMS": {"mean": 0.500, "std": 0.150, "min": 0.000, "max": 1.000},

    # --------------------------------------------------------------------------
    # [비정상 이벤트 캘린더 설정]
    # --------------------------------------------------------------------------
    "ABNORMAL_EVENTS": {
        ("2026-08-05", "01"): "OVERKILL_A2_REAL",
        ("2026-08-07", "02"): "OVERKILL_A1_FALSE",
        ("2026-08-09", "03"): "OVERKILL_B_OUTLIER",
        ("2026-08-10", "04"): "UNDERKILL_EVENT",
        ("2026-08-03", "01"): "LINE_SHUTDOWN",
        ("2026-08-08", "02"): "LINE_SURGE",
    },

    "EVENT_DESCRIPTIONS": {
        "OVERKILL_A2_REAL": {
            "title": "이전공정 문제로 인한 실불량 집중 (PRE_PROCESS_A = 2)",
            "details": "전공정 A의 조건이 2일 때 실불량(INSPECTOR_JUDGE=NG) 발생 비율이 35% 이상으로 비정상적으로 높게 발생하는 비정상 이벤트."
        },
        "OVERKILL_A1_FALSE": {
            "title": "이전공정 문제로 인한 과검(가성불량) 폭증 (PRE_PROCESS_A = 1)",
            "details": "전공정 A의 조건이 1일 때 정상품(INSPECTOR_JUDGE=OK)임에도 AI가 과검(AI_JUDGE=RE/NG)으로 판정하는 비율이 10~20% 내외로 폭증하는 이벤트."
        },
        "OVERKILL_B_OUTLIER": {
            "title": "이전공정 B 가우시안 분포 이탈 이상치 발생",
            "details": "PRE_PROCESS_B 수치가 정상적인 가우시안 분포(0.0~1.0)의 중심을 이탈하여 0.1 이하 극소치 또는 0.9 이상 극대치 이상치가 35% 이상 대량 발생하는 이벤트."
        },
        "UNDERKILL_EVENT": {
            "title": "미검(Escape/Underkill) 발생 이벤트",
            "details": "실제 불량품(INSPECTOR_JUDGE=NG)임에도 불구하고 AI가 정상품(AI_JUDGE=OK)으로 잘못 오판정하는 미검이 일일 1~5건 발생하는 위험 이벤트."
        },
        "LINE_SHUTDOWN": {
            "title": "라인 정지 (0건 생산)",
            "details": "설비 점검 또는 이상 발생으로 해당 라인의 일일 생산량이 0건(생산 중단)인 이벤트."
        },
        "LINE_SURGE": {
            "title": "라인 일시적 생산량 급증",
            "details": "특정 일자에 해당 라인의 생산량이 평소의 약 1.8배로 급격히 증가하는 이벤트."
        }
    },

    "RANDOM_SEED": 42
}


# ==============================================================================
# 2. 보조 클래스 및 함수 정의
# ==============================================================================

class GlobalLotManager:
    """글로벌 랏 번호(AAA, AAB, AAC...) 생성기"""
    def __init__(self, start_idx=0):
        self.current_idx = start_idx

    def _idx_to_str(self, idx):
        c1 = chr(ord('A') + (idx // (26 * 26)) % 26)
        c2 = chr(ord('A') + (idx // 26) % 26)
        c3 = chr(ord('A') + idx % 26)
        return f"{c1}{c2}{c3}"

    def next_lot(self):
        lot_str = self._idx_to_str(self.current_idx)
        self.current_idx += 1
        return lot_str


def parse_visa_annotations(archive_dir, line_category_map):
    """VisA 원본 데이터셋 정보 파싱"""
    cat_data = {}
    for line_num, cat_name in line_category_map.items():
        cat_dir = os.path.join(archive_dir, cat_name)
        csv_path = os.path.join(cat_dir, "image_anno.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"VisA 메타데이터 파일을 찾을 수 없습니다: {csv_path}")

        df = pd.read_csv(csv_path)
        normals = []
        anomalies = []

        for _, row in df.iterrows():
            rel_img = str(row['image']).strip()
            label = str(row['label']).strip()
            mask_rel = str(row.get('mask', '')).strip() if pd.notna(row.get('mask')) else ''
            full_img_path = os.path.join(archive_dir, rel_img)
            full_mask_path = os.path.join(archive_dir, mask_rel) if mask_rel else ''

            if label == 'normal':
                normals.append((full_img_path, 'normal', full_mask_path, rel_img))
            else:
                anomalies.append((full_img_path, label, full_mask_path, rel_img))

        random.shuffle(normals)
        random.shuffle(anomalies)

        cat_data[cat_name] = {
            "normal": normals,
            "anomaly": anomalies
        }
        print(f"[{cat_name}] 로드 완료 (Line {line_num}): 정상 {len(normals)}장, 불량 {len(anomalies)}장")

    return cat_data


def get_defect_error_code(gt_label, defect_map):
    """Ground Truth 불량 라벨 ↔ 에러코드 매핑"""
    if not gt_label or gt_label == 'normal':
        return CONFIG["NORMAL_ERROR_CODE"]

    tags = [t.strip() for t in str(gt_label).split(',')]
    for tag in tags:
        if tag in defect_map:
            return defect_map[tag]["code"]
    return "ESA9"


def generate_ad_score(judge):
    """Anomaly Score 생성"""
    params = CONFIG["AD_SCORE_PARAMS"].get(judge, CONFIG["AD_SCORE_PARAMS"]["OK"])
    score = np.random.normal(params["mean"], params["std"])
    score = np.clip(score, params["min"], params["max"])
    return round(float(score), 2)


def export_error_code_mapping(path):
    """error_code_mapping.csv 자동 생성"""
    rows = [
        {"ERROR_CODE": CONFIG["NORMAL_ERROR_CODE"], "DEFECT_NAME": "normal", "DESCRIPTION": CONFIG["NORMAL_ERROR_DESC"]}
    ]
    for tag, info in CONFIG["DEFECT_CODE_MAP"].items():
        rows.append({
            "ERROR_CODE": info["code"],
            "DEFECT_NAME": tag,
            "DESCRIPTION": info["desc"]
        })
    df_map = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_map.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"매핑 CSV 파일 생성 완료: {path}")


def generate_event_scenarios_md(path, mes_df):
    """event_scenarios_guide.md 자동 작성"""
    md_content = []
    md_content.append("# MES Factory Dataset - 비정상 이벤트 시나리오 가이드\n")
    md_content.append("본 문서는 `pcb_factory_dataset` 및 `data/` 생성 시 적용된 비정상 이벤트 시나리오 조건 및 실제 발생 통계 요약입니다.\n")
    md_content.append("## 1. 이벤트 개요 목록\n")
    md_content.append("| 발생 일자 | 적용 라인 | 이벤트 코드 | 시나리오 제목 |")
    md_content.append("| :--- | :--- | :--- | :--- |")

    for (date_str, line_num), event_type in sorted(CONFIG["ABNORMAL_EVENTS"].items()):
        info = CONFIG["EVENT_DESCRIPTIONS"].get(event_type, {"title": event_type, "details": ""})
        md_content.append(f"| `{date_str}` | Line `{line_num}` | `{event_type}` | {info['title']} |")

    md_content.append("\n---\n")
    md_content.append("## 2. 상세 시나리오 조건 및 실제 관측 통계\n")

    for (date_str, line_num), event_type in sorted(CONFIG["ABNORMAL_EVENTS"].items()):
        info = CONFIG["EVENT_DESCRIPTIONS"].get(event_type, {"title": event_type, "details": ""})
        md_content.append(f"### 📍 [`{date_str}`] Line `{line_num}` - {info['title']}\n")
        md_content.append(f"- **이벤트 코드**: `{event_type}`")
        md_content.append(f"- **조건 설명**: {info['details']}\n")

        day_line_df = mes_df[(mes_df['LINE_NUM'] == line_num) & (mes_df['START_TIME'].str.startswith(date_str))]

        if len(day_line_df) == 0:
            md_content.append("- **실제 결과**: 생산 0건 (라인 정지 확인 또는 생산분 소진)\n")
        else:
            total_cnt = len(day_line_df)
            insp_ok = (day_line_df['INSPECTOR_JUDGE'] == 'OK').sum()
            insp_ng = (day_line_df['INSPECTOR_JUDGE'] == 'NG').sum()

            ai_ok = (day_line_df['AI_JUDGE'] == 'OK').sum()
            ai_re = (day_line_df['AI_JUDGE'] == 'RE').sum()
            ai_ng = (day_line_df['AI_JUDGE'] == 'NG').sum()

            overkill_cnt = ((day_line_df['INSPECTOR_JUDGE'] == 'OK') & (day_line_df['AI_JUDGE'].isin(['RE', 'NG']))).sum()
            underkill_cnt = ((day_line_df['INSPECTOR_JUDGE'] == 'NG') & (day_line_df['AI_JUDGE'] == 'OK')).sum()

            pre_a_counts = day_line_df['PRE_PROCESS_A'].value_counts().to_dict()

            md_content.append("**실제 관측 통계:**")
            md_content.append(f"- 총 생산 수량: `{total_cnt}` 건")
            md_content.append(f"- 검사원 판정 (Ground Truth): OK `{insp_ok}` 건 / NG `{insp_ng}` 건")
            md_content.append(f"- AI 판정: OK `{ai_ok}` 건 / RE `{ai_re}` 건 / NG `{ai_ng}` 건")
            md_content.append(f"- 과검 (Overkill): `{overkill_cnt}` 건 (OK 중 비율: `{overkill_cnt/max(1,insp_ok):.1%}`)")
            md_content.append(f"- 미검 (Underkill): `{underkill_cnt}` 건")
            md_content.append(f"- PRE_PROCESS_A 분포: `1`: {pre_a_counts.get(1, 0)}건, `2`: {pre_a_counts.get(2, 0)}건\n")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"이벤트 가이드 MD 파일 생성 완료: {path}")


# ==============================================================================
# 3. 메인 실행 로직
# ==============================================================================

def main():
    sampling_mode = CONFIG.get("SAMPLING_MODE", "WITHOUT_REPLACEMENT")
    print("=" * 70)
    print(f"가상 공장 MES 데이터셋 및 이미지 DB 생성 시작 (추출 방식: {sampling_mode})")
    print("=" * 70)

    random.seed(CONFIG["RANDOM_SEED"])
    np.random.seed(CONFIG["RANDOM_SEED"])

    visa_db = parse_visa_annotations(CONFIG["ARCHIVE_DIR"], CONFIG["LINE_CATEGORY_MAP"])

    is_without_replacement = (sampling_mode == "WITHOUT_REPLACEMENT")
    
    if is_without_replacement:
        pools = {}
        for cat_name, data in visa_db.items():
            pools[cat_name] = {
                "normal": list(data["normal"]),
                "anomaly": list(data["anomaly"])
            }

    lot_mgr = GlobalLotManager(start_idx=0)

    line_state = {}
    line_active = {}
    for line_num in CONFIG["LINE_CATEGORY_MAP"].keys():
        line_state[line_num] = {
            "current_lot": lot_mgr.next_lot(),
            "cell_in_lot": 0,
            "cell_seq": 100001
        }
        line_active[line_num] = True

    mes_records = []
    manifest_records = []
    total_images_copied = 0

    start_dt = datetime.datetime.strptime(CONFIG["START_DATE"], "%Y-%m-%d")
    dates = [start_dt + datetime.timedelta(days=i) for i in range(CONFIG["NUM_DAYS"])]

    # 기존 타겟 디렉토리 초기화
    for target_dir in [CONFIG["FACTORY_DIR"], CONFIG["OUTPUT_DIR"]]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

    for date_obj in dates:
        date_str = date_obj.strftime("%Y-%m-%d")
        yyyymmdd = date_obj.strftime("%Y%m%d")

        if not any(line_active.values()):
            print(f"[{date_str}] 모든 라인의 원본 이미지가 소진되어 생산을 종료합니다.")
            break

        for line_num, cat_name in CONFIG["LINE_CATEGORY_MAP"].items():
            if not line_active[line_num]:
                continue

            event_type = CONFIG["ABNORMAL_EVENTS"].get((date_str, line_num), "NORMAL")

            if event_type == "LINE_SHUTDOWN":
                print(f"[{date_str}] Line {line_num}: LINE_SHUTDOWN 이벤트 (0건 생산)")
                continue

            if event_type == "LINE_SURGE":
                daily_qty = int(CONFIG["DAILY_BASE_QTY_PER_LINE"] * 1.8)
            else:
                daily_qty = int(np.random.normal(CONFIG["DAILY_BASE_QTY_PER_LINE"], CONFIG["DAILY_QTY_STD"]))
                daily_qty = max(10, daily_qty)

            if event_type == "OVERKILL_A1_FALSE":
                daily_overkill_rate = np.random.uniform(0.12, 0.18)
            else:
                daily_overkill_rate = max(0.01, np.random.normal(CONFIG["NORMAL_OVERKILL_RATE"], CONFIG["NORMAL_OVERKILL_STD"]))

            underkill_remaining = 0
            if event_type == "UNDERKILL_EVENT":
                underkill_remaining = random.randint(1, 5)

            current_timestamp = date_obj.replace(hour=8, minute=0, second=0)

            line_folder_name = f"L{line_num}"
            day_folder_factory = os.path.join(CONFIG["FACTORY_DIR"], f"line_{line_num}", date_str)
            day_folder_output = os.path.join(CONFIG["OUTPUT_DIR"], line_folder_name, yyyymmdd)

            cat_normals = visa_db[cat_name]["normal"]
            cat_anomalies = visa_db[cat_name]["anomaly"]

            for item_idx in range(daily_qty):
                st_state = line_state[line_num]

                if st_state["cell_in_lot"] >= CONFIG["LOT_CELL_CAPACITY"]:
                    st_state["current_lot"] = lot_mgr.next_lot()
                    st_state["cell_in_lot"] = 0
                    st_state["cell_seq"] = 100001

                cell_id = f"L{line_num}{st_state['current_lot']}{st_state['cell_seq']}"
                lot_id_full = f"LOT-{yyyymmdd}-{st_state['current_lot']}"

                start_time_str = current_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                duration_sec = 120 + random.randint(-5, 5)
                end_timestamp = current_timestamp + datetime.timedelta(seconds=duration_sec)
                end_time_str = end_timestamp.strftime("%Y-%m-%d %H:%M:%S")
                current_timestamp += datetime.timedelta(seconds=random.randint(20, 40))

                if event_type == "OVERKILL_A2_REAL":
                    pre_a = 2 if random.random() < 0.7 else 1
                elif event_type == "OVERKILL_A1_FALSE":
                    pre_a = 1 if random.random() < 0.7 else 2
                else:
                    pre_a = 1 if random.random() < CONFIG["PRE_PROCESS_A_RATIO"] else 2

                if event_type == "OVERKILL_B_OUTLIER" and random.random() < 0.35:
                    pre_b = round(random.choice([np.random.uniform(0.001, 0.099), np.random.uniform(0.901, 0.999)]), 3)
                else:
                    pb_params = CONFIG["PRE_PROCESS_B_PARAMS"]
                    pre_b_val = np.random.normal(pb_params["mean"], pb_params["std"])
                    pre_b = round(float(np.clip(pre_b_val, pb_params["min"], pb_params["max"])), 3)

                is_gt_defect = False
                if event_type == "OVERKILL_A2_REAL" and pre_a == 2:
                    is_gt_defect = random.random() < 0.35
                else:
                    is_gt_defect = random.random() < CONFIG["BASE_DEFECT_RATE"]

                if is_gt_defect:
                    inspector_judge = "NG"
                    if is_without_replacement:
                        if len(pools[cat_name]["anomaly"]) == 0:
                            print(f"[{date_str}] Line {line_num} 불량 이미지 소진으로 생산 중단")
                            line_active[line_num] = False
                            break
                        src_img_path, gt_label, mask_path, visa_source = pools[cat_name]["anomaly"].pop()
                    else:
                        src_img_path, gt_label, mask_path, visa_source = random.choice(cat_anomalies)
                    inspector_error_code = get_defect_error_code(gt_label, CONFIG["DEFECT_CODE_MAP"])
                else:
                    inspector_judge = "OK"
                    if is_without_replacement:
                        if len(pools[cat_name]["normal"]) == 0:
                            print(f"[{date_str}] Line {line_num} 정상 이미지 소진으로 생산 중단")
                            line_active[line_num] = False
                            break
                        src_img_path, gt_label, mask_path, visa_source = pools[cat_name]["normal"].pop()
                    else:
                        src_img_path, gt_label, mask_path, visa_source = random.choice(cat_normals)
                    gt_label = "normal"
                    inspector_error_code = CONFIG["NORMAL_ERROR_CODE"]

                st_state["cell_in_lot"] += 1
                st_state["cell_seq"] += 1

                if inspector_judge == "NG" and underkill_remaining > 0:
                    ai_judge = "OK"
                    ai_error_code = CONFIG["NORMAL_ERROR_CODE"]
                    ad_score = generate_ad_score("OK")
                    underkill_remaining -= 1
                elif inspector_judge == "OK":
                    is_overkill = False
                    if event_type == "OVERKILL_A1_FALSE" and pre_a == 1:
                        is_overkill = random.random() < 0.25
                    else:
                        is_overkill = random.random() < daily_overkill_rate

                    if is_overkill:
                        is_re = random.random() < CONFIG["RE_NG_RATIO"]
                        ai_judge = "RE" if is_re else "NG"
                        ad_score = generate_ad_score(ai_judge)
                        ai_error_code = random.choice([v["code"] for v in CONFIG["DEFECT_CODE_MAP"].values()])
                    else:
                        ai_judge = "OK"
                        ai_error_code = CONFIG["NORMAL_ERROR_CODE"]
                        ad_score = generate_ad_score("OK")
                else:
                    is_re = random.random() < CONFIG["RE_NG_RATIO"]
                    ai_judge = "RE" if is_re else "NG"
                    ad_score = generate_ad_score(ai_judge)
                    if random.random() < 0.95:
                        ai_error_code = inspector_error_code
                    else:
                        ai_error_code = random.choice([v["code"] for v in CONFIG["DEFECT_CODE_MAP"].values()])

                # 복사 대상 폴더 생성 및 복사
                os.makedirs(day_folder_output, exist_ok=True)
                os.makedirs(day_folder_factory, exist_ok=True)
                _, img_ext = os.path.splitext(src_img_path)
                dest_img_name = f"{cell_id}{img_ext}"

                dest_img_output = os.path.join(day_folder_output, dest_img_name)
                dest_img_factory = os.path.join(day_folder_factory, dest_img_name)

                shutil.copy2(src_img_path, dest_img_output)
                shutil.copy2(src_img_path, dest_img_factory)
                total_images_copied += 1

                rel_factory_img = os.path.relpath(dest_img_factory, CONFIG["FACTORY_DIR"])

                # MES 레코드 수집
                mes_records.append({
                    "lot_id": lot_id_full,
                    "line": f"line_{line_num}",
                    "object": cat_name,
                    "date": date_str,
                    "started_at": start_time_str,
                    "equipment_id": f"EQ-{line_num}-A",
                    "inspected_count": 1,
                    "defect_count": 1 if inspector_judge == "NG" else 0,
                    "operator_shift": "day",
                    "LINE_NUM": line_num,
                    "CELL_ID": cell_id,
                    "START_TIME": start_time_str,
                    "END_TIME": end_time_str,
                    "PRE_PROCESS_A": pre_a,
                    "PRE_PROCESS_B": pre_b,
                    "AI_JUDGE": ai_judge,
                    "AI_ERROR_CODE": ai_error_code,
                    "AD_SCORE": ad_score,
                    "INSPECTOR_JUDGE": inspector_judge,
                    "INSPECTOR_ERROR_CODE": inspector_error_code
                })

                # Manifest 레코드 수집
                manifest_records.append({
                    "image_path": rel_factory_img,
                    "line": f"line_{line_num}",
                    "object": cat_name,
                    "date": date_str,
                    "lot_id": lot_id_full,
                    "equipment_id": f"EQ-{line_num}-A",
                    "split": "operation",
                    "label": "defect" if inspector_judge == "NG" else "normal",
                    "mask_path": mask_path,
                    "visa_source": visa_source
                })

    mes_df = pd.DataFrame(mes_records)
    manifest_df = pd.DataFrame(manifest_records)

    # 내보내기
    mes_df.to_csv(CONFIG["MES_CSV_PATH"], index=False, encoding="utf-8-sig")
    mes_df.to_csv(CONFIG["ROOT_MES_CSV_PATH"], index=False, encoding="utf-8-sig")
    manifest_df.to_csv(CONFIG["MANIFEST_CSV_PATH"], index=False, encoding="utf-8-sig")

    export_error_code_mapping(CONFIG["ERROR_MAPPING_CSV_PATH"])
    generate_event_scenarios_md(CONFIG["EVENT_GUIDE_MD_PATH"], mes_df)

    print("=" * 70)
    print("생성 작업 완료!")
    print(f"추출 모드: {sampling_mode}")
    print(f"총 MES 데이터 건수: {len(mes_df):,} 건")
    print(f"생성된 Manifest 건수: {len(manifest_df):,} 건")
    print(f"메타데이터 저장 위치: {CONFIG['MES_CSV_PATH']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
