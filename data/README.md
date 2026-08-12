# 🏭 Synthetic PCB Factory Vision Inspection & MES Dataset

본 디렉토리(`data/`)는 **이동현(eundong)** 담당 파트로, 아마존 **VisA (Visual Anomaly Benchmark Dataset)** 원본 이미지(`pcb1`~`pcb4`)를 기반으로 4개 공정 라인의 **가상 비전 검사 이미지 데이터베이스** 및 **통합 MES CSV 메타데이터**를 구축하는 공간입니다.

---

## 📌 목차
1. [담당자 및 개요](#1-담당자-및-개요)
2. [data/ 디렉토리 구조](#2-data-디렉토리-구조)
3. [데이터 명세 및 스키마](#3-데이터-명세-및-스키마)
4. [데이터 생성 및 판정 규칙](#4-데이터-생성-및-판정-규칙)
5. [비정상 이벤트 시나리오](#5-비정상-이벤트-시나리오)
6. [사용 방법](#6-사용-방법)

---

## 1. 담당자 및 개요

- **소유자 / 담당자**: 이동현 (eundong)
- **원천 데이터**: Amazon VisA Dataset (`pcb1`, `pcb2`, `pcb3`, `pcb4`)
- **대상 생산 라인**: Line 01 (pcb1), Line 02 (pcb2), Line 03 (pcb3), Line 04 (pcb4)
- **추출 모드**:
  - **비복원 추출 (`WITHOUT_REPLACEMENT`)**: 원본 이미지 1회 사용 (중복 0건, 약 10~11일분 약 4,200건 생산) **[기본 설정]**
  - **복원 추출 (`WITH_REPLACEMENT`)**: 원본 이미지 무작위 재사용 (30일 전체 일수 생산)

---

## 2. data/ 디렉토리 구조

```
data/
├── README.md                 # 본 데이터 담당 가이드 문서
├── build_factory.py          # 가상공장 이미지 DB 및 MES 메타데이터 생성 스크립트
├── mes.csv                   # 팀 진단 에이전트 연동용 통합 MES 메타데이터 (4,213건)
├── manifest.csv              # 이미지 경로-조건-원천 조인 매니페스트 메타데이터 (4,213건)
├── error_code_mapping.csv    # MES 불량 에러코드 ↔ VisA 원본 불량명 매핑표
├── event_scenarios_guide.md  # 해당 데이터셋 비정상 이벤트 시나리오 가이드 보고서
├── criteria.yaml             # 품질 검사 기준
├── quality_baseline.yaml     # 품질 베이스라인
├── scenarios.yaml            # 시나리오 정의
└── factory/                  # [자동 .gitignore] 881MB 가상공장 이미지 데이터베이스
```

---

## 3. 데이터 명세 및 스키마

### 3.1 파일 및 디렉토리 명명 규칙
- **이미지 폴더 경로**: `data/factory/line_{라인번호}/{YYYY-MM-DD}/{CELL_ID}.JPG`
- **셀 아이디 규격 (`CELL_ID`)**: `L{LINE_NUM}{LOT_ID}{CELL_NUM:06d}` (예: `L01AAA100001`)
  - `L01`: 라인 번호 (01, 02, 03, 04)
  - `AAA`: 라인간 중복되지 않는 전역 고유 3글자 Alphabet 랏(LOT) 번호
  - `100001`: 새로운 랏 시작 시 100001부터 순차 증가하는 6자리 셀 시퀀스

### 3.2 MES CSV 메타데이터 명세 (`data/mes.csv`)

| 컬럼명 | 데이터 타입 | 설명 | 예시 / 값 범위 |
| :--- | :--- | :--- | :--- |
| `lot_id` | String | 팀 공통 로트 키 | `LOT-20260801-AAA` |
| `line` | String | 라인 이름 | `line_01`, `line_02` |
| `object` | String | VisA 객체명 | `pcb1`, `pcb2` |
| `date` | String | 생성 일자 | `2026-08-01` |
| `started_at` | DateTime | 시작 일시 | `2026-08-01 08:00:00` |
| `LINE_NUM` | String | 라인 번호 (2자리) | `01`, `02`, `03`, `04` |
| `CELL_ID` | String | 셀 아이디 (파일명) | `L01AAA100001` |
| `PRE_PROCESS_A` | Integer | 전공정 조건 A | `1` 또는 `2` |
| `PRE_PROCESS_B` | Float | 전공정 조건 B (가우시안 분포) | `0.000` ~ `1.000` (소수점 3자리) |
| `AI_JUDGE` | String | AI 비전 검사 판정 | `OK`, `RE`, `NG` |
| `AI_ERROR_CODE` | String | AI 검사 불량 코드 | `0000` (정상) 또는 `ESA0`~`ESA8` |
| `AD_SCORE` | Float | PatchCore Anomaly Score 수치 | OK: 0.80~1.00 / RE: 1.01~1.50 / NG: 1.51~2.00 |
| `INSPECTOR_JUDGE` | String | 검사원 판정 (Ground Truth) | `OK`, `NG` |
| `INSPECTOR_ERROR_CODE`| String | 검사원 기준 불량 코드 (GT) | `0000` (정상) 또는 `ESA0`~`ESA8` |

### 3.3 에러 코드 매핑표 (`data/error_code_mapping.csv`)

| ERROR_CODE | DEFECT_NAME | DESCRIPTION |
| :--- | :--- | :--- |
| `0000` | normal | 정상품 (Normal / Good) |
| `ESA0` | bent | 휘어짐 / 굽힘 (Bent) |
| `ESA1` | burnt | 소손 / 탄화 (Burnt) |
| `ESA2` | damage | 파손 / 크랙 (Damage) |
| `ESA3` | dirt | 이물 / 오염 (Dirt) |
| `ESA4` | extra | 잉여물 / 동박 돌기 (Extra material) |
| `ESA5` | melt | 용융 / 녹음 (Melt) |
| `ESA6` | missing | 미삽 / 부품 누락 (Missing component) |
| `ESA7` | scratch | 스크래치 / 긁힘 (Scratch) |
| `ESA8` | wrong place | 오삽 / 위치 오류 (Wrong place) |

---

## 4. 데이터 생성 및 판정 규칙

1. **정상품 (Normal)**: `INSPECTOR_JUDGE` = `OK`, `AI_JUDGE` = `OK` (`AD_SCORE`: 0.80 ~ 1.00)
2. **실불량 (Real Defect)**: `INSPECTOR_JUDGE` = `NG`, `AI_JUDGE` = `RE` 또는 `NG` (`AD_SCORE`: 1.01 ~ 2.00)
3. **과검 (Overkill)**: `INSPECTOR_JUDGE` = `OK` 이지만 `AI_JUDGE` = `RE` 또는 `NG` (일일 정상품 중 **약 5% ±0.5%**)
4. **미검 (Underkill)**: `INSPECTOR_JUDGE` = `NG` 이지만 `AI_JUDGE` = `OK` (비정상 이벤트 발생 일자에 1~5건 발생)

---

## 5. 비정상 이벤트 시나리오

특정 일자/라인에서 발생하는 6가지 비정상 이벤트 시나리오가 적용되며, 상세 내역은 [event_scenarios_guide.md](event_scenarios_guide.md) 문서에 자동 기록됩니다.

---

## 6. 사용 방법

```bash
# 가상공장 이미지 DB 및 MES 메타데이터 자동 생성
python3 data/build_factory.py
```
