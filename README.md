# 🏭 Synthetic PCB Factory Vision Inspection & MES Data Generator

본 프로젝트는 아마존의 **VisA (Visual Anomaly Benchmark Dataset)** 원본 이미지 데이터셋(`pcb1`, `pcb2`, `pcb3`, `pcb4`)을 기반으로 가상의 4개 공정 라인에서 발생하는 **비전 검사 이미지 데이터베이스** 및 **통합 MES(Manufacturing Execution System) CSV 메타데이터**를 생성하는 자동화 데이터 생성 시스템입니다.

---

## 📌 목차
1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [데이터 규격 및 명세](#3-데이터-규격-및-명세)
4. [데이터 생성 및 판정 규칙](#4-데이터-생성-및-판정-규칙)
5. [비정상 이벤트 시나리오](#5-비정상-이벤트-시나리오)
6. [사용 방법 및 설정 변경 가이드](#6-사용-방법-및-설정-변경-가이드)
7. [데이터 검증 시스템](#7-데이터-검증-시스템)

---

## 1. 프로젝트 개요

실제 스마트 팩토리의 비전 검사 환경에서 발생하는 공정 데이터, AI 알고리즘 판정 수치(PatchCore Anomaly Score), 검사원 Ground Truth, 과검/미검 비율, 비정상 공정 이상 이벤트를 현실감 있게 모사한 합성 데이터셋을 구축합니다.

- **담당자**: 이동현 (eundong)
- **원천 데이터**: Amazon VisA Dataset (`pcb1`, `pcb2`, `pcb3`, `pcb4`)
- **대상 생산 라인**: Line 01 (pcb1), Line 02 (pcb2), Line 03 (pcb3), Line 04 (pcb4)
- **추출 모드**:
  - **비복원 추출 (`WITHOUT_REPLACEMENT`)**: 원본 이미지 1회 사용 (중복 0건, 약 10~11일분 약 4,200건 생산) **[기본 설정]**
  - **복원 추출 (`WITH_REPLACEMENT`)**: 원본 이미지 무작위 재사용 (30일 전체 일수 생산)

---

## 2. 디렉토리 구조

```
[Shared]SK_AI_Hackathon/
├── README.md                     # 프로젝트 종합 가이드 문서
├── mes_data.csv                  # 루트 경로 메타데이터 복사본
├── data/                         # 가상공장 데이터 및 스크립트 메인 폴더
│   ├── build_factory.py          # 파이썬 가상공장 데이터셋 & MES 생성기
│   ├── mes.csv                   # 팀 표준 MES 메타데이터 CSV
│   ├── manifest.csv              # 이미지-조건 조인 키 매니페스트 CSV
│   ├── error_code_mapping.csv    # MES 불량 코드 ↔ VisA 원본 불량명 1:1 매핑표
│   └── factory/                  # [자동 .gitignore] 이미지 데이터베이스
├── docs/
│   └── event_scenarios_guide.md  # 비정상 이벤트 조건 및 관측 통계 마크다운 보고서
└── archive/                      # [자동 .gitignore] VisA 원본 데이터셋 폴더
```

---

## 3. 데이터 규격 및 명세

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

---

## 4. 데이터 생성 및 판정 규칙

1. **정상품 (Normal)**: `INSPECTOR_JUDGE` = `OK`, `AI_JUDGE` = `OK` (`AD_SCORE`: 0.80 ~ 1.00)
2. **실불량 (Real Defect)**: `INSPECTOR_JUDGE` = `NG`, `AI_JUDGE` = `RE` 또는 `NG` (`AD_SCORE`: 1.01 ~ 2.00)
3. **과검 (Overkill)**: `INSPECTOR_JUDGE` = `OK` 이지만 `AI_JUDGE` = `RE` 또는 `NG` (일일 정상품 중 **약 5% ±0.5%**)
4. **미검 (Underkill)**: `INSPECTOR_JUDGE` = `NG` 이지만 `AI_JUDGE` = `OK` (비정상 이벤트 발생 일자에 1~5건 발생)

---

## 5. 비정상 이벤트 시나리오

특정 일자/라인에서 발생하는 6가지 비정상 이벤트 시나리오가 적용되며, 상세 내역은 [docs/event_scenarios_guide.md](docs/event_scenarios_guide.md) 문서에 자동 기록됩니다.

---

## 6. 사용 방법

```bash
# 가상공장 이미지 DB 및 MES 메타데이터 자동 생성
python3 data/build_factory.py
```
