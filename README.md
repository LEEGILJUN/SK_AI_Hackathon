# 검사 AI 자율 운영 에이전트

**못 잡는 불량을 스스로 학습해 배포까지 준비하는 Self-Healing Vision Ops**

2026 SK AI 해커톤 · AI Solution 리그

---

## 이게 무엇인가

제조 현장에 배포된 AI 외관검사 모델이 "못 잡는 불량"을 만났을 때, 원인 진단부터 데이터 선별, 뱅크 재구성, 성능 검증, 배포 패키지 생성까지를 에이전트가 스스로 수행합니다.

검사 모델을 새로 만드는 프로젝트가 아니라, **이미 배포된 모델을 계속 살아 있게 유지하는 운영의 문제**를 다룹니다.

기준 모델은 PatchCore입니다. 정상 패치를 메모리 뱅크에 담아두고 최근접 거리로 판정하는 구조라, 미검출이 발생했을 때 그 이미지가 어떤 정상 패치와 가까웠는지를 되짚을 수 있습니다. 진단 에이전트는 이 경로를 따라가 원인을 규명합니다.

## 문서

| 문서 | 내용 |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | 프로젝트 컨텍스트와 작업 원칙. 코드 작업 전 필독 |
| [`docs/작업목록.md`](docs/작업목록.md) | 원인 분류 체계, 역할 분담, 작업 목록, 저장소 규칙, 정량 목표 |
| [`docs/기획서.md`](docs/기획서.md) | 예선 제출 기획서. 설계 근거가 필요할 때 |
| [`docs/전체아키텍처.png`](docs/전체아키텍처.png) | 구성 요소 간 관계 |
| [`docs/서비스개념도.png`](docs/서비스개념도.png) | As-Is / To-Be 흐름 |

## 팀

| 이름 | 담당 |
|---|---|
| 이길준 | LLM·VLM 실행 환경, 에이전트 오케스트레이션, PatchCore 추론과 최근접 패치 역추적, 진단 판정 로직, 데모 환경 |
| 장영진 | 원인 분류 체계, 시나리오와 정답 라벨, 화질 지표와 판정 기준, 진단 결과 채점, 업무 프로세스 표준화 |
| 이동현 | 가상 공장 스토리지와 MES 데이터, 학습 이력 인덱서, 이슈 이력 그래프와 검색, 조회 계층, 스케줄러 |

## 개발 환경 준비

```bash
git clone https://github.com/LEEGILJUN/SK_AI_Hackathon.git
cd SK_AI_Hackathon
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# VisA 데이터셋을 내려받아 가상 공장 폴더트리를 생성한다
python data/build_factory.py
```

`data/build_factory.py`는 시드를 고정해 동작하므로, 각자 실행하면 동일한 구조가 만들어집니다. **VisA 원본 이미지와 메모리 뱅크 파일은 저장소에 포함되지 않습니다.**

## 저장소 규칙

- 대용량 바이너리(VisA 원본, `.pt`/`.npz` 뱅크, 실행 로그)는 커밋하지 않습니다. 스크립트로 재현합니다.
- `data/scenarios.yaml`은 **정답 파일**입니다. 진단 정확도를 맞추기 위해 수정하지 않습니다. 변경이 필요하면 합의 후 진행합니다.
- 대회 기간 중에는 저장소를 비공개로 유지합니다.

자세한 내용은 [`docs/작업목록.md`](docs/작업목록.md) 7장을 참고하세요.

## 데이터 출처 및 라이선스

본 프로젝트의 시연 데이터는 **VisA (Visual Anomaly) dataset**을 사용합니다.

> Zou, Y., Jeong, J., Pemula, L., Zhang, D., Dabeer, O. *SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation.* ECCV 2022. Amazon Science.
> 라이선스: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
> 저장소: https://github.com/amazon-science/spot-diff

**사내 데이터는 일절 사용하지 않습니다.** 모든 생산 데이터, MES 정보, 이슈 이력은 공개 데이터셋과 가상 생성 데이터로 구성됩니다.

사용한 오픈소스 라이브러리의 라이선스는 [`THIRD_PARTY.md`](THIRD_PARTY.md)에 정리합니다.
