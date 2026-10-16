# Manufacturing AI Portfolio

반도체 제조 데이터로 불량을 예측하고 해석하는 프로젝트 모음입니다.
모델 점수보다 **실제 생산 라인에서 믿을 수 있는 평가**를 설계하는 데 집중했습니다.

| 프로젝트 | 데이터 | 문제 | 상태 |
|---|---|---|---|
| [SECOM](secom/) | UCI SECOM, 공정 센서 590개 × 1567개 생산 단위 | 센서 값으로 불량 예측, 점검할 센서 찾기 | 완료 |
| [Wafer Map](wafer-map/) | WM-811K, 라벨 있는 웨이퍼 맵 172,950개 | 웨이퍼 불량 패턴 이미지 분류 (9가지) | 진행 중 |

## SECOM 요약

**문제**: 공정 센서 590개로 생산 단위의 불량(6.6%)을 미리 예측하고, 엔지니어가 먼저 점검할 센서를 찾는다.

**평가 설계**
- 데이터를 가장 먼저 시간순으로 나누고, 평가 데이터는 코드로 잠갔다.
- EDA부터 판정 기준까지 모든 결정을 학습 데이터로 정해 [평가 계획서](secom/PROTOCOL.md)로 먼저 커밋한 뒤, 평가 데이터를 한 번만 열었다.
- 첫 작업에서 탐색 중 평가 데이터 점수를 확인한 실수가 있어, 이전 작업을 보관하고 처음부터 다시 진행했다.

**결과**
- 위험도 상위 20%를 검사하면 로지스틱 회귀가 불량 24개 중 6개를 잡았다(무작위 기대값 4.8개). 다만 신뢰구간으로 보면 무작위와 통계적으로 구분되지 않는다.
- 같은 모델을 무작위 분할로 평가하면 PR-AUC가 약 2~2.4배 높게 나온다. 같은 시기 제품이 학습과 평가에 섞이면 성능이 부풀려진다.
- 학습 기간에 정한 확률 기준값은 다음 시기에 판정 비율이 6~24%로 흔들렸다. 운영에서는 묶음 내 상위 N%를 검사하는 방식이 안정적이다.
- SHAP과 시기 통제 분석으로 점검 후보 센서(s059 등)를 골랐고, 결측이 시기를 대신하는 센서는 후보에서 낮게 평가했다.

자세한 내용: [secom/README.md](secom/README.md)

## 기술

Python, pandas, scikit-learn (Pipeline, TimeSeriesSplit), LightGBM, SHAP, pytest

## 실행 환경

Python 3.13에서 확인했습니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r secom/requirements.txt
```

macOS에서는 LightGBM이 OpenMP를 필요로 합니다: `brew install libomp`.
정확한 버전은 `secom/requirements.lock.txt`에 있습니다.
