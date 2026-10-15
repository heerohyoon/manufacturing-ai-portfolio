# SECOM: 공정 센서 기반 불량 예측

반도체 공정 센서 590개로 생산 단위의 Pass/Fail을 예측하고, 불량에 영향을 준 센서를 해석한다.

## 평가 원칙

1. 데이터를 가장 먼저 시간순으로 나눈다. 앞 75%(2008-07-19 ~ 09-29, 1175행)는 학습용, 뒤 25%(392행)는 평가용이다.
2. EDA, 전처리 기준, 모델 설정, 판정 기준은 모두 학습 데이터로만 정한다.
3. 모든 결정을 `PROTOCOL.md`에 적어 커밋한 뒤, `scripts/final_evaluation.py`로 평가 데이터를 한 번만 연다.
4. `src/data.py`의 `load_test()`는 잠금 문구가 있어야 열리고, 테스트가 최종 평가 스크립트 외의 호출을 막는다.

1차 작업(태그 `archive/v1`)에서는 탐색 중 평가 데이터 점수를 확인한 적이 있어 처음부터 다시 진행했다.

## 실행

```bash
# 저장소 루트에서 가상환경 활성화 후
cd secom
python scripts/download_data.py        # data/raw 다운로드 → data/processed/secom.parquet
python -m pytest -q
jupyter notebook notebooks/
```

## 구조

```
secom/
├── scripts/download_data.py   # UCI 원본 다운로드 + 라벨/시각 병합
├── src/data.py                # 학습용 / 평가용(잠금) 불러오기
├── notebooks/01_eda.ipynb     # 학습 데이터만으로 EDA
├── tests/test_data.py         # 평가 데이터 잠금 규칙 확인
├── data/                      # git 미포함 (스크립트로 재생성)
└── reports/figures/
```

## EDA 결과 (학습 데이터)

| 발견 | 수치 | 결정 |
|---|---|---|
| 불량이 드묾 | 80 / 1175 (6.8%) | PR-AUC, 불량 재현율. 정확도는 참고만 |
| 고결측 센서 | 50% 초과 24개 | 제거, 나머지 중앙값 대체(로지스틱) |
| 결측과 불량의 연관 | 8개 센서, 단 시기와 교란 | 결측 지시자 on/off는 교차검증으로 결정 |
| 상수 센서 | 122개 | 제거 |
| 시기 변화 | 월별 불량률 22% → 2.7% | 시간순 교차검증 |
| 센서 오류값 | 74개 센서 (예: 보통 0.17인데 9998) | 로지스틱: 0.5~99.5% 분위수로 자르기 |
| 중복 센서 | 191개가 \|r\|>0.95 | L2 규제, SHAP은 상관 그룹으로 해석 |

![fail rate over time](reports/figures/03_fail_rate_over_time.png)
![extreme values](reports/figures/04_extreme_values.png)
