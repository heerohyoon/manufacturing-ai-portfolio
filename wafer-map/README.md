# Wafer Map: 불량 패턴 분류

WM-811K 웨이퍼 맵으로 Center, Edge-Ring, Scratch 등 9가지 불량 패턴을 분류한다. (진행 중)

## 데이터

| 항목 | 내용 |
|---|---|
| 출처 | [WM-811K (Kaggle)](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map), Wu et al. (2015) |
| 전체 | 웨이퍼 맵 811,457개, 그중 라벨 있는 것 172,950개 |
| 클래스 | Center, Donut, Edge-Loc, Edge-Ring, Loc, Near-full, Random, Scratch, none |
| 분할 | 원본 분할 그대로. 학습 54,355개(5,809 로트) / 평가 118,595개(4,953 로트), 겹치는 로트 없음 |

원본 분할을 쓴 이유
- 이미 로트 단위로 나뉘어 있어 같은 로트가 학습과 평가에 섞이지 않는다.
- 평가 데이터 라벨을 보지 않고 바로 잠글 수 있다. 새로 나누려면 층화를 위해 전체 라벨을 먼저 봐야 한다.
- 같은 분할을 쓰는 기존 연구와 비교할 수 있다.

## 실행

```bash
# 저장소 루트에서 가상환경 활성화 후
cd wafer-map
# Kaggle에서 LSWMD.pkl을 내려받아 data/raw/에 둔다
python scripts/prepare_data.py   # 학습용 / 평가용(잠금) 파일 생성
python -m pytest -q
```

## 구조

```
wafer-map/
├── scripts/
│   ├── prepare_data.py       # 원본 읽기(구버전 pickle 호환), 라벨 있는 웨이퍼만, 원본 분할로 저장
│   └── make_viewer.py        # 학습 데이터를 패턴별로 넘겨보는 HTML 뷰어
├── notebooks/01_eda.ipynb    # 학습 데이터만으로 EDA
├── src/data.py               # 학습용 / 평가용(잠금) 불러오기
├── tests/test_data.py        # 평가 데이터 잠금 규칙 확인
└── data/                     # git 미포함
```

## EDA 결과 (학습 데이터)

![examples](reports/figures/02_examples.png)

| 발견 | 수치 | 결정 |
|---|---|---|
| 심한 불균형 | none 67.6%, Near-full 54개 | 매크로 F1, 패턴별 재현율, 클래스 가중치 |
| 패턴마다 위치가 다름 | 고리별 불량 밀도가 패턴마다 다름. Edge-Loc과 Edge-Ring은 방향 정보가 필요 | 기준 모델: 고리별·방향별 밀도 등 직접 만든 특징 + LightGBM |
| 크기가 제각각 | 94%가 64칸 이하. 큰 맵을 단순히 줄이면 가는 선이 사라짐 | 64×64, 줄일 때 불량 칩 보존 |
| **맵 크기가 답을 알려줌** | 맵 크기만으로 매크로 F1 0.48 (Edge-Ring 0.94) | 크기만 쓴 모델을 기준선으로, 학습에 없던 크기의 웨이퍼 성능 따로 보고 |
| 로트 안 패턴 반복 | Edge-Ring이 나온 로트의 92%가 Edge-Ring | 교차검증도 로트 단위 |
| 똑같은 맵 | 평가 데이터에 학습 데이터와 똑같은 맵 3,158개 | 똑같은 맵을 뺀 점수도 보고 |
| 라벨 오류 가능성 | 눈으로 봐도 다른 패턴 같은 예가 있음 | 결과 해석 시 한계로 언급 |

![radial profile](reports/figures/04_radial_profile.png)
