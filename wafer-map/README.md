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
├── scripts/prepare_data.py   # 원본 읽기(구버전 pickle 호환), 라벨 있는 웨이퍼만, 원본 분할로 저장
├── src/data.py               # 학습용 / 평가용(잠금) 불러오기
├── tests/test_data.py        # 평가 데이터 잠금 규칙 확인
└── data/                     # git 미포함
```
