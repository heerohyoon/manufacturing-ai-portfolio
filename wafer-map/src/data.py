"""WM-811K 불러오기. 학습용과 평가용을 처음부터 따로 다룬다.

분할은 원본의 trianTestLabel을 그대로 쓴다 (학습 54,355 / 평가 118,595, 겹치는 로트 없음).
EDA, 전처리, 모델 설정, 판정 기준은 모두 load_train()만으로 정한다.
load_test()는 PROTOCOL.md를 커밋한 뒤 scripts/final_evaluation.py에서 한 번만 부른다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
UNLOCK_PHRASE = "final-evaluation"

# 원본 라벨 이름 그대로. 'none'은 특정 패턴이 없는 웨이퍼다.
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]


def load_train(path: Path = PROCESSED_DIR / "train.pkl") -> pd.DataFrame:
    """학습용 웨이퍼. 열: wafer_map, lot, wafer_index, die_size, label."""
    return pd.read_pickle(path)


def load_test(unlock: str, path: Path = PROCESSED_DIR / "test_locked.pkl") -> pd.DataFrame:
    """평가용 웨이퍼. 실수로 부르지 않도록 잠금 문구를 요구한다."""
    if unlock != UNLOCK_PHRASE:
        raise PermissionError("평가 데이터는 최종 평가 스크립트에서만 연다 (PROTOCOL.md 참고).")
    return pd.read_pickle(path)
