"""SECOM 데이터 불러오기와 시간순 분할."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "secom.parquet"


def load_secom(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(센서값 X, 불량 여부 y, 생산 시각)을 시간순으로 정렬해서 돌려준다."""
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)
    X = df.filter(like="sensor_")
    y = df["fail"]
    return X, y, df["timestamp"]


def time_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.25):
    """앞쪽(과거)은 학습용, 뒤쪽(최근)은 평가용으로 나눈다.

    섞지 않는다. 실제 라인에서는 과거 데이터로 학습한 모델이 앞으로 나올 제품을
    예측하므로, 평가도 같은 조건이어야 성능을 부풀리지 않는다.
    """
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
