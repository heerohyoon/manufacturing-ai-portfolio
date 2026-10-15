"""SECOM 데이터 불러오기. 학습용과 평가용을 처음부터 따로 다룬다.

시간순으로 앞 75%는 학습용, 뒤 25%는 평가용이다. 이 경계는 다른 어떤 분석보다 먼저 정했다.
EDA, 전처리 기준, 모델 설정, 판정 기준은 모두 load_train()만으로 정한다.
load_test()는 PROTOCOL.md를 커밋한 뒤 scripts/final_evaluation.py에서 한 번만 부른다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "secom.parquet"
TEST_SIZE = 0.25
UNLOCK_PHRASE = "final-evaluation"


def _load_sorted(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)


def _cut_index(n_rows: int) -> int:
    return int(n_rows * (1 - TEST_SIZE))


def load_train(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """학습용 구간의 (센서값 X, 불량 여부 y, 생산 시각)."""
    df = _load_sorted(path)
    df = df.iloc[:_cut_index(len(df))]
    return df.filter(like="sensor_"), df["fail"], df["timestamp"]


def load_test(unlock: str, path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """평가용 구간. 실수로 부르지 않도록 잠금 문구를 요구한다."""
    if unlock != UNLOCK_PHRASE:
        raise PermissionError("평가 데이터는 최종 평가 스크립트에서만 연다 (PROTOCOL.md 참고).")
    df = _load_sorted(path)
    df = df.iloc[_cut_index(len(df)):]
    return df.filter(like="sensor_"), df["fail"], df["timestamp"]
