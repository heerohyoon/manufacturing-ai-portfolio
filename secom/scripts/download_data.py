"""UCI SECOM 데이터를 내려받아 하나의 표로 합친다.

원본 파일
  - secom.data        : 1567행 x 590개 센서값, 공백 구분, 결측은 "NaN"
  - secom_labels.data : 행마다 판정(-1 = 정상, 1 = 불량)과 생산 시각

결과
  - data/raw/                  원본 그대로 보관
  - data/processed/secom.parquet  시각 + 불량 여부(0/1) + 센서 590개

실행
  python scripts/download_data.py
"""
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom"
FILES = ["secom.data", "secom_labels.data"]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def download(name: str) -> Path:
    path = RAW_DIR / name
    if path.exists():
        print(f"[skip] {name} 이미 있음")
        return path
    url = f"{BASE_URL}/{name}"
    print(f"[download] {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def build_table() -> pd.DataFrame:
    # 원본에는 센서 이름이 없어서(익명화) 번호로 붙인다
    X = pd.read_csv(RAW_DIR / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        RAW_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label_raw", "timestamp"],  # 시각은 "19/07/2008 11:55:00"처럼 따옴표로 묶인 한 칸
    )
    # -1/1을 0/1로 바꾼다. 불량을 1(양성)로 둬야 재현율, PR-AUC가 불량 기준으로 계산된다
    y = (labels["label_raw"] == 1).astype(int).rename("fail")
    ts = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")

    assert len(X) == len(y), "센서 행 수와 라벨 행 수가 다르다"
    return pd.concat([ts, y, X], axis=1)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)

    df = build_table()
    out = PROCESSED_DIR / "secom.parquet"
    df.to_parquet(out, index=False)
    print(f"[saved] {out}  shape={df.shape}  불량률={df['fail'].mean():.3%}")


if __name__ == "__main__":
    main()
