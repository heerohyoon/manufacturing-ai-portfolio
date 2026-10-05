"""Download the UCI SECOM dataset and save it as a single tidy file.

Raw files:
  - secom.data        : 1567 rows x 590 space-separated sensor values ("NaN" for missing)
  - secom_labels.data : label (-1 = pass, 1 = fail) and timestamp per row

Output:
  - data/raw/secom.data, data/raw/secom_labels.data (originals, untouched)
  - data/processed/secom.parquet (sensors + label(0/1) + timestamp merged)

Usage:
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
        print(f"[skip] {name} already exists")
        return path
    url = f"{BASE_URL}/{name}"
    print(f"[download] {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def build_table() -> pd.DataFrame:
    # Sensor names are anonymized in the source, so use sensor_000..sensor_589
    X = pd.read_csv(RAW_DIR / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]

    labels = pd.read_csv(
        RAW_DIR / "secom_labels.data",
        sep=r"\s+",
        header=None,
        names=["label_raw", "timestamp"],  # timestamp is one quoted field: "19/07/2008 11:55:00"
    )
    # -1 = pass, 1 = fail -> remap to 0/1 so "fail" is the positive class
    y = (labels["label_raw"] == 1).astype(int).rename("fail")
    ts = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S")

    assert len(X) == len(y), "sensor rows and labels must align"
    return pd.concat([ts, y, X], axis=1)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(name)

    df = build_table()
    out = PROCESSED_DIR / "secom.parquet"
    df.to_parquet(out, index=False)
    print(f"[saved] {out}  shape={df.shape}  fail_rate={df['fail'].mean():.3%}")


if __name__ == "__main__":
    main()
