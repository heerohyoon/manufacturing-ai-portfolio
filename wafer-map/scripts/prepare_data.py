"""WM-811K 원본(LSWMD.pkl)에서 라벨 있는 웨이퍼만 골라 학습용과 평가용으로 나눠 저장한다.

원본 파일은 Kaggle에서 직접 내려받아 data/raw/LSWMD.pkl에 둔다.
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

분할은 원본 작성자의 trianTestLabel(원본 철자 그대로)을 쓴다. 두 쪽에 같은 로트가 없다는 것을 확인했다.
이 스크립트는 분할 크기와 로트 겹침만 출력하고, 평가용의 불량 패턴 분포는 출력하지 않는다.

실행 (wafer-map/ 폴더에서): python scripts/prepare_data.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "LSWMD.pkl"
OUT_DIR = ROOT / "data" / "processed"


class LegacyUnpickler(pickle.Unpickler):
    """2019년 이전 pandas로 저장된 파일을 읽기 위해 바뀐 모듈 이름을 지금 이름으로 연결한다."""

    def find_class(self, module, name):
        if module.startswith("pandas.indexes"):
            module = module.replace("pandas.indexes", "pandas.core.indexes", 1)
        return super().find_class(module, name)


def unwrap(value) -> str | None:
    """원본은 라벨을 [['Center']] 같은 2차원 배열로 저장했다. 비어 있으면 라벨이 없는 웨이퍼다."""
    if isinstance(value, np.ndarray) and value.size > 0:
        return str(value.ravel()[0])
    return None


def main() -> None:
    with open(RAW_PATH, "rb") as f:
        raw = LegacyUnpickler(f, encoding="latin1").load()  # 파이썬 2 시절 형식이라 latin1로 읽는다

    df = pd.DataFrame({
        "wafer_map": raw["waferMap"],
        "lot": raw["lotName"].astype(str),
        "wafer_index": raw["waferIndex"].astype(int),
        "die_size": raw["dieSize"].astype(int),
        "split": raw["trianTestLabel"].map(unwrap),
        "label": raw["failureType"].map(unwrap),
    })
    labeled = df[df["split"].isin(["Training", "Test"])].reset_index(drop=True)
    assert labeled["label"].notna().all(), "분할 표시가 있는데 라벨이 없는 웨이퍼가 있다"

    train = labeled[labeled["split"] == "Training"].drop(columns="split").reset_index(drop=True)
    test = labeled[labeled["split"] == "Test"].drop(columns="split").reset_index(drop=True)

    overlap = set(train["lot"]) & set(test["lot"])
    assert not overlap, f"학습/평가에 같은 로트가 있다: {len(overlap)}개"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_pickle(OUT_DIR / "train.pkl")
    test.to_pickle(OUT_DIR / "test_locked.pkl")
    print(f"[saved] 학습 {len(train):,}개 ({train['lot'].nunique():,}개 로트), "
          f"평가 {len(test):,}개 ({test['lot'].nunique():,}개 로트), 겹치는 로트 0개")


if __name__ == "__main__":
    main()
