"""평가 데이터 잠금 규칙을 확인한다.

실행 (secom/ 폴더에서): python -m pytest -q
"""
from pathlib import Path

import pytest

from src import data

ROOT = Path(__file__).resolve().parents[1]
FINAL_SCRIPT = ROOT / "scripts" / "final_evaluation.py"


def test_load_test_requires_unlock():
    with pytest.raises(PermissionError):
        data.load_test(unlock="")


def test_only_final_script_opens_test():
    """load_test를 부르는 파일은 최종 평가 스크립트 하나뿐이어야 한다 (노트북 포함)."""
    callers = []
    for path in list(ROOT.rglob("*.py")) + list(ROOT.rglob("*.ipynb")):
        if ".venv" in path.parts or path.name in {"data.py", "test_data.py"}:
            continue
        if "load_test(" in path.read_text(encoding="utf-8"):
            callers.append(path.relative_to(ROOT))
    assert set(callers) <= {FINAL_SCRIPT.relative_to(ROOT)}, callers


def test_train_comes_before_test(tmp_path):
    import numpy as np
    import pandas as pd

    n = 20
    df = pd.DataFrame({
        "timestamp": pd.date_range("2008-01-01", periods=n, freq="h")[::-1],  # 일부러 역순으로 저장
        "fail": np.r_[np.zeros(15, int), np.ones(5, int)],
        "sensor_000": np.arange(n, dtype=float),
    })
    path = tmp_path / "toy.parquet"
    df.to_parquet(path)

    _, _, ts_train = data.load_train(path)
    _, _, ts_test = data.load_test(data.UNLOCK_PHRASE, path)
    assert ts_train.max() < ts_test.min()
    assert len(ts_train) == 15 and len(ts_test) == 5
