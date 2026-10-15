"""전처리 파이프라인이 학습 데이터에서만 값을 배우는지 확인한다.

실행 (secom/ 폴더에서): python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import build_linear_preprocessor, build_tree_preprocessor


@pytest.fixture
def toy():
    """앞 320행을 학습, 뒤 80행을 다른 구간으로 쓰는 작은 데이터."""
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({
        "sensor_a": rng.normal(0, 1, n),
        "sensor_b": rng.normal(5, 2, n),
        "sensor_const": np.ones(n),                       # 상수 → 제거돼야 함
        "sensor_mostly_nan": [np.nan] * 280 + [1.0] * 120,  # 학습 구간 결측 50% 초과 → 제거돼야 함
    })
    X.loc[5, "sensor_a"] = 9998.0                         # 학습 구간 안의 오류값
    X.loc[320:, "sensor_a"] = 1000.0                       # 다른 구간에만 있는 오류값
    X.loc[[3, 10, 330], "sensor_b"] = np.nan
    return X.iloc[:320], X.iloc[320:]


def test_statistics_come_from_train(toy):
    X_tr, _ = toy
    pipe = build_linear_preprocessor().fit(X_tr)

    imputer = pipe.named_steps["impute"]
    b_idx = list(imputer.feature_names_in_).index("sensor_b")
    assert imputer.statistics_[b_idx] == pytest.approx(X_tr["sensor_b"].median())

    clip = pipe.named_steps["clip"]
    # 분위수 경계는 오류값이 아니라 정상 범위 안에 있어야 한다
    assert clip.high_["sensor_a"] <= X_tr["sensor_a"].drop(index=5).max()


def test_extreme_values_are_capped(toy):
    X_tr, X_other = toy
    pipe = build_linear_preprocessor().fit(X_tr)
    assert pipe.transform(X_tr)["sensor_a"].abs().max() < 10
    assert pipe.transform(X_other)["sensor_a"].abs().max() < 10


def test_linear_output_is_clean(toy):
    X_tr, X_other = toy
    pipe = build_linear_preprocessor().fit(X_tr)
    out_tr, out_other = pipe.transform(X_tr), pipe.transform(X_other)
    assert "sensor_const" not in out_tr.columns
    assert "sensor_mostly_nan" not in out_tr.columns
    assert list(out_tr.columns) == list(out_other.columns)
    assert not out_other.isna().any().any()


def test_indicator_survives_dropped_sensor(toy):
    X_tr, _ = toy
    out = build_linear_preprocessor(add_indicator=True).fit_transform(X_tr)
    assert "sensor_mostly_nan" not in out.columns
    assert "sensor_mostly_nan_isna" in out.columns


def test_tree_keeps_nan_and_raw_values(toy):
    X_tr, _ = toy
    out = build_tree_preprocessor().fit_transform(X_tr)
    assert out["sensor_b"].isna().sum() == 2
    assert out["sensor_a"].max() == 9998.0
    assert "sensor_const" not in out.columns
