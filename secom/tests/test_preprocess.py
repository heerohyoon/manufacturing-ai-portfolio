"""전처리 파이프라인이 학습 데이터에서만 값을 배우는지(누수가 없는지) 확인한다.

실행 (secom/ 폴더에서): python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src.data import time_split
from src.preprocess import build_linear_preprocessor, build_tree_preprocessor


@pytest.fixture
def toy():
    """학습 구간과 평가 구간의 분포를 일부러 다르게 만든 작은 데이터."""
    rng = np.random.default_rng(0)
    n = 100
    X = pd.DataFrame({
        "sensor_a": rng.normal(0, 1, n),
        "sensor_b": rng.normal(5, 2, n),
        "sensor_const": np.ones(n),                       # 상수 → 제거돼야 함
        "sensor_mostly_nan": [np.nan] * 70 + [1.0] * 30,  # 학습 구간 결측 70%를 넘음 → 제거돼야 함
    })
    X.loc[80:, "sensor_a"] = 1000.0                       # 평가 구간에만 있는 오류값
    X.loc[[3, 10, 85], "sensor_b"] = np.nan
    y = pd.Series([0] * 90 + [1] * 10)
    return X, y


def test_time_split_keeps_order(toy):
    X, y = toy
    X_tr, X_te, _, _ = time_split(X, y, test_size=0.2)
    assert X_tr.index.max() < X_te.index.min()
    assert len(X_tr) == 80 and len(X_te) == 20


def test_linear_learns_only_from_train(toy):
    X, y = toy
    X_tr, X_te, y_tr, _ = time_split(X, y, test_size=0.2)
    pipe = build_linear_preprocessor().fit(X_tr, y_tr)

    # 결측을 채운 중앙값은 전체가 아니라 학습 구간의 중앙값이어야 한다
    imputer = pipe.named_steps["impute"]
    b_idx = list(imputer.feature_names_in_).index("sensor_b")
    assert imputer.statistics_[b_idx] == pytest.approx(X_tr["sensor_b"].median())

    # 자르는 범위도 학습 구간 기준이라, 평가 구간의 1000.0은 잘려야 한다
    clip = pipe.named_steps["clip"]
    assert clip.max_["sensor_a"] == pytest.approx(X_tr["sensor_a"].max())
    out = pipe.transform(X_te)
    assert out["sensor_a"].abs().max() < 10


def test_linear_output_is_clean(toy):
    X, y = toy
    X_tr, X_te, y_tr, _ = time_split(X, y, test_size=0.2)
    pipe = build_linear_preprocessor().fit(X_tr, y_tr)
    out_tr, out_te = pipe.transform(X_tr), pipe.transform(X_te)

    assert "sensor_const" not in out_tr.columns
    assert "sensor_mostly_nan" not in out_tr.columns
    assert list(out_tr.columns) == list(out_te.columns)
    assert not out_te.isna().any().any()
    assert out_tr.mean().abs().max() < 1e-9


def test_indicator_survives_dropped_sensor(toy):
    X, y = toy
    X_tr, _, y_tr, _ = time_split(X, y, test_size=0.2)
    out = build_linear_preprocessor(add_indicator=True).fit_transform(X_tr, y_tr)
    assert "sensor_mostly_nan" not in out.columns
    assert "sensor_mostly_nan_isna" in out.columns


def test_tree_keeps_nan(toy):
    X, y = toy
    X_tr, _, y_tr, _ = time_split(X, y, test_size=0.2)
    out = build_tree_preprocessor().fit_transform(X_tr, y_tr)
    assert out["sensor_b"].isna().sum() == 2
    assert "sensor_const" not in out.columns
