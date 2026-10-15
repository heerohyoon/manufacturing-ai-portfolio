"""SECOM 전처리 파이프라인 (PROTOCOL.md 2절).

데이터에서 값을 배우는 단계(결측률, 중앙값, 분위수, 평균/표준편차)는 모두 sklearn Pipeline 안에 둔다.
fit은 학습 데이터만 보고, 다른 데이터는 학습 데이터에서 배운 값으로 변환만 한다.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AddMissingIndicators(BaseEstimator, TransformerMixin):
    """자주 비는 센서마다 '<센서>_isna' 열(비었으면 1)을 붙인다.

    DropHighMissing보다 먼저 실행해서, 값은 버려지는 센서도 '비었다'는 정보는 남긴다.
    """

    def __init__(self, min_rate: float = 0.05):
        self.min_rate = min_rate

    def fit(self, X: pd.DataFrame, y=None):
        missing_rate = X.isna().mean()
        self.flag_cols_ = missing_rate.index[(missing_rate >= self.min_rate) & (missing_rate < 1.0)].tolist()
        self.feature_names_out_ = list(X.columns) + [f"{c}_isna" for c in self.flag_cols_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        flags = X[self.flag_cols_].isna().astype(float).add_suffix("_isna")
        return pd.concat([X, flags], axis=1)

    def get_feature_names_out(self, input_features=None):
        return pd.Index(self.feature_names_out_).to_numpy(dtype=object)


class DropHighMissing(BaseEstimator, TransformerMixin):
    """학습 데이터에서 결측률이 threshold를 넘는 센서를 뺀다.

    절반 넘게 비어 있는 센서를 채우면 실제 측정보다 채운 값이 더 많아진다.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None):
        missing_rate = X.isna().mean()
        self.dropped_ = missing_rate.index[missing_rate > self.threshold].tolist()
        self.keep_ = missing_rate.index[missing_rate <= self.threshold].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.keep_]

    def get_feature_names_out(self, input_features=None):
        return pd.Index(self.keep_).to_numpy(dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """각 열을 학습 데이터의 [lower, upper] 분위수로 자른다.

    학습 데이터에도 9998 같은 센서 오류값이 있어서 최솟값~최댓값으로 자르면 오류값이 그대로 남는다.
    0.5% / 99.5% 분위수로 자르면 극소수 오류값만 경계값으로 바뀌고 나머지는 그대로다.
    """

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X: pd.DataFrame, y=None):
        self.low_ = X.quantile(self.lower)
        self.high_ = X.quantile(self.upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self.low_, upper=self.high_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.low_.index.to_numpy(dtype=object)


def missing_flags_step(add_indicator: bool):
    """결측 지시자 단계. 끌 때도 자리는 남겨둬서(passthrough) 교차검증에서 켜고 끄며 비교할 수 있게 한다."""
    return ("missing_flags", AddMissingIndicators() if add_indicator else "passthrough")


def build_linear_preprocessor(add_indicator: bool = False) -> Pipeline:
    """로지스틱 회귀용. 결측이 없어야 하고, 극단값을 막고, 센서 간 크기를 맞춰야 한다.

    결측을 먼저 채워야 분산과 분위수를 계산할 수 있고,
    극단값을 자른 뒤 표준화해야 평균/표준편차가 오류값에 끌려가지 않는다.
    """
    steps = [
        missing_flags_step(add_indicator),
        ("drop_missing", DropHighMissing(threshold=0.5)),
        ("impute", SimpleImputer(strategy="median")),  # 센서값은 치우침이 있어 평균보다 중앙값이 안전하다
        ("drop_constant", VarianceThreshold(threshold=0.0)),  # 값이 하나뿐인 센서는 구분에 도움이 안 된다
        ("clip", QuantileClipper(lower=0.005, upper=0.995)),
        ("scale", StandardScaler()),  # 평균 0, 표준편차 1로 맞춰 가중치 크기를 공정하게 비교한다
    ]
    return Pipeline(steps).set_output(transform="pandas")


def build_tree_preprocessor(add_indicator: bool = False) -> Pipeline:
    """LightGBM용. 결측 대체, 극단값 자르기, 표준화를 하지 않는다.

    LightGBM은 결측을 어느 가지로 보낼지 스스로 배운다. 중앙값으로 덮으면 그 정보가 사라진다.
    트리는 '값이 X보다 큰가'만 보므로 단위가 상관없고, 극단값은 가장 바깥 구간에 들어갈 뿐이다.
    """
    steps = [
        missing_flags_step(add_indicator),
        ("drop_missing", DropHighMissing(threshold=0.5)),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]
    return Pipeline(steps).set_output(transform="pandas")
