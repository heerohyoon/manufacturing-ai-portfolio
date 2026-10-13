"""SECOM 전처리 파이프라인.

데이터에서 값을 배우는 단계(결측률, 중앙값, 최소/최대, 평균/표준편차)는 모두
sklearn Pipeline 안에 둔다. 그래야 fit은 학습 데이터만 보고, 평가 데이터는
학습 데이터에서 배운 값으로 변환만 된다. 즉 평가 데이터 정보가 새지 않는다.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DropHighMissing(BaseEstimator, TransformerMixin):
    """학습 데이터에서 결측률이 threshold를 넘는 센서를 뺀다.

    절반 넘게 비어 있는 센서를 중앙값으로 채우면 실제 측정보다 채운 값이 더 많아진다.
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


class AddMissingIndicators(BaseEstimator, TransformerMixin):
    """자주 비는 센서마다 '<센서>_isna' 열(비었으면 1)을 붙인다.

    EDA에서 일부 센서는 비어 있는지 여부가 불량률과 관련 있어 보였다.
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


class ClipToTrainRange(BaseEstimator, TransformerMixin):
    """각 센서값을 학습 데이터에서 본 [최소, 최대] 범위로 자른다.

    평가 기간에 9998.9 같은 센서 오류값이 있다(학습 기간 범위는 0.03~3.2).
    로지스틱 회귀는 값에 가중치를 곱해 더하므로, 이런 값 하나가 점수 전체를 좌우할 수 있다.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.min_ = X.min()
        self.max_ = X.max()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(lower=self.min_, upper=self.max_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return self.min_.index.to_numpy(dtype=object)


def build_linear_preprocessor(missing_threshold: float = 0.5, add_indicator: bool = False) -> Pipeline:
    """로지스틱 회귀용. 결측이 없어야 하고, 극단값을 막고, 센서 간 크기를 맞춰야 한다.

    순서: 결측 센서 제거 → 중앙값 대체 → 상수 센서 제거 → 범위 자르기 → 표준화
    결측을 먼저 채워야 분산과 범위를 계산할 수 있고,
    오류값을 자른 뒤 표준화해야 평균/표준편차가 오류값에 끌려가지 않는다.
    """
    steps = [("missing_flags", AddMissingIndicators())] if add_indicator else []
    steps += [
        ("drop_missing", DropHighMissing(threshold=missing_threshold)),
        ("impute", SimpleImputer(strategy="median")),  # 센서값은 치우침이 있어 평균보다 중앙값이 안전하다
        ("drop_constant", VarianceThreshold(threshold=0.0)),  # 값이 하나뿐인 센서는 구분에 도움이 안 된다
        ("clip", ClipToTrainRange()),
        ("scale", StandardScaler()),  # 평균 0, 표준편차 1로 맞춰 가중치 크기를 공정하게 비교한다
    ]
    return Pipeline(steps).set_output(transform="pandas")


def build_tree_preprocessor(missing_threshold: float = 0.5, add_indicator: bool = False) -> Pipeline:
    """LightGBM용. 결측 대체, 범위 자르기, 표준화를 하지 않는다.

    LightGBM은 결측을 어느 가지로 보낼지 스스로 배운다. 중앙값으로 덮으면 그 정보가 사라진다.
    트리는 '값이 X보다 큰가'만 보므로 단위가 상관없고, 극단값은 가장 바깥 구간에 들어갈 뿐이다.
    """
    steps = [("missing_flags", AddMissingIndicators())] if add_indicator else []
    steps += [
        ("drop_missing", DropHighMissing(threshold=missing_threshold)),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]
    return Pipeline(steps).set_output(transform="pandas")
