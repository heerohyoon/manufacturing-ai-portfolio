"""로지스틱 회귀(기준 모델)와 LightGBM(주력 모델) 정의, 시간순 교차검증으로 설정값 고르기."""
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.preprocess import AddMissingIndicators, build_linear_preprocessor, build_tree_preprocessor

RANDOM_STATE = 42


def build_logistic(C: float = 0.01, add_indicator: bool = False) -> Pipeline:
    """전처리 + L2 로지스틱 회귀.

    C가 작을수록 규제가 강하다. 센서 444개에 불량 80개뿐이라 규제 없이는 학습 데이터를 외운다.
    class_weight="balanced": 불량을 틀렸을 때 손실을 (정상 수 / 불량 수)배, 약 14배로 키운다.
    """
    return Pipeline([
        ("prep", build_linear_preprocessor(add_indicator=add_indicator)),
        ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
    ])


def build_lightgbm(add_indicator: bool = False, **params) -> Pipeline:
    """전처리 + LightGBM.

    데이터가 작아서 나무를 얕고 단순하게 둔다.
    - num_leaves, max_depth: 나무 하나가 만들 수 있는 질문 수를 제한
    - min_child_samples: 잎 하나에 최소 20개 제품이 있어야 갈라진다 (몇 개짜리 우연한 패턴 방지)
    - subsample, colsample_bytree: 나무마다 제품 80%, 센서 30%만 무작위로 써서 특정 센서 의존을 줄인다
    """
    defaults = dict(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=8,
        max_depth=3,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.3,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,  # 교차검증을 병렬로 돌리므로 모델 내부 스레드는 1개로 둔다
        verbose=-1,
    )
    defaults.update(params)
    return Pipeline([
        ("prep", build_tree_preprocessor(add_indicator=add_indicator)),
        ("model", LGBMClassifier(**defaults)),
    ])


# 후보 설정값. 결측 지시자는 끈 경우("passthrough")와 켠 경우를 함께 비교한다.
INDICATOR_OPTIONS = ["passthrough", AddMissingIndicators()]
LOGISTIC_GRID = {
    "model__C": [0.001, 0.01, 0.1, 1.0],
    "prep__missing_flags": INDICATOR_OPTIONS,
}
# 깊이 2면 잎이 최대 4개라 num_leaves(기본 8)는 깊이로만 조절한다
LIGHTGBM_GRID = {
    "model__max_depth": [2, 3],
    "model__colsample_bytree": [0.1, 0.3],
    "prep__missing_flags": INDICATOR_OPTIONS,
}


def tune(pipe: Pipeline, grid: dict, X: pd.DataFrame, y: pd.Series, n_splits: int = 4) -> GridSearchCV:
    """학습 데이터 안에서만 시간순 교차검증으로 설정값을 고른다.

    TimeSeriesSplit은 항상 과거 구간으로 학습하고 바로 다음 구간으로 검증한다.
    평가 데이터(test)는 여기서 전혀 쓰지 않는다. test는 마지막에 한 번만 본다.
    기준은 PR-AUC(average_precision): 불량이 드물 때 모델 간 차이를 가장 정직하게 보여준다.
    """
    search = GridSearchCV(
        pipe,
        grid,
        cv=TimeSeriesSplit(n_splits=n_splits),
        scoring="average_precision",
        refit=True,  # 가장 좋은 설정으로 학습 데이터 전체를 다시 학습
        n_jobs=-1,
    )
    return search.fit(X, y)
