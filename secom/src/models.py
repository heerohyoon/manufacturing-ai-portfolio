"""로지스틱 회귀와 LightGBM 정의, 시간순 교차검증으로 설정값 고르기 (PROTOCOL.md 3~4절)."""
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from src.preprocess import AddMissingIndicators, build_linear_preprocessor, build_tree_preprocessor

RANDOM_STATE = 42
N_SPLITS = 4


def build_logistic(C: float = 0.01, add_indicator: bool = False) -> Pipeline:
    """전처리 + L2 로지스틱 회귀.

    C가 작을수록 규제가 강하다. 센서 수백 개에 불량 80개뿐이라 규제 없이는 학습 데이터를 외운다.
    class_weight="balanced": 불량을 틀렸을 때 손실을 (정상 수 / 불량 수)배, 약 14배로 키운다.
    """
    return Pipeline([
        ("prep", build_linear_preprocessor(add_indicator=add_indicator)),
        ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=5000)),
    ])


def build_lightgbm(max_depth: int = 3, colsample_bytree: float = 0.3, add_indicator: bool = False) -> Pipeline:
    """전처리 + LightGBM.

    데이터가 작아서 나무를 얕고 단순하게 둔다.
    - max_depth: 나무 하나가 연달아 던질 수 있는 질문 수
    - min_child_samples=20: 잎 하나에 제품이 20개 이상 있어야 갈라진다 (우연한 패턴 방지)
    - subsample, colsample_bytree: 나무마다 제품 80%, 센서 일부만 무작위로 써서 특정 센서 의존을 줄인다
    """
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=8,
        max_depth=max_depth,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,  # 교차검증을 병렬로 돌리므로 모델 내부 스레드는 1개로 둔다
        verbose=-1,
    )
    return Pipeline([("prep", build_tree_preprocessor(add_indicator=add_indicator)), ("model", model)])


# 결측 지시자는 끈 경우("passthrough")와 켠 경우를 함께 비교한다
INDICATOR_OPTIONS = ["passthrough", AddMissingIndicators()]

LOGISTIC_GRID = {
    "model__C": [0.001, 0.01, 0.1, 1.0],
    "prep__missing_flags": INDICATOR_OPTIONS,
}
LIGHTGBM_GRID = {
    "model__max_depth": [2, 3],
    "model__colsample_bytree": [0.1, 0.3],
    "prep__missing_flags": INDICATOR_OPTIONS,
}


def tune(pipe: Pipeline, grid: dict, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
    """학습 데이터 안에서 시간순 교차검증으로 설정값을 고른다.

    TimeSeriesSplit은 항상 과거 구간으로 학습하고 바로 다음 구간으로 검증한다.
    기준은 4개 fold PR-AUC(average_precision)의 평균이다.
    """
    search = GridSearchCV(
        pipe,
        grid,
        cv=TimeSeriesSplit(n_splits=N_SPLITS),
        scoring="average_precision",
        refit=True,  # 가장 좋은 설정으로 학습 데이터 전체를 다시 학습
        n_jobs=-1,
    )
    return search.fit(X, y)


def readable_params(best_params: dict) -> dict:
    """GridSearchCV 결과를 사람이 읽고 JSON으로 저장할 수 있는 형태로 바꾼다."""
    out = {}
    for key, value in best_params.items():
        name = key.replace("model__", "").replace("prep__", "")
        if name == "missing_flags":
            out["add_indicator"] = not (isinstance(value, str) and value == "passthrough")
        else:
            out[name] = value
    return out


def build_from_params(name: str, params: dict) -> Pipeline:
    """reports/best_params.json에 고정한 설정으로 모델을 다시 만든다."""
    if name == "logistic":
        return build_logistic(**params)
    if name == "lightgbm":
        return build_lightgbm(**params)
    raise ValueError(name)
