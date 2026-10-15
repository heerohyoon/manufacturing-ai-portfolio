"""평가 도구: 학습 데이터 안의 예측(OOF)으로 판정 기준 정하기, 지표 계산, 부트스트랩 신뢰구간."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.models import build_lightgbm, build_logistic


def models_from_params(best_params: dict) -> dict:
    """reports/best_params.json에 고정해 둔 설정으로 모델을 다시 만든다."""
    lr, gb = best_params["logistic"], best_params["lightgbm"]
    return {
        "로지스틱 회귀": build_logistic(C=lr["C"], add_indicator=lr["missing_flags"] == "on"),
        "LightGBM": build_lightgbm(
            add_indicator=gb["missing_flags"] == "on",
            max_depth=gb["max_depth"],
            colsample_bytree=gb["colsample_bytree"],
        ),
    }


def oof_predictions(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """학습 데이터 안에서 시간순 교차검증을 돌려, 각 검증 구간의 예측 확률을 모은다.

    판정 기준은 이 예측으로 정한다. test를 보고 기준을 정하면 그 자체가 누수다.
    첫 구간은 검증에 쓰이지 않으므로 (정답, 확률) 쌍은 학습 데이터 일부만 나온다.
    """
    y_true, proba = [], []
    for tr, va in TimeSeriesSplit(n_splits=n_splits).split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        proba.append(m.predict_proba(X.iloc[va])[:, 1])
        y_true.append(y.iloc[va].to_numpy())
    return np.concatenate(y_true), np.concatenate(proba)


def threshold_for_rate(proba: np.ndarray, inspect_rate: float) -> float:
    """위험도 상위 inspect_rate 비율만 '불량'으로 판정되도록 하는 확률 기준값."""
    return float(np.quantile(proba, 1 - inspect_rate))


def flag_top_rate(proba: np.ndarray, rate: float) -> float:
    """이번 묶음(batch) 안에서 위험도 상위 rate 비율이 되는 기준값. 정답 라벨은 쓰지 않는다.

    '매 묶음마다 위험한 순서로 20%를 검사한다'는 할당량 방식이다. 시기에 따라 모델 확률이
    전체적으로 올라가거나 내려가도 검사 비율은 그대로 유지된다.
    """
    return float(np.quantile(proba, 1 - rate))


def classification_summary(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "판정 기준": threshold,
        "불량 판정 비율": pred.mean(),
        "재현율(불량 중 잡은 비율)": recall_score(y_true, pred, zero_division=0),
        "정밀도(불량 판정 중 진짜 불량)": precision_score(y_true, pred, zero_division=0),
        "정확도(참고)": (tp + tn) / len(pred),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def bootstrap_ci(y_true, proba, metric, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """평가 데이터를 복원추출로 n_boot번 다시 뽑아 지표의 95% 구간을 구한다.

    test 불량이 24개뿐이라 점수 하나만 보면 우연과 실력을 구분할 수 없다.
    불량이 한 개도 안 뽑힌 표본은 지표를 계산할 수 없어서 건너뛴다.
    """
    rng = np.random.default_rng(seed)
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        scores.append(metric(y_true[idx], proba[idx]))
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    # 대표값은 재표본 평균이 아니라 원래 test 전체로 계산한 점수를 쓴다
    return float(metric(y_true, proba)), float(lo), float(hi)


def bootstrap_diff_ci(y_true, proba_a, proba_b, metric=average_precision_score,
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """두 모델 점수 차이(a - b)의 95% 구간. 같은 표본으로 둘 다 계산해야 공정하게 비교된다.

    구간이 0을 포함하면 '두 모델 차이가 우연일 수 있다'는 뜻이다.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        diffs.append(metric(y_true[idx], proba_a[idx]) - metric(y_true[idx], proba_b[idx]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, proba_a) - metric(y_true, proba_b)), float(lo), float(hi)


def recall_at_rate(y_true, proba, rate: float) -> float:
    """묶음 안에서 위험도 상위 rate 비율을 검사했을 때 잡히는 불량 비율. flag_top_rate와 같은 기준을 쓴다."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = proba >= flag_top_rate(proba, rate)
    return float((pred & (y_true == 1)).sum() / max(1, y_true.sum()))
