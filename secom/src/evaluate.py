"""판정 규칙 결정(PROTOCOL.md 5절)과 평가 지표·신뢰구간(6~7절)."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from src.models import N_SPLITS

INSPECT_RATE = 0.20
DRIFT_TOLERANCE = 0.05  # 판정 비율이 20%에서 평균 5%p 넘게 벗어나면 고정 기준값을 쓰지 않는다


def oof_by_fold(model, X: pd.DataFrame, y: pd.Series) -> list[dict]:
    """학습 데이터 안의 시간순 교차검증 예측을 fold별로 돌려준다.

    fold k의 예측은 fold k보다 앞선 데이터로만 학습한 모델이 낸 것이다.
    """
    folds = []
    for tr, va in TimeSeriesSplit(n_splits=N_SPLITS).split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        folds.append({"y": y.iloc[va].to_numpy(), "proba": m.predict_proba(X.iloc[va])[:, 1]})
    return folds


def cutoff_for_rate(proba: np.ndarray, rate: float = INSPECT_RATE) -> float:
    """예측 확률 상위 rate 비율의 경계값."""
    return float(np.quantile(proba, 1 - rate))


def fixed_cutoff_drift(folds: list[dict], rate: float = INSPECT_RATE) -> pd.DataFrame:
    """고정 기준값이 다음 시기에도 같은 비율을 걸러내는지 학습 데이터 안에서 확인한다.

    fold 2~4 각각에 대해, 이전 fold들의 예측으로 정한 기준값을 그 fold에 적용해 판정 비율을 잰다.
    """
    rows = []
    for k in range(1, len(folds)):
        past = np.concatenate([f["proba"] for f in folds[:k]])
        cutoff = cutoff_for_rate(past, rate)
        flag_rate = float((folds[k]["proba"] >= cutoff).mean())
        rows.append({"fold": k + 1, "기준값": cutoff, "판정 비율": flag_rate, "20%와의 차이": flag_rate - rate})
    return pd.DataFrame(rows)


def choose_rule(drift: pd.DataFrame, tolerance: float = DRIFT_TOLERANCE) -> str:
    """판정 비율이 평균적으로 tolerance 넘게 어긋나면 묶음 내 할당량(B), 아니면 고정 기준값(A)."""
    return "B_batch_quota" if drift["20%와의 차이"].abs().mean() > tolerance else "A_fixed_cutoff"


def classification_summary(y_true, proba, cutoff: float) -> dict:
    pred = (np.asarray(proba) >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "판정 기준값": cutoff,
        "불량 판정 비율": pred.mean(),
        "재현율": recall_score(y_true, pred, zero_division=0),
        "정밀도": precision_score(y_true, pred, zero_division=0),
        "정확도(참고)": (tp + tn) / len(pred),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
    }


def recall_at_rate(y_true, proba, rate: float) -> float:
    """묶음 안에서 위험도 상위 rate 비율을 검사했을 때 잡히는 불량 비율."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    pred = proba >= cutoff_for_rate(proba, rate)
    return float((pred & (y_true == 1)).sum() / max(1, y_true.sum()))


def bootstrap_ci(y_true, proba, metric, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """평가 데이터를 복원추출로 n_boot번 다시 뽑아 지표의 95% 구간을 구한다.

    불량이 한 개도 안 뽑힌 표본은 지표를 계산할 수 없어서 건너뛴다.
    대표값은 재표본 평균이 아니라 원래 데이터로 계산한 점수다.
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
    return float(metric(y_true, proba)), float(lo), float(hi)


def bootstrap_diff_ci(y_true, proba_a, proba_b, metric, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """두 모델 점수 차이(a - b)의 95% 구간. 같은 표본으로 둘 다 계산해야 공정하다."""
    rng = np.random.default_rng(seed)
    y_true, proba_a, proba_b = map(np.asarray, (y_true, proba_a, proba_b))
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        diffs.append(metric(y_true[idx], proba_a[idx]) - metric(y_true[idx], proba_b[idx]))
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, proba_a) - metric(y_true, proba_b)), float(lo), float(hi)
