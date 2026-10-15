"""SHAP 해석 도구 (PROTOCOL.md 8절). 학습 데이터만 사용한다."""
import numpy as np
import pandas as pd
import shap
from scipy.sparse.csgraph import connected_components

from src.models import build_from_params


def shap_frame(name: str, params: dict, X_fit: pd.DataFrame, y_fit: pd.Series, X_explain: pd.DataFrame) -> pd.DataFrame:
    """X_fit으로 학습한 모델의 SHAP 값(로그 오즈 단위)을 X_explain 각 행·각 입력 열에 대해 계산한다.

    로지스틱 회귀는 LinearExplainer, LightGBM은 TreeExplainer를 쓴다.
    SHAP 값이 양수면 그 센서 값이 불량 쪽으로, 음수면 정상 쪽으로 밀었다는 뜻이다.
    """
    pipe = build_from_params(name, params).fit(X_fit, y_fit)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Z = prep.transform(X_explain)
    if name == "logistic":
        background = prep.transform(X_fit)
        masker = shap.maskers.Independent(background, max_samples=len(background))
        values = shap.LinearExplainer(model, masker).shap_values(Z)
    else:
        values = shap.TreeExplainer(model).shap_values(Z)
    return pd.DataFrame(np.asarray(values), columns=Z.columns, index=X_explain.index)


def correlation_groups(X: pd.DataFrame, threshold: float = 0.95) -> pd.Series:
    """순위 상관(스피어만) |r| > threshold로 이어지는 센서끼리 같은 그룹 번호를 준다.

    피어슨 상관은 오류값 몇 행에 끌려가서, 실제로는 무관한 센서가 한 그룹으로 묶인다.
    순위 상관은 값의 크기가 아니라 순서만 보므로 9998 같은 값의 영향을 받지 않는다.
    A-B, B-C가 높으면 A, B, C를 한 그룹으로 본다.
    """
    corr = np.array(X.corr(method="spearman").abs().fillna(0))
    np.fill_diagonal(corr, 0)
    _, labels = connected_components((corr > threshold).astype(int), directed=False)
    return pd.Series(labels, index=X.columns, name="group")


def group_names(groups: pd.Series, max_members: int = 4) -> dict:
    names = {}
    for g, members in groups.groupby(groups):
        short = [m.replace("sensor_", "s") for m in members.index]
        names[g] = "+".join(short[:max_members]) + ("…" if len(short) > max_members else "")
    return names


def group_importance(S: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """같은 그룹의 SHAP 값을 행마다 더한 뒤, 절댓값 평균으로 그룹 중요도를 낸다.

    SHAP은 더할 수 있어서, 그룹의 합은 '이 그룹 전체가 예측을 얼마나 움직였나'가 된다.
    결측 지시자('<센서>_isna')는 센서 값과 의미가 달라서 따로 둔다.
    """
    names = group_names(groups)
    keys = {c: (f"{c[:-5].replace('sensor_', 's')} 결측 여부" if c.endswith("_isna") else names[groups[c]])
            for c in S.columns}
    summed = S.T.groupby(keys).sum().T
    return summed.abs().mean().sort_values(ascending=False)
