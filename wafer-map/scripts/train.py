"""학습 데이터만으로 M0, M1 교차검증과 설정 선택을 하고, 이어서 CNN 학습(scripts/train_cnn.py)을 별도 프로세스로 실행한다.

LightGBM과 PyTorch를 한 프로세스에서 함께 쓰면 macOS에서 멈추거나 죽기 때문에 나눴다 (src/models.py 참고).

결과
  reports/decisions.json      고정한 설정 (M1 설정, CNN 에폭 수)
  reports/cv_results.csv      교차검증 매크로 F1
  reports/cv_per_class.csv    교차검증 클래스별 F1
  (CNN 결과는 scripts/train_cnn.py 참고)

실행 (wafer-map/ 폴더에서): python scripts/train.py
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, load_train  # noqa: E402
from src.features import shape_features, size_features  # noqa: E402
from src.models import CV, M1_GRID, build_m0, build_m1, macro_f1  # noqa: E402
from src.preprocess import stack_maps  # noqa: E402

REPORT_DIR = ROOT / "reports"
CACHE = ROOT / "data" / "processed" / "train_cache.npz"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_inputs():
    df = load_train()
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        X_img = z["X_img"]
        F_size = pd.DataFrame(z["F_size"], columns=z["size_cols"])
        F_shape = pd.DataFrame(z["F_shape"], columns=z["shape_cols"])
    else:
        log("64x64 변환과 특징 계산")
        X_img = stack_maps(df["wafer_map"])
        F_size = size_features(df["wafer_map"])
        F_shape = shape_features(X_img)
        np.savez_compressed(CACHE, X_img=X_img, F_size=F_size.to_numpy(), size_cols=np.array(F_size.columns, dtype=str),
                            F_shape=F_shape.to_numpy(), shape_cols=np.array(F_shape.columns, dtype=str))
    return df, y, X_img, F_size, F_shape


def oof_lgbm(build, X: pd.DataFrame, y, folds):
    """fold마다 나머지로 학습해 그 fold를 예측한다 (out-of-fold). fold별 매크로 F1과 전체 예측을 돌려준다."""
    pred = np.empty(len(y), dtype=int)
    scores = []
    for tr, va in folds:
        m = build().fit(X.iloc[tr], y[tr])
        pred[va] = m.predict(X.iloc[va])
        scores.append(macro_f1(y[va], pred[va]))
    return scores, pred


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    df, y, X_img, F_size, F_shape = load_inputs()
    folds = list(CV.split(F_size, y, groups=df["lot"]))
    log(f"학습 {len(y):,}개, fold 검증 크기 {[len(va) for _, va in folds]}")

    rows, per_class = [], {}

    # M0: 크기 기준선
    s0, p0 = oof_lgbm(build_m0, F_size, y, folds)
    rows.append({"model": "M0 크기 기준선", "setting": "-", **{f"fold{i}": s for i, s in enumerate(s0)}})
    per_class["M0"] = f1_score(y, p0, labels=range(len(CLASSES)), average=None, zero_division=0)
    log(f"M0 CV 매크로 F1 {np.mean(s0):.3f}")

    # M1: 설정 4가지 중 CV 평균이 가장 높은 것
    best = None
    for params in M1_GRID:
        s1, p1 = oof_lgbm(lambda: build_m1(**params), F_shape, y, folds)
        rows.append({"model": "M1 특징 모델", "setting": json.dumps(params), **{f"fold{i}": s for i, s in enumerate(s1)}})
        log(f"M1 {params} CV 매크로 F1 {np.mean(s1):.3f}")
        if best is None or np.mean(s1) > best[0]:
            best = (np.mean(s1), params, p1)
    m1_params = best[1]
    per_class["M1"] = f1_score(y, best[2], labels=range(len(CLASSES)), average=None, zero_division=0)

    cv = pd.DataFrame(rows)
    fold_cols = [c for c in cv.columns if c.startswith("fold")]
    cv["mean"] = cv[fold_cols].mean(axis=1)
    cv.to_csv(REPORT_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(per_class, index=CLASSES).to_csv(REPORT_DIR / "cv_per_class.csv", encoding="utf-8-sig")

    decisions = {"m0": "고정 설정", "m1_params": m1_params,
                 "cv_macro_f1": {"M0": float(np.mean(s0)), "M1": float(best[0])}}
    (REPORT_DIR / "decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False))
    log(f"M0, M1 완료: {decisions}")

    log("CNN 학습을 별도 프로세스로 시작")
    subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "train_cnn.py")], check=True)


if __name__ == "__main__":
    main()
