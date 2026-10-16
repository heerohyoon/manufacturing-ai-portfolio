"""M2 CNN: fold 0으로 멈출 시점을 정하고, 학습 데이터 전체로 다시 학습한다 (PROTOCOL.md 3~4절).

scripts/train.py가 만든 캐시와 decisions.json을 쓴다. LightGBM은 불러오지 않는다.

결과
  reports/cnn_history.csv     에폭별 기록 (fold 0 검증)
  reports/decisions.json      cnn_best_epoch, CNN fold 0 매크로 F1 추가
  reports/cv_per_class.csv    M2 (fold 0) 열 추가
  models/cnn_final.pt         최종 CNN 가중치

실행 (wafer-map/ 폴더에서): python scripts/train_cnn.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cnn import predict_proba, train_cnn  # noqa: E402
from src.data import CLASSES, load_train  # noqa: E402
from src.evaluation import CV, macro_f1  # noqa: E402

REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"
CACHE = ROOT / "data" / "processed" / "train_cache.npz"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_train()
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    X_img = np.load(CACHE)["X_img"]
    folds = list(CV.split(np.zeros(len(y)), y, groups=df["lot"]))  # train.py와 같은 분할

    tr, va = folds[0]
    log(f"M2 fold 0: 학습 {len(tr):,}, 검증 {len(va):,}")
    model, history, best_epoch = train_cnn(X_img[tr], y[tr], epochs=30, X_val=X_img[va], y_val=y[va],
                                           patience=5, seed=0, log=lambda r: log(f"M2 {r}"))
    pred = predict_proba(model, X_img[va]).argmax(1)
    f1_fold0 = macro_f1(y[va], pred)
    pd.DataFrame(history).to_csv(REPORT_DIR / "cnn_history.csv", index=False)

    per_class = pd.read_csv(REPORT_DIR / "cv_per_class.csv", index_col=0)
    per_class["M2 (fold 0)"] = f1_score(y[va], pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    per_class.to_csv(REPORT_DIR / "cv_per_class.csv", encoding="utf-8-sig")

    log(f"M2 최종 학습: 전체 {len(y):,}개, {best_epoch} 에폭")
    final, _, _ = train_cnn(X_img, y, epochs=best_epoch, seed=0, log=lambda r: log(f"M2 final {r}"))
    torch.save(final.state_dict(), MODEL_DIR / "cnn_final.pt")

    decisions = json.loads((REPORT_DIR / "decisions.json").read_text())
    decisions["cnn_best_epoch"] = int(best_epoch)
    decisions["cv_macro_f1"]["M2_fold0"] = f1_fold0
    (REPORT_DIR / "decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False))
    log(f"완료: {decisions}")


if __name__ == "__main__":
    main()
