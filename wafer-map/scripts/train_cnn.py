"""M2 CNN: 학습률 코사인 스케줄로 30에폭 학습한다 (PROTOCOL.md 3~4절, 변경 기록 참고).

fold 0으로 학습하면서 검증 곡선을 기록해 수렴하는지 확인하고, 최종 모델은 학습 데이터 전체로 30에폭 학습한다.

scripts/train.py가 만든 캐시와 decisions.json을 쓴다. LightGBM은 불러오지 않는다.

결과
  reports/cnn_history.csv     에폭별 기록 (fold 0 검증)
  reports/decisions.json      CNN 에폭 수, fold 0 매크로 F1(마지막 에폭) 추가
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
EPOCHS = 30


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
    model, history, _ = train_cnn(X_img[tr], y[tr], epochs=EPOCHS, X_val=X_img[va], y_val=y[va],
                                  seed=0, log=lambda r: log(f"M2 {r}"))
    pred = predict_proba(model, X_img[va]).argmax(1)
    f1_fold0 = macro_f1(y[va], pred)
    pd.DataFrame(history).to_csv(REPORT_DIR / "cnn_history.csv", index=False)

    per_class = pd.read_csv(REPORT_DIR / "cv_per_class.csv", index_col=0)
    per_class["M2 (fold 0)"] = f1_score(y[va], pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    per_class.to_csv(REPORT_DIR / "cv_per_class.csv", encoding="utf-8-sig")

    log(f"M2 최종 학습: 전체 {len(y):,}개, {EPOCHS} 에폭")
    final, _, _ = train_cnn(X_img, y, epochs=EPOCHS, seed=0, log=lambda r: log(f"M2 final {r}"))
    torch.save(final.state_dict(), MODEL_DIR / "cnn_final.pt")

    decisions = json.loads((REPORT_DIR / "decisions.json").read_text())
    decisions.pop("cnn_best_epoch", None)
    decisions["cnn_epochs"] = EPOCHS
    decisions["cnn_schedule"] = "cosine, 1e-3 → 0"
    decisions["cv_macro_f1"]["M2_fold0"] = f1_fold0
    (REPORT_DIR / "decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False))
    log(f"완료: {decisions}")


if __name__ == "__main__":
    main()
