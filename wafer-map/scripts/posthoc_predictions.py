"""사후 분석용으로 평가 데이터의 예측을 저장한다 (PROTOCOL.md 8절, 변경 기록 참고).

최종 평가(scripts/final_evaluation.py)가 끝난 뒤에만 쓴다. 결과로 모델이나 보고한 점수를 바꾸지 않는다.
M1은 최종 평가와 같은 설정으로 다시 학습해 예측하고(같은 seed라 같은 결과), M2는 커밋한 가중치로 예측한다.

결과: data/processed/posthoc_test.pkl (git 미포함)
  wafer_map, lot, label, size, pred_m1, pred_m2, proba_m2, train_label_same_map
  train_label_same_map: 칸 값까지 똑같은 맵이 학습 데이터에 있으면 그 학습 라벨, 없으면 None

실행 (wafer-map/ 폴더에서): python scripts/posthoc_predictions.py
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, UNLOCK_PHRASE, load_test, load_train  # noqa: E402
from src.features import shape_features  # noqa: E402
from src.models import build_m1  # noqa: E402
from src.preprocess import stack_maps  # noqa: E402

OUT = ROOT / "data" / "processed" / "posthoc_test.pkl"


def main() -> None:
    if not (ROOT / "reports" / "final" / "final_evaluation.md").exists():
        sys.exit("최종 평가를 먼저 실행해야 한다.")
    decisions = json.loads((ROOT / "reports" / "decisions.json").read_text())
    train, test = load_train(), load_test(unlock=UNLOCK_PHRASE)
    y_train = train["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    img_train, img_test = stack_maps(train["wafer_map"]), stack_maps(test["wafer_map"])
    m1 = build_m1(**decisions["m1_params"]).fit(shape_features(img_train), y_train)
    pred_m1 = m1.predict(shape_features(img_test))

    src, dst = ROOT / "data" / "processed" / "eval_images.npy", ROOT / "data" / "processed" / "eval_cnn_proba.npy"
    np.save(src, img_test)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "predict_cnn.py"), str(src), str(dst)], check=True)
    proba_m2 = np.load(dst)
    src.unlink(); dst.unlink()

    def key(m):
        return m.shape, m.astype(np.int8).tobytes()
    train_label_by_map = dict(zip(train["wafer_map"].map(key), train["label"]))

    out = test[["wafer_map", "lot", "label"]].copy()
    out["size"] = test["wafer_map"].map(lambda m: f"{m.shape[0]}x{m.shape[1]}")
    out["pred_m1"] = [CLASSES[i] for i in pred_m1]
    out["pred_m2"] = [CLASSES[i] for i in proba_m2.argmax(1)]
    out["proba_m2"] = list(proba_m2.astype(np.float32))
    out["train_label_same_map"] = test["wafer_map"].map(lambda m: train_label_by_map.get(key(m)))
    out.to_pickle(OUT)
    print(f"[saved] {OUT} ({len(out):,}개)")


if __name__ == "__main__":
    main()
