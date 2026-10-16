"""최종 평가 (PROTOCOL.md 5~7절). 평가 데이터를 여는 유일한 파일이다.

- M0, M1: 고정한 설정으로 학습 데이터 전체를 학습해 평가 데이터를 예측한다.
- M2: 커밋해 둔 models/cnn_final.pt로 예측한다. LightGBM과 충돌하지 않도록 scripts/predict_cnn.py를 별도 프로세스로 실행한다.
- 판정은 확률이 가장 높은 클래스. 매크로 F1, 클래스별 지표, 혼동행렬, 새 크기·중복 제외 부분집합, 로트 단위 부트스트랩 구간.
결과는 reports/final/에 저장한다. 이미 결과가 있으면 다시 실행하지 않는다.

실행 (wafer-map/ 폴더에서): python scripts/final_evaluation.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASSES, UNLOCK_PHRASE, load_test, load_train  # noqa: E402
from src.features import shape_features, size_features  # noqa: E402
from src.models import build_m0, build_m1  # noqa: E402
from src.preprocess import stack_maps  # noqa: E402

OUT_DIR = ROOT / "reports" / "final"
FIG_DIR = ROOT / "reports" / "figures"
TMP_DIR = ROOT / "data" / "processed"
MODELS = ["M0", "M1", "M2"]
NAMES = {"M0": "M0 크기 기준선", "M1": "M1 특징 모델", "M2": "M2 CNN"}
K = len(CLASSES)
N_BOOT = 1000


# ---------------- 지표: 혼동행렬에서 계산 ----------------

def confusion(y_true, y_pred) -> np.ndarray:
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class(cm: np.ndarray) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(tp, predicted, out=np.zeros(K), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(K), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(K), where=(precision + recall) > 0)
    return pd.DataFrame({"정밀도": precision, "재현율": recall, "F1": f1, "개수": support}, index=CLASSES)


def macro_f1(cm: np.ndarray) -> tuple[float, int]:
    """그 표본에 실제로 있는 클래스만 평균한다 (PROTOCOL.md 6절). (매크로 F1, 사용한 클래스 수)"""
    table = per_class(cm)
    present = table["개수"] > 0
    return float(table.loc[present, "F1"].mean()), int(present.sum())


# ---------------- 로트 단위 부트스트랩 ----------------

def lot_bootstrap(y, preds: dict, lots: np.ndarray, seed: int = 0) -> dict:
    """로트를 복원추출해 매크로 F1과 모델 간 차이의 95% 구간을 구한다.

    같은 로트 웨이퍼끼리 비슷해서 웨이퍼 단위로 뽑으면 구간이 실제보다 좁아진다.
    로트마다 혼동행렬을 미리 만들어 두고, 뽑힌 로트의 행렬을 더해서 빠르게 계산한다.
    """
    lot_ids, lot_idx = np.unique(lots, return_inverse=True)
    n_lots = len(lot_ids)
    per_lot = {m: np.zeros((n_lots, K, K), dtype=np.int64) for m in preds}
    for m, p in preds.items():
        np.add.at(per_lot[m], (lot_idx, y, p), 1)

    rng = np.random.default_rng(seed)
    samples = {m: [] for m in preds}
    for _ in range(N_BOOT):
        counts = np.bincount(rng.integers(0, n_lots, n_lots), minlength=n_lots)
        for m in preds:
            samples[m].append(macro_f1(np.tensordot(counts, per_lot[m], axes=1))[0])
    samples = {m: np.array(v) for m, v in samples.items()}

    def ci(a):
        return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    out = {f"{m}": ci(samples[m]) for m in preds}
    for a, b in [("M1", "M0"), ("M2", "M0"), ("M2", "M1")]:
        out[f"{a}-{b}"] = ci(samples[a] - samples[b])
    return out


# ---------------- 예측 ----------------

def predict_all(train: pd.DataFrame, test: pd.DataFrame, decisions: dict, cnn_weights: Path) -> tuple[dict, np.ndarray]:
    y_train = train["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    img_train, img_test = stack_maps(train["wafer_map"]), stack_maps(test["wafer_map"])

    m0 = build_m0().fit(size_features(train["wafer_map"]), y_train)
    m1 = build_m1(**decisions["m1_params"]).fit(shape_features(img_train), y_train)
    preds = {
        "M0": m0.predict(size_features(test["wafer_map"])),
        "M1": m1.predict(shape_features(img_test)),
    }

    src, dst = TMP_DIR / "eval_images.npy", TMP_DIR / "eval_cnn_proba.npy"
    np.save(src, img_test)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "predict_cnn.py"), str(src), str(dst), str(cnn_weights)],
                   check=True)
    preds["M2"] = np.load(dst).argmax(1)
    src.unlink(); dst.unlink()
    return {m: np.asarray(p, dtype=np.int64) for m, p in preds.items()}, y_train


def subsets(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """새 크기: 원래 맵 크기가 학습 데이터에 없던 웨이퍼. 중복 제외: 학습 데이터와 칸 값까지 같은 맵을 뺀 웨이퍼."""
    train_sizes = set(train["wafer_map"].map(lambda m: m.shape))
    new_size = ~test["wafer_map"].map(lambda m: m.shape).isin(train_sizes).to_numpy()

    def key(m):
        return m.shape, m.astype(np.int8).tobytes()
    train_keys = set(train["wafer_map"].map(key))
    duplicate = test["wafer_map"].map(key).isin(train_keys).to_numpy()
    return {"전체": np.ones(len(test), dtype=bool), "새 크기": new_size, "중복 제외": ~duplicate}


# ---------------- 보고 ----------------

def evaluate(train, test, decisions, cnn_weights, out_dir: Path, fig_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, _ = predict_all(train, test, decisions, cnn_weights)
    y = test["label"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    lots = test["lot"].to_numpy()
    masks = subsets(train, test)

    # 부분집합별 매크로 F1
    rows = []
    for name, mask in masks.items():
        row = {"부분집합": name, "웨이퍼 수": int(mask.sum()), "로트 수": int(pd.Series(lots[mask]).nunique())}
        for m in MODELS:
            f1, n_cls = macro_f1(confusion(y[mask], preds[m][mask]))
            row[NAMES[m]] = f1
        row["클래스 수"] = n_cls
        rows.append(row)
    subset_table = pd.DataFrame(rows).set_index("부분집합")
    subset_table.to_csv(out_dir / "macro_f1_by_subset.csv", encoding="utf-8-sig")

    # 클래스별 지표, 혼동행렬
    cms = {m: confusion(y, preds[m]) for m in MODELS}
    per_cls = pd.concat({NAMES[m]: per_class(cms[m]) for m in MODELS}, axis=1)
    per_cls.to_csv(out_dir / "per_class.csv", encoding="utf-8-sig")
    accuracy = {m: float((preds[m] == y).mean()) for m in MODELS}

    # 로트 단위 부트스트랩 (전체 평가 데이터)
    boot = lot_bootstrap(y, preds, lots)
    (out_dir / "bootstrap.json").write_text(json.dumps(boot, indent=2, ensure_ascii=False))

    plot(cms, per_cls, subset_table, fig_dir)
    return report(subset_table, per_cls, accuracy, boot, int(len(y)), int(pd.Series(lots).nunique()))


def plot(cms, per_cls, subset_table, fig_dir: Path) -> None:
    sns.set_theme(style="whitegrid")
    labels = CLASSES

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, m in zip(axes, MODELS):
        norm = cms[m] / np.maximum(cms[m].sum(1, keepdims=True), 1)
        sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels, annot_kws={"size": 7})
        ax.set_title({"M0": "M0 size only", "M1": "M1 shape features", "M2": "M2 CNN"}[m])
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.suptitle("Test confusion matrices (row-normalized: share of each actual pattern)", y=1.0)
    fig.tight_layout(); fig.savefig(fig_dir / "08_confusion_matrices.png", dpi=150); plt.close(fig)

    f1 = per_cls.xs("F1", axis=1, level=1)
    f1.columns = ["M0 size only", "M1 shape features", "M2 CNN"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    f1.plot.bar(ax=ax, color=["#BBBBBB", "#4C72B0", "#DD8452"], width=0.8, rot=30)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1"); ax.set_title("Per-pattern F1 on test")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "09_test_per_class.png", dpi=150); plt.close(fig)

    st = subset_table[[NAMES[m] for m in MODELS]]
    st.columns = ["M0 size only", "M1 shape features", "M2 CNN"]
    st.index = [{"전체": "all test", "새 크기": "unseen map sizes", "중복 제외": "duplicates removed"}[i] for i in st.index]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    st.plot.bar(ax=ax, color=["#BBBBBB", "#4C72B0", "#DD8452"], width=0.8, rot=0)
    ax.set_ylim(0, 1.05); ax.set_ylabel("macro F1"); ax.set_title("Macro F1 by test subset")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "10_test_subsets.png", dpi=150); plt.close(fig)


def verdict(lo: float, hi: float) -> str:
    """PROTOCOL.md 7절 해석 규칙."""
    if lo > 0:
        return "앞 모델이 낫다"
    if hi < 0:
        return "앞 모델이 낮다"
    return "우열을 단정할 수 없다"


def report(subset_table, per_cls, accuracy, boot, n_test, n_lots) -> str:
    st = subset_table.copy()
    lines = [
        "# 최종 평가 결과", "",
        f"- 평가 데이터: {n_test:,}개 웨이퍼, {n_lots:,}개 로트",
        "- 판정: 9개 클래스 중 예측 확률이 가장 높은 클래스",
        "", "## 매크로 F1", "",
        "| 모델 | 전체 [95% 구간, 로트 단위] | 새 크기 | 중복 제외 | 정확도(참고) |",
        "|---|---|---|---|---|",
    ]
    for m in MODELS:
        lo, hi = boot[m]
        lines.append(f"| {NAMES[m]} | {st.loc['전체', NAMES[m]]:.3f} [{lo:.3f}, {hi:.3f}] | "
                     f"{st.loc['새 크기', NAMES[m]]:.3f} | {st.loc['중복 제외', NAMES[m]]:.3f} | {accuracy[m]:.3f} |")
    lines += ["",
              f"부분집합 크기: 새 크기 {int(st.loc['새 크기', '웨이퍼 수']):,}개({int(st.loc['새 크기', '로트 수']):,} 로트, "
              f"{int(st.loc['새 크기', '클래스 수'])}개 클래스), 중복 제외 {int(st.loc['중복 제외', '웨이퍼 수']):,}개",
              "", "## 모델 간 차이 (계획서 7절 규칙)", "",
              "| 비교 | 매크로 F1 차이 | 95% 구간 | 해석 |", "|---|---|---|---|"]
    for pair in ["M1-M0", "M2-M0", "M2-M1"]:
        a, b = pair.split("-")
        diff = st.loc["전체", NAMES[a]] - st.loc["전체", NAMES[b]]
        lo, hi = boot[pair]
        lines.append(f"| {NAMES[a]} − {NAMES[b]} | {diff:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict(lo, hi)} |")
    f1 = per_cls.xs("F1", axis=1, level=1).round(3)
    f1["개수"] = per_cls[(NAMES["M0"], "개수")].astype(int)
    lines += ["", "## 클래스별 F1", "", f1.to_markdown(), ""]
    return "\n".join(lines)


def git_commit() -> str:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", ".")
    return run("rev-parse", "--short", "HEAD") + (" (커밋되지 않은 변경 있음)" if dirty else "")


def main() -> None:
    if (OUT_DIR / "final_evaluation.md").exists() and "--force" not in sys.argv:
        sys.exit("이미 최종 평가를 실행했다. 결과는 reports/final/에 있다.")
    decisions = json.loads((ROOT / "reports" / "decisions.json").read_text())
    header = f"- 실행 시각: {datetime.now():%Y-%m-%d %H:%M}\n- 코드 버전: `{git_commit()}`\n"

    train = load_train()
    test = load_test(unlock=UNLOCK_PHRASE)
    body = evaluate(train, test, decisions, ROOT / "models" / "cnn_final.pt", OUT_DIR, FIG_DIR)

    text = body.replace("# 최종 평가 결과\n\n", "# 최종 평가 결과\n\n" + header, 1)
    (OUT_DIR / "final_evaluation.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
