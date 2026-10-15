"""최종 평가 (PROTOCOL.md 6~8절). 평가 데이터를 여는 유일한 파일이다.

reports/best_params.json에 고정한 설정으로 학습 데이터 전체를 학습하고, 평가 데이터를 한 번 예측한다.
결과는 reports/final/에 저장한다. 이미 결과가 있으면 다시 실행하지 않는다.

실행 (secom/ 폴더에서): python scripts/final_evaluation.py
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import UNLOCK_PHRASE, load_test, load_train  # noqa: E402
from src.evaluate import (INSPECT_RATE, bootstrap_ci, bootstrap_diff_ci, classification_summary,  # noqa: E402
                          cutoff_for_rate, recall_at_rate)
from src.models import build_from_params  # noqa: E402

OUT_DIR = ROOT / "reports" / "final"
FIG_DIR = ROOT / "reports" / "figures"
LABELS = {"logistic": "로지스틱 회귀", "lightgbm": "LightGBM"}
PLOT_LABELS = {"logistic": "Logistic", "lightgbm": "LightGBM"}
GAIN_RATES = [0.1, 0.2, 0.3, 0.5]


def git_state() -> dict:
    """어떤 코드 버전으로 평가했는지 남긴다. 커밋되지 않은 변경이 있으면 기록에 표시된다."""
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "uncommitted_changes": run("status", "--porcelain", "--", ".")}


def evaluate(decisions: dict, X_train, y_train, X_test, y_test) -> dict:
    """고정된 설정으로 학습하고 평가 데이터 지표를 계산한다. 결과를 보고 바꾸는 단계는 없다."""
    base_rate = float(y_test.mean())
    proba, rows, cms = {}, [], {}

    for name, d in decisions.items():
        model = build_from_params(name, d["params"]).fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        proba[name] = p

        # 판정 규칙: 계획서 5절에서 학습 데이터로 정한 방식 (두 모델 모두 B: 평가 기간 전체를 한 묶음으로 상위 20%)
        if d["rule"] != "B_batch_quota":
            raise ValueError(f"계획서에 없는 판정 방식: {d['rule']}")
        cutoff = cutoff_for_rate(p, INSPECT_RATE)
        summary = classification_summary(y_test, p, cutoff)
        cms[name] = confusion_matrix(y_test, (p >= cutoff).astype(int), labels=[0, 1])

        pr = bootstrap_ci(y_test, p, average_precision_score)
        rc = bootstrap_ci(y_test, p, lambda yt, pp: recall_at_rate(yt, pp, INSPECT_RATE))
        rows.append({
            "모델": LABELS[name],
            "PR-AUC": pr[0], "PR-AUC 하한": pr[1], "PR-AUC 상한": pr[2],
            "재현율(상위 20%)": rc[0], "재현율 하한": rc[1], "재현율 상한": rc[2],
            "정밀도": summary["정밀도"], "ROC-AUC": roc_auc_score(y_test, p),
            "정확도(참고)": summary["정확도(참고)"], "불량 판정 비율": summary["불량 판정 비율"],
            "TP": summary["TP"], "FN": summary["FN"], "FP": summary["FP"], "TN": summary["TN"],
        })

    metrics = pd.DataFrame(rows).set_index("모델")
    gain = pd.DataFrame({LABELS[n]: [recall_at_rate(y_test, p, r) for r in GAIN_RATES] for n, p in proba.items()},
                        index=[f"상위 {int(r*100)}%" for r in GAIN_RATES])
    gain["무작위"] = GAIN_RATES
    diff = bootstrap_diff_ci(y_test, proba["logistic"], proba["lightgbm"], average_precision_score)
    return {"base_rate": base_rate, "metrics": metrics, "gain": gain, "diff": diff, "proba": proba, "cms": cms,
            "n_test": len(y_test), "n_fail": int(y_test.sum())}


def interpret(result: dict) -> list[str]:
    """계획서 7절의 해석 규칙을 그대로 적용한 문장."""
    lines, base = [], result["base_rate"]
    for model, r in result["metrics"].iterrows():
        pr_ok = r["PR-AUC 하한"] > base
        rc_ok = r["재현율 하한"] > INSPECT_RATE
        lines.append(
            f"- {model}: PR-AUC 구간 하한 {r['PR-AUC 하한']:.3f}이 무작위 기준선 {base:.3f}보다 "
            + ("높다 → 무작위보다 낫다." if pr_ok else "높지 않다 → 무작위보다 낫다고 단정할 수 없다.")
            + f" 상위 20% 검사 재현율 구간 하한 {r['재현율 하한']:.2f}은 0.20보다 "
            + ("높다." if rc_ok else "높지 않다.")
        )
    d = result["diff"]
    lines.append(
        f"- PR-AUC 차이(로지스틱 - LightGBM) {d[0]:+.3f}, 95% 구간 [{d[1]:+.3f}, {d[2]:+.3f}] → "
        + ("0을 포함한다 → 우열을 단정할 수 없다." if d[1] <= 0 <= d[2] else "0을 포함하지 않는다.")
    )
    return lines


def plot(result: dict) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, cm) in zip(axes, result["cms"].items()):
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["pred pass", "pred fail"], yticklabels=["actual pass", "actual fail"])
        ax.set_title(f"{PLOT_LABELS[name]}  (caught {cm[1, 1]}/{cm[1].sum()})")
    fig.suptitle("Test confusion matrix: top 20% risk flagged", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_confusion_matrix.png", dpi=150)
    plt.close(fig)

    rates = np.linspace(0.01, 1, 100)
    y_test = result["y_test"]
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, p in result["proba"].items():
        ax.plot(rates, [recall_at_rate(y_test, p, r) for r in rates], label=PLOT_LABELS[name])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random")
    ax.axvline(INSPECT_RATE, color="gray", ls=":", lw=1)
    ax.set_xlabel("share of products inspected (highest risk first)")
    ax.set_ylabel("share of defects caught")
    ax.set_title("Cumulative gain on test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_gain_curve.png", dpi=150)
    plt.close(fig)


def random_split_comparison(decisions: dict, X_all, y_all, n_seeds: int = 5) -> pd.DataFrame:
    """계획서 8절 사후 분석: 같은 설정으로 층화 무작위 분할했을 때의 PR-AUC. 결정을 바꾸지 않는다."""
    rows = {}
    for name, d in decisions.items():
        scores = []
        for seed in range(n_seeds):
            Xa, Xb, ya, yb = train_test_split(X_all, y_all, test_size=0.25, stratify=y_all, random_state=seed)
            m = build_from_params(name, d["params"]).fit(Xa, ya)
            scores.append(average_precision_score(yb, m.predict_proba(Xb)[:, 1]))
        rows[LABELS[name]] = {"무작위 분할 평균": np.mean(scores), "무작위 분할 표준편차": np.std(scores)}
    return pd.DataFrame(rows).T


def write_report(result: dict, split_cmp: pd.DataFrame, state: dict) -> str:
    m = result["metrics"]
    lines = [
        "# 최종 평가 결과",
        "",
        f"- 실행 시각: {datetime.now():%Y-%m-%d %H:%M}",
        f"- 코드 버전: `{state['commit'][:7]}`" + (" (커밋되지 않은 변경 있음)" if state["uncommitted_changes"] else ""),
        f"- 평가 데이터: {result['n_test']}행, 불량 {result['n_fail']}개 (무작위 PR-AUC 기준선 {result['base_rate']:.3f})",
        "- 판정: 평가 기간 전체에서 위험도 상위 20%를 불량으로 판정",
        "",
        "## 주요 지표",
        "",
        "| 모델 | PR-AUC [95% 구간] | 재현율 상위 20% [95% 구간] | 정밀도 | ROC-AUC | 잡은 불량 |",
        "|---|---|---|---|---|---|",
        f"| 무작위 | {result['base_rate']:.3f} | 0.20 | {result['base_rate']:.3f} | 0.5 | 기대값 {0.2 * result['n_fail']:.1f}개 |",
    ]
    for model, r in m.iterrows():
        lines.append(
            f"| {model} | {r['PR-AUC']:.3f} [{r['PR-AUC 하한']:.3f}, {r['PR-AUC 상한']:.3f}] "
            f"| {r['재현율(상위 20%)']:.2f} [{r['재현율 하한']:.2f}, {r['재현율 상한']:.2f}] "
            f"| {r['정밀도']:.3f} | {r['ROC-AUC']:.3f} | {int(r['TP'])}/{int(r['TP'] + r['FN'])} |"
        )
    lines += ["", "## 계획서 7절 규칙에 따른 해석", ""] + interpret(result)
    lines += ["", "## 검사 비율별 재현율", "", result["gain"].round(3).to_markdown()]
    lines += ["", "## 사후 분석: 무작위 분할과 비교 (결정에 쓰지 않음)", "",
              split_cmp.assign(**{"시간순 분할": m["PR-AUC"]}).round(3).to_markdown()]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if (OUT_DIR / "metrics.csv").exists() and "--force" not in sys.argv:
        sys.exit("이미 최종 평가를 실행했다. 결과는 reports/final/에 있다.")

    decisions = json.loads((ROOT / "reports" / "best_params.json").read_text())
    state = git_state()

    X_train, y_train, _ = load_train()
    X_test, y_test, _ = load_test(unlock=UNLOCK_PHRASE)

    result = evaluate(decisions, X_train, y_train, X_test, y_test)
    result["y_test"] = y_test.to_numpy()
    result["metrics"].to_csv(OUT_DIR / "metrics.csv", encoding="utf-8-sig")
    result["gain"].to_csv(OUT_DIR / "gain.csv", encoding="utf-8-sig")
    plot(result)

    X_all, y_all = pd.concat([X_train, X_test]), pd.concat([y_train, y_test])
    split_cmp = random_split_comparison(decisions, X_all, y_all)
    split_cmp.to_csv(OUT_DIR / "random_split_comparison.csv", encoding="utf-8-sig")

    report = write_report(result, split_cmp, state)
    (OUT_DIR / "final_evaluation.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
