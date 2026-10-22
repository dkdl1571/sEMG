"""main.py - dataset.npz 로 여러 모델을 학습·평가하고 결과(표, Confusion Matrix)를 results/ 에 저장

실행:  python main.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

SEED = 42
OUT = "results"


def load_dataset(path="dataset.npz"):
    if not os.path.exists(path):          # dataset.npz 가 없으면 원신호로부터 새로 만든다
        import data
        data.build_dataset(path)
    d = np.load(path, allow_pickle=True)
    classes = [str(c) for c in d["classes"]] if "classes" in d.files else \
              [chr(ord("A") + i) for i in range(int(d["ytr"].max()) + 1)]
    return d["Xtr"], d["ytr"], d["Xte"], d["yte"], classes


def get_models():
    return {
        "SVM (RBF)":     SVC(kernel="rbf", C=1.0, gamma="scale", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
        "k-NN (k=5)":    KNeighborsClassifier(n_neighbors=5),
    }


def evaluate(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": p, "Recall": r, "F1-score": f1}


def save_confusion_matrix(cm, classes, name, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(cm, classes):
    """Confusion Matrix 에서 분석 재료를 뽑는다 (가장 잘/못 분류된 클래스, 최다 오분류 쌍)."""
    recall = cm.diagonal() / cm.sum(axis=1)
    best, worst = int(np.argmax(recall)), int(np.argmin(recall))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    return {"best": (classes[best], recall[best]), "worst": (classes[worst], recall[worst]),
            "top_pair": (classes[i], classes[j], int(off[i, j])), "recall": recall}


def main():
    os.makedirs(OUT, exist_ok=True)
    Xtr, ytr, Xte, yte, classes = load_dataset()
    print(f"학습 {Xtr.shape}, 테스트 {Xte.shape}, 클래스 {classes}\n")

    rows, details = [], {}
    for name, model in get_models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        m = evaluate(yte, pred)
        cm = confusion_matrix(yte, pred)
        a = analyze(cm, classes)
        rows.append((name, m))
        details[name] = (cm, a)
        fname = name.split()[0].lower().replace("-", "") + "_cm.png"
        save_confusion_matrix(cm, classes, name, os.path.join(OUT, fname))
        print(f"{name:14s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))

    # ── metrics.csv ──
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Model", "Accuracy", "Precision", "Recall", "F1-score"])
        for name, m in rows:
            w.writerow([name] + [f"{m[k]:.4f}" for k in ("Accuracy", "Precision", "Recall", "F1-score")])

    # ── README 에 붙여넣을 초안 (results/report_draft.md) ──
    best_model = max(rows, key=lambda r: r[1]["F1-score"])[0]
    worst_model = min(rows, key=lambda r: r[1]["F1-score"])[0]
    L = ["## 모델 성능 비교 (Precision/Recall/F1 = macro 평균)\n",
         "| Model | Accuracy | Precision | Recall | F1-score |", "|---|---|---|---|---|"]
    for name, m in rows:
        L.append(f"| {name} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1-score']:.4f} |")
    L += ["", f"- F1-score 가장 높은 모델: **{best_model}**", f"- F1-score 가장 낮은 모델: **{worst_model}**", ""]
    for name, (cm, a) in details.items():
        fname = name.split()[0].lower().replace("-", "") + "_cm.png"
        L += [f"### {name}", f"![{name}](results/{fname})", "",
              f"- 가장 잘 분류된 클래스: {a['best'][0]} (재현율 {a['best'][1]:.3f})",
              f"- 가장 많이 오분류된 클래스: {a['worst'][0]} (재현율 {a['worst'][1]:.3f})",
              f"- 주요 오분류 유형: 실제 {a['top_pair'][0]} → {a['top_pair'][1]}로 예측 ({a['top_pair'][2]}개)",
              "- 오분류가 발생한 이유: (직접 작성)", ""]
    with open(os.path.join(OUT, "report_draft.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"\n저장 완료: {OUT}/metrics.csv, {OUT}/report_draft.md, {OUT}/*_cm.png")


if __name__ == "__main__":
    main()
