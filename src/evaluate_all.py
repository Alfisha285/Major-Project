"""
Day 5 - final evaluation of all four models (Member A).

Reads the saved metrics of the DNN, CNN, RNN-LSTM and the ensemble and creates:
  - comparison charts (validation and test)
  - a per-class F1 chart
  - confusion matrix charts for all four models
  - tables for the report (CSV and Markdown) in the reports folder

Run from the project folder:
    python src/evaluate_all.py
"""
import csv
import json
from pathlib import Path

import numpy as np

import train_models as tm

MODELS = ["dnn", "cnn", "lstm", "ensemble"]
LABELS = {"dnn": "DNN", "cnn": "CNN", "lstm": "RNN-LSTM", "ensemble": "Ensemble"}
KEYS = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "f1_macro"]
KEY_LABELS = ["Accuracy", "Precision (w)", "Recall (w)", "F1 (w)", "F1 (macro)"]
REPORT_DIR = tm.ROOT / "reports"


def load_all():
    metrics = {}
    for m in MODELS:
        path = tm.METRIC_DIR / (m + "_metrics.json")
        if path.is_file():
            with open(path) as f:
                metrics[m] = json.load(f)
    return metrics


def grouped_bar(metrics, split, path):
    import matplotlib.pyplot as plt
    names = [m for m in MODELS if m in metrics]
    x = np.arange(len(KEYS))
    width = 0.8 / len(names)
    fig = plt.figure(figsize=(10, 4.5))
    for i, m in enumerate(names):
        vals = [metrics[m][split][k] for k in KEYS]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=LABELS[m])
    plt.xticks(x, KEY_LABELS)
    plt.ylim(0, 1)
    plt.title("Model comparison on the " + split + " data")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_class_chart(metrics, class_names, path):
    import matplotlib.pyplot as plt
    names = [m for m in MODELS if m in metrics]
    x = np.arange(len(class_names))
    width = 0.8 / len(names)
    fig = plt.figure(figsize=(12, 5))
    for i, m in enumerate(names):
        vals = [metrics[m]["test"]["per_class"][c]["f1-score"] for c in class_names]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=LABELS[m])
    plt.xticks(x, class_names, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.title("F1-score per class (test file)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_tables(metrics, class_names):
    REPORT_DIR.mkdir(exist_ok=True)
    names = [m for m in MODELS if m in metrics]

    with open(REPORT_DIR / "results_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "split"] + KEYS)
        for m in names:
            for split in ("validation", "test"):
                w.writerow([LABELS[m], split] + ["%.4f" % metrics[m][split][k] for k in KEYS])

    with open(REPORT_DIR / "per_class_test.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "support"] + [LABELS[m] + " " + t for m in names
                                          for t in ("precision", "recall", "f1")])
        for c in class_names:
            support = metrics[names[0]]["test"]["per_class"][c]["support"]
            row = [c, int(support)]
            for m in names:
                r = metrics[m]["test"]["per_class"][c]
                row += ["%.3f" % r["precision"], "%.3f" % r["recall"], "%.3f" % r["f1-score"]]
            w.writerow(row)

    with open(REPORT_DIR / "results_summary.md", "w") as f:
        for split in ("test", "validation"):
            f.write("## Results on the %s data\n\n" % split)
            f.write("| Model | " + " | ".join(KEY_LABELS) + " |\n")
            f.write("|---|" + "---|" * len(KEYS) + "\n")
            for m in names:
                f.write("| %s | %s |\n" % (LABELS[m], " | ".join(
                    "%.3f" % metrics[m][split][k] for k in KEYS)))
            f.write("\n")
        f.write("## F1 per class (test)\n\n")
        f.write("| Class | " + " | ".join(LABELS[m] for m in names) + " |\n")
        f.write("|---|" + "---|" * len(names) + "\n")
        for c in class_names:
            f.write("| %s | %s |\n" % (c, " | ".join(
                "%.3f" % metrics[m]["test"]["per_class"][c]["f1-score"] for m in names)))


def main():
    metrics = load_all()
    if not metrics:
        print("No metrics found. Train the models first.")
        return
    with open(tm.ART_DIR / "class_names.json") as f:
        class_names = json.load(f)

    tm.FIG_DIR.mkdir(parents=True, exist_ok=True)
    grouped_bar(metrics, "test", tm.FIG_DIR / "comparison_test.png")
    grouped_bar(metrics, "validation", tm.FIG_DIR / "comparison_validation.png")
    per_class_chart(metrics, class_names, tm.FIG_DIR / "per_class_f1_test.png")

    import matplotlib.pyplot as plt
    for m in metrics:
        fig = tm.plot_confusion(metrics[m]["test"]["confusion_matrix"], class_names,
                                LABELS[m] + " test confusion matrix",
                                tm.FIG_DIR / (m + "_confusion_test.png"))
        plt.close(fig)

    write_tables(metrics, class_names)

    print("Models found:", ", ".join(LABELS[m] for m in metrics))
    print("%-10s" % "model", " ".join("%12s" % k for k in KEY_LABELS), "  (test)")
    for m in metrics:
        print("%-10s" % LABELS[m], " ".join("%12.4f" % metrics[m]["test"][k] for k in KEYS))
    best = max(metrics, key=lambda m: metrics[m]["test"]["f1_macro"])
    print("\nBest test macro F1:", LABELS[best])
    print("Charts saved in", tm.FIG_DIR)
    print("Tables saved in", REPORT_DIR)


if __name__ == "__main__":
    main()
