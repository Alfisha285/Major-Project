"""
Shows the per-class results of the trained models.
Put this file in the main project folder and run:
    python show_results.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRIC_DIR = ROOT / "models" / "metrics"

with open(ROOT / "models" / "preprocessing" / "class_names.json") as f:
    names = json.load(f)

for model in ["dnn", "cnn"]:
    path = METRIC_DIR / (model + "_metrics.json")
    if not path.is_file():
        print("No metrics file for", model)
        continue
    with open(path) as f:
        m = json.load(f)
    test, val = m["test"], m["validation"]

    print("\n" + "=" * 70)
    print(model.upper() + " - per class results on the TEST file")
    print("=" * 70)
    print("%-15s %9s %9s %9s %9s %9s" %
          ("class", "precision", "recall", "f1", "support", "val f1"))
    for c in names:
        r = test["per_class"][c]
        vf = val["per_class"][c]["f1-score"]
        print("%-15s %9.3f %9.3f %9.3f %9d %9.3f" %
              (c, r["precision"], r["recall"], r["f1-score"], r["support"], vf))

    print("\nMost common mistake for each real class (test):")
    cm = test["confusion_matrix"]
    for i, c in enumerate(names):
        row = cm[i]
        total = sum(row)
        if total == 0:
            continue
        wrong = [(row[j], names[j]) for j in range(len(names)) if j != i]
        count, other = max(wrong)
        print("%-15s %5.1f%% correct | most often mistaken for %-15s (%.1f%%)" %
              (c, 100.0 * row[i] / total, other, 100.0 * count / total))
