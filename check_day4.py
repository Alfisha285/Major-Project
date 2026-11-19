"""
Day 4 quick check.
Put this file in the main project folder (next to check_day3.py) and run:
    python check_day4.py
It checks the LSTM and ensemble files, loads the ensemble, and prints the
scores of all four models.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
METRIC_DIR = MODEL_DIR / "metrics"
PRED_DIR = MODEL_DIR / "predictions"
FIG_DIR = ROOT / "notebooks" / "figures"
sys.path.insert(0, str(ROOT / "src"))

all_fine = True


def report(name, passed, detail=""):
    global all_fine
    tag = "[ok]      " if passed else "[PROBLEM] "
    print(tag + name + ("  ->  " + str(detail) if detail != "" else ""))
    if not passed:
        all_fine = False


print("--- 1. Files ---")
for name in ["lstm"]:
    report(name + "_model.h5 exists", (MODEL_DIR / (name + "_model.h5")).is_file())
    report(name + "_metrics.json exists", (METRIC_DIR / (name + "_metrics.json")).is_file())
    report(name + "_val_probs.npy exists", (PRED_DIR / (name + "_val_probs.npy")).is_file())
    report(name + "_test_probs.npy exists", (PRED_DIR / (name + "_test_probs.npy")).is_file())
    report(name + "_history.png exists", (FIG_DIR / (name + "_history.png")).is_file())
    report(name + "_confusion_test.png exists", (FIG_DIR / (name + "_confusion_test.png")).is_file())
report("ensemble_config.json exists", (MODEL_DIR / "ensemble_config.json").is_file())
report("ensemble_metrics.json exists", (METRIC_DIR / "ensemble_metrics.json").is_file())
report("ensemble_test_probs.npy exists", (PRED_DIR / "ensemble_test_probs.npy").is_file())
report("ensemble_confusion_test.png exists", (FIG_DIR / "ensemble_confusion_test.png").is_file())
for name in ["dnn", "cnn"]:
    report(name + "_model.h5 exists (from Day 3)", (MODEL_DIR / (name + "_model.h5")).is_file())

if not all_fine:
    print("\nSome files are missing. Run the Day 4 notebook (or python src\\lstm_ensemble.py) first.")
    sys.exit()

with open(MODEL_DIR / "preprocessing" / "class_names.json") as f:
    class_names = json.load(f)
n_classes = len(class_names)
X_test = np.load(PROC / "X_test.npy")

print("\n--- 2. Load the ensemble and predict ---")
try:
    import lstm_ensemble as le
    ens = le.load_ensemble()
    X = X_test[:500]
    probs = ens.predict_proba(X)
    report("ensemble loads and predicts", True)
    report("output shape is (500, " + str(n_classes) + ")", probs.shape == (500, n_classes), probs.shape)
    report("probabilities add up to 1", bool(np.allclose(probs.sum(axis=1), 1, atol=1e-3)))
    saved = np.load(PRED_DIR / "ensemble_test_probs.npy")[:500]
    agree = float(np.mean(np.argmax(probs, axis=1) == np.argmax(saved, axis=1)))
    report("same answers as the saved ensemble predictions", agree >= 0.99,
           str(round(agree * 100, 1)) + "% the same")
    labels = ens.predict_labels(X_test[:3])
    print("    example predictions:", labels)
except Exception as e:
    report("ensemble loads and predicts", False, repr(e))
    print("\nSome checks show [PROBLEM]. Copy this whole output and send it to me.")
    sys.exit()

print("\n--- 3. Scores of the four models ---")
scores = {}
for name in ["dnn", "cnn", "lstm", "ensemble"]:
    with open(METRIC_DIR / (name + "_metrics.json")) as f:
        scores[name] = json.load(f)

print("%-10s %-12s %10s %10s" % ("model", "", "validation", "test"))
for name, m in scores.items():
    for key in ["accuracy", "f1_weighted", "f1_macro"]:
        print("%-10s %-12s %10.4f %10.4f" % (name, key, m["validation"][key], m["test"][key]))

best_single = max(["dnn", "cnn", "lstm"], key=lambda n: scores[n]["test"]["accuracy"])
best_acc = scores[best_single]["test"]["accuracy"]
ens_acc = scores["ensemble"]["test"]["accuracy"]
print("\nBest single model on test accuracy:", best_single, round(best_acc, 4))
print("Ensemble on test accuracy:         ", round(ens_acc, 4))
report("ensemble is not clearly worse than the best single model", ens_acc >= best_acc - 0.01,
       "difference " + str(round(ens_acc - best_acc, 4)))

print()
if all_fine:
    print("ALL CHECKS PASSED. Day 4 is ready to commit.")
else:
    print("Some checks show [PROBLEM]. Copy this whole output and send it to me.")
