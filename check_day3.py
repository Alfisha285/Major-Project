"""
Day 3 quick check.
Put this file in the main project folder (next to check_day2.py) and run:
    python check_day3.py
It checks that the DNN and CNN files exist, that the models load and predict,
and it prints their scores.
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

all_fine = True


def report(name, passed, detail=""):
    global all_fine
    tag = "[ok]      " if passed else "[PROBLEM] "
    print(tag + name + ("  ->  " + str(detail) if detail != "" else ""))
    if not passed:
        all_fine = False


with open(MODEL_DIR / "preprocessing" / "class_names.json") as f:
    class_names = json.load(f)
n_classes = len(class_names)
X_test = np.load(PROC / "X_test.npy")
y_test = np.load(PROC / "y_test.npy")

print("--- 1. Files ---")
for name in ["dnn", "cnn"]:
    report(name + "_model.h5 exists", (MODEL_DIR / (name + "_model.h5")).is_file())
    report(name + "_metrics.json exists", (METRIC_DIR / (name + "_metrics.json")).is_file())
    report(name + "_history.json exists", (METRIC_DIR / (name + "_history.json")).is_file())
    report(name + "_val_probs.npy exists", (PRED_DIR / (name + "_val_probs.npy")).is_file())
    report(name + "_test_probs.npy exists", (PRED_DIR / (name + "_test_probs.npy")).is_file())
    report(name + "_history.png exists", (FIG_DIR / (name + "_history.png")).is_file())
    report(name + "_confusion_test.png exists", (FIG_DIR / (name + "_confusion_test.png")).is_file())

if not all_fine:
    print("\nSome files are missing. Run the Day 3 notebook (or python src\\train_models.py) first.")
    sys.exit()

print("\n--- 2. Load the models and predict ---")
try:
    from tensorflow import keras
except Exception as e:
    print("[PROBLEM] TensorFlow could not be imported:", repr(e))
    sys.exit()

summary = {}
for name in ["dnn", "cnn"]:
    try:
        model = keras.models.load_model(MODEL_DIR / (name + "_model.h5"), compile=False)
        X = X_test[:500]
        if name == "cnn":
            X = X.reshape(X.shape[0], X.shape[1], 1)
        probs = model.predict(X, verbose=0)
        report(name + " loads and predicts", True)
        report(name + " output shape is (500, " + str(n_classes) + ")",
               probs.shape == (500, n_classes), probs.shape)
        report(name + " probabilities add up to 1", bool(np.allclose(probs.sum(axis=1), 1, atol=1e-3)))

        saved = np.load(PRED_DIR / (name + "_test_probs.npy"))[:500]
        agree = float(np.mean(np.argmax(probs, axis=1) == np.argmax(saved, axis=1)))
        report(name + " gives the same answers as the saved predictions", agree >= 0.99,
               str(round(agree * 100, 1)) + "% the same")
    except Exception as e:
        report(name + " loads and predicts", False, repr(e))
        continue

    with open(METRIC_DIR / (name + "_metrics.json")) as f:
        m = json.load(f)
    summary[name] = m

print("\n--- 3. Scores ---")
print("%-8s %-12s %10s %10s" % ("model", "", "validation", "test"))
for name, m in summary.items():
    for key in ["accuracy", "f1_weighted", "f1_macro"]:
        print("%-8s %-12s %10.4f %10.4f" % (name, key, m["validation"][key], m["test"][key]))
    report(name + " test accuracy is well above chance (10%)", m["test"]["accuracy"] > 0.3,
           round(m["test"]["accuracy"], 4))

    small = ["Worms", "Shellcode", "Backdoor", "Analysis"]
    zeros = [c for c in small if c in class_names and m["test"]["per_class"][c]["f1-score"] == 0]
    if zeros:
        print("[note]    " + name + " has F1 = 0 for: " + ", ".join(zeros)
              + " (the model never detects these)")

print()
if all_fine:
    print("ALL CHECKS PASSED. Day 3 is ready to commit.")
else:
    print("Some checks show [PROBLEM]. Copy this whole output and send it to me.")
