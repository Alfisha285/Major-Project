"""
Day 2 quick check.
Put this file in the main project folder (next to check_setup.py) and run:
    python check_day2.py
It checks the saved files, looks for NaN values, checks the value range,
and tests the preprocess_input function that Member B will use.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
ART = ROOT / "models" / "preprocessing"
sys.path.insert(0, str(ROOT / "src"))

all_fine = True


def report(name, passed, detail=""):
    global all_fine
    tag = "[ok]      " if passed else "[PROBLEM] "
    print(tag + name + ("  ->  " + str(detail) if detail != "" else ""))
    if not passed:
        all_fine = False


# 1. the saved files exist
print("--- 1. Files ---")
array_names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
for n in array_names:
    report(n + ".npy exists", (PROC / (n + ".npy")).is_file())

art_files = ["scaler.joblib", "feature_encoders.joblib", "target_encoder.joblib",
             "selected_features.json", "feature_columns.json",
             "class_names.json", "class_weights.json", "meta.json"]
for f in art_files:
    report(f + " exists", (ART / f).is_file())

if not all_fine:
    print("\nSome files are missing. Run  python src\\preprocess.py  first.")
    sys.exit()

# 2. load everything
arrays = {n: np.load(PROC / (n + ".npy")) for n in array_names}
selected = json.load(open(ART / "selected_features.json"))
class_names = json.load(open(ART / "class_names.json"))

# 3. shapes
print("\n--- 2. Shapes ---")
for n in array_names:
    print("   ", n, arrays[n].shape, arrays[n].dtype)
for part in ["train", "val", "test"]:
    X, y = arrays["X_" + part], arrays["y_" + part]
    report("X_" + part + " and y_" + part + " have the same number of rows", len(X) == len(y))
    report("X_" + part + " has " + str(len(selected)) + " columns (selected features)",
           X.shape[1] == len(selected), X.shape[1])

# 4. NaN and range
print("\n--- 3. NaN and value range ---")
for part in ["train", "val", "test"]:
    X = arrays["X_" + part]
    report("No NaN in X_" + part, not np.isnan(X).any())
    report("X_" + part + " values between 0 and 1", X.min() >= 0 and X.max() <= 1,
           "min " + str(round(float(X.min()), 4)) + ", max " + str(round(float(X.max()), 4)))

# 5. classes
print("\n--- 4. Classes ---")
n_classes = len(class_names)
for part in ["train", "val", "test"]:
    found = len(np.unique(arrays["y_" + part]))
    report("All " + str(n_classes) + " classes in y_" + part, found == n_classes, found)

print("\nTrain rows per class:")
counts = np.bincount(arrays["y_train"], minlength=n_classes)
for i, name in enumerate(class_names):
    print("   ", i, name, counts[i])

# 6. the function Member B will use
print("\n--- 5. preprocess_input (for Member B) ---")
try:
    import pandas as pd
    import preprocess as pp

    raw_test = pd.read_csv(ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv",
                           encoding="utf-8-sig")
    out = pp.preprocess_input(raw_test.head(5))
    saved = arrays["X_test"][:5]
    report("Output shape is (5, " + str(len(selected)) + ")", out.shape == (5, len(selected)), out.shape)
    report("Matches saved X_test", bool(np.allclose(out, saved)))
    one = pp.preprocess_input(raw_test.iloc[0].to_dict())
    report("Single row works", one.shape == (1, len(selected)), one.shape)
except Exception as e:
    report("preprocess_input runs", False, repr(e))

print()
if all_fine:
    print("ALL CHECKS PASSED. Day 2 is ready to commit.")
else:
    print("Some checks show [PROBLEM]. Copy this whole output and send it to me.")
