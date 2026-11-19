"""
Prediction layer for the web app.

It connects the Flask app to the work from Days 2 to 4:
  - preprocess_input (src/preprocess.py) turns raw values into the 30 numbers
  - the ensemble (src/lstm_ensemble.py) turns those numbers into probabilities

ONE LINE TO SWITCH between the real model and a fake one (for testing):
    MODEL_MODE = "real"   uses the trained DNN + CNN + LSTM ensemble
    MODEL_MODE = "mock"   gives fake predictions, no TensorFlow needed
You can also set it from the terminal:  set MODEL_MODE=mock
"""
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(ROOT / "src"))

import preprocess as pp          # noqa: E402
import lstm_ensemble as le       # noqa: E402

MODEL_MODE = os.environ.get("MODEL_MODE", "real")     # <- the one line
MAX_CSV_ROWS = 20000


class MockEnsemble:
    """Fake model: always gives the same answer for the same input."""

    def __init__(self, n_classes):
        self.n = n_classes

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        rows = len(X)
        idx = (np.abs(X.sum(axis=1)) * 1000).astype(int) % self.n
        probs = np.full((rows, self.n), 0.3 / (self.n - 1))
        probs[np.arange(rows), idx] = 0.7
        return probs


class Predictor:
    def __init__(self, mode=None):
        self.mode = MODEL_MODE if mode is None else mode
        self.artifacts = pp.load_artifacts()
        self.class_names = self.artifacts["class_names"]
        self.selected = self.artifacts["selected_features"]
        self._model = None

    # ---- the model is loaded the first time it is needed -------------
    @property
    def model(self):
        if self._model is None:
            if self.mode == "mock":
                self._model = MockEnsemble(len(self.class_names))
            else:
                self._model = le.load_ensemble()
        return self._model

    # ---- form fields for the web page --------------------------------
    def feature_specs(self):
        specs = []
        for name in self.selected:
            if name in pp.CATEGORICAL_COLS:
                choices = [str(c) for c in self.artifacts["encoders"][name].classes_]
                specs.append({"name": name, "kind": "categorical", "choices": choices})
            else:
                specs.append({"name": name, "kind": "numeric", "choices": None})
        return specs

    # ---- checking user input -----------------------------------------
    def validate_form(self, form):
        row, errors = {}, []
        for spec in self.feature_specs():
            name = spec["name"]
            raw = (form.get(name) or "").strip()
            if raw == "":
                errors.append(name + ": a value is required")
                continue
            if spec["kind"] == "categorical":
                if raw not in spec["choices"]:
                    errors.append(name + ": choose one of the listed values")
                    continue
                row[name] = raw
            else:
                try:
                    value = float(raw)
                except ValueError:
                    errors.append(name + ": must be a number")
                    continue
                if not math.isfinite(value):
                    errors.append(name + ": must be a finite number")
                    continue
                row[name] = value
        return row, errors

    def validate_csv(self, df):
        errors, warnings = [], []
        if len(df) == 0:
            errors.append("The file has no rows.")
            return errors, warnings
        if len(df) > MAX_CSV_ROWS:
            errors.append("The file has %d rows. The limit is %d." % (len(df), MAX_CSV_ROWS))
            return errors, warnings

        missing = [c for c in self.selected if c not in df.columns]
        if missing:
            errors.append("Missing columns: " + ", ".join(missing))
            return errors, warnings

        for spec in self.feature_specs():
            name = spec["name"]
            if spec["kind"] == "numeric":
                bad = int(pd.to_numeric(df[name], errors="coerce").isna().sum())
                if bad:
                    errors.append("%s: %d missing or non-numeric values" % (name, bad))
            else:
                unseen = int((~df[name].astype(str).str.strip().isin(spec["choices"])).sum())
                if unseen:
                    warnings.append("%s: %d rows have values not seen in training "
                                    "(treated as unknown)" % (name, unseen))
        return errors, warnings

    # ---- predicting ----------------------------------------------------
    def predict_df(self, df):
        X = pp.preprocess_input(df, self.artifacts)
        probs = self.model.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        out = pd.DataFrame({
            "prediction": [self.class_names[i] for i in idx],
            "confidence": probs.max(axis=1),
        })
        return out, probs

    def predict_row(self, row):
        out, probs = self.predict_df(pd.DataFrame([row]))
        ranked = sorted(zip(self.class_names, probs[0]), key=lambda t: -t[1])
        return {
            "prediction": out.loc[0, "prediction"],
            "confidence": float(out.loc[0, "confidence"]),
            "ranked": [(n, float(p)) for n, p in ranked],
        }
