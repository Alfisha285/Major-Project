"""
Day 4 - RNN-LSTM and the soft-voting ensemble (Member A).

What this file does:
  1. builds and trains the RNN-LSTM (same data, same class weights as Day 3)
  2. saves it like the DNN and the CNN (model, metrics, probabilities, graphs)
  3. builds the soft-voting ensemble: it averages the probabilities of the
     DNN, the CNN and the LSTM and picks the class with the highest average
  4. gives Member B one object (EnsembleModel) that loads the three models
     and predicts, so the web app needs only a few lines

It uses the helper functions from train_models.py (same src folder).

Train the LSTM and build the ensemble from the project folder with:
    python src/lstm_ensemble.py
or use the step by step notebook: notebooks/05_lstm_ensemble.ipynb
"""
import json

import numpy as np

import train_models as tm

MODEL_NAMES = ["dnn", "cnn", "lstm"]
LSTM_EPOCHS = 30


# ------------------------------------------------------------------
# RNN-LSTM
# ------------------------------------------------------------------
def build_lstm(n_features, n_classes):
    """The 30 features are read one after another like a short sequence."""
    from tensorflow import keras
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="lstm")
    tm.compile_model(model)
    return model


def run_lstm(epochs=LSTM_EPOCHS, batch_size=tm.BATCH_SIZE, weight_mode=tm.WEIGHT_MODE):
    data, class_names, raw_weights = tm.load_data()
    n_features = data["X_train"].shape[1]
    n_classes = len(class_names)
    class_weight = tm.make_class_weights(raw_weights, weight_mode)

    tm.set_seed()
    model = build_lstm(n_features, n_classes)

    Xtr = tm.reshape_for_cnn(data["X_train"])
    Xva = tm.reshape_for_cnn(data["X_val"])
    Xte = tm.reshape_for_cnn(data["X_test"])

    history = tm.train_model(model, Xtr, data["y_train"], Xva, data["y_val"],
                             class_weight, epochs, batch_size)

    val_probs = tm.predict_probs(model, Xva)
    test_probs = tm.predict_probs(model, Xte)
    metrics_val = tm.compute_metrics(data["y_val"], val_probs, class_names)
    metrics_test = tm.compute_metrics(data["y_test"], test_probs, class_names)

    tm.save_results("lstm", model, history, val_probs, test_probs,
                    metrics_val, metrics_test, class_names)
    tm.print_summary("lstm", metrics_val, metrics_test)
    return metrics_val, metrics_test


# ------------------------------------------------------------------
# Soft voting
# ------------------------------------------------------------------
def load_saved_probs(split, names=MODEL_NAMES):
    return {n: np.load(tm.PRED_DIR / (n + "_" + split + "_probs.npy")) for n in names}


def soft_vote(prob_list, weights=None):
    """Average the probabilities of several models (optionally with weights)."""
    probs = np.stack(prob_list)                     # (models, rows, classes)
    if weights is None:
        weights = np.ones(len(prob_list))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return np.tensordot(w, probs, axes=1)           # (rows, classes)


def run_ensemble(weights=None, names=MODEL_NAMES):
    data, class_names, _ = tm.load_data()
    val_probs = soft_vote(list(load_saved_probs("val", names).values()), weights)
    test_probs = soft_vote(list(load_saved_probs("test", names).values()), weights)

    metrics_val = tm.compute_metrics(data["y_val"], val_probs, class_names)
    metrics_test = tm.compute_metrics(data["y_test"], test_probs, class_names)

    for d in (tm.METRIC_DIR, tm.PRED_DIR, tm.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    np.save(tm.PRED_DIR / "ensemble_val_probs.npy", val_probs.astype("float32"))
    np.save(tm.PRED_DIR / "ensemble_test_probs.npy", test_probs.astype("float32"))
    tm.save_json({"model": "ensemble", "members": list(names),
                  "validation": metrics_val, "test": metrics_test},
                 tm.METRIC_DIR / "ensemble_metrics.json")

    w = [1.0] * len(names) if weights is None else [float(x) for x in weights]
    tm.save_json({"models": list(names), "weights": w, "class_names": class_names},
                 tm.MODEL_DIR / "ensemble_config.json")

    tm.plot_confusion(metrics_test["confusion_matrix"], class_names,
                      "ENSEMBLE test confusion matrix",
                      tm.FIG_DIR / "ensemble_confusion_test.png")

    tm.print_summary("ensemble", metrics_val, metrics_test)
    return metrics_val, metrics_test


# ------------------------------------------------------------------
# For Member B: one object that predicts
# ------------------------------------------------------------------
class EnsembleModel:
    """
    Usage in the web app:
        ens = load_ensemble()
        x = preprocess_input(form_data)        # shape (1, 30)
        probs = ens.predict_proba(x)           # shape (1, 10)
        name = ens.predict_labels(x)[0]        # for example "Exploits"
    """

    def __init__(self, models, weights, class_names):
        self.models = models
        self.weights = weights
        self.class_names = class_names

    def predict_proba(self, X):
        X = np.asarray(X, dtype="float32")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probs = []
        for name, model in self.models.items():
            X_in = X if name == "dnn" else tm.reshape_for_cnn(X)
            probs.append(model.predict(X_in, verbose=0))
        return soft_vote(probs, self.weights)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_labels(self, X):
        return [self.class_names[i] for i in self.predict(X)]


def _keras_loader(path):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)


def load_ensemble(model_dir=None, loader=None):
    model_dir = tm.MODEL_DIR if model_dir is None else model_dir
    loader = _keras_loader if loader is None else loader
    with open(model_dir / "ensemble_config.json") as f:
        config = json.load(f)
    models = {n: loader(model_dir / (n + "_model.h5")) for n in config["models"]}
    return EnsembleModel(models, config["weights"], config["class_names"])


if __name__ == "__main__":
    run_lstm()
    run_ensemble()
    print("\nDone. Models are in", tm.MODEL_DIR)
