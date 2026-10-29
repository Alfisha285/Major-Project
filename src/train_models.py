"""
Day 3 - train the DNN and the CNN (Member A).

What this file does:
  1. load the processed data from Day 2
  2. prepare the class weights (so small classes are not ignored)
  3. build the DNN and the CNN
  4. train them with early stopping
  5. measure accuracy, precision, recall, F1 and the confusion matrix
  6. save the models (.h5), the metrics (.json), the predicted
     probabilities (.npy, used for the ensemble on Day 4) and the graphs

Run both models from the project folder with:
    python src/train_models.py
or use the step by step notebook: notebooks/04_dnn_cnn_training.ipynb
"""
import json
import random
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------
# Settings you can change
# ------------------------------------------------------------------
SEED = 42
EPOCHS = 40              # maximum epochs (early stopping usually stops sooner)
BATCH_SIZE = 256
WEIGHT_MODE = "sqrt"     # "balanced", "sqrt" or "none"  (explained in the notebook)

# ------------------------------------------------------------------
# Folders
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "data" / "processed"
ART_DIR = ROOT / "models" / "preprocessing"
MODEL_DIR = ROOT / "models"
METRIC_DIR = MODEL_DIR / "metrics"
PRED_DIR = MODEL_DIR / "predictions"
FIG_DIR = ROOT / "notebooks" / "figures"


# ------------------------------------------------------------------
# Step 1: load
# ------------------------------------------------------------------
def load_data():
    names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]
    data = {n: np.load(PROC_DIR / (n + ".npy")) for n in names}

    with open(ART_DIR / "class_names.json") as f:
        class_names = json.load(f)
    with open(ART_DIR / "class_weights.json") as f:
        raw_weights = {int(k): float(v) for k, v in json.load(f).items()}

    return data, class_names, raw_weights


# ------------------------------------------------------------------
# Step 2: class weights
# ------------------------------------------------------------------
def make_class_weights(raw_weights, mode=WEIGHT_MODE):
    if mode == "none":
        return None
    if mode == "balanced":
        return dict(raw_weights)
    if mode == "sqrt":
        # a softer version: square root of the balanced weights
        return {k: float(np.sqrt(v)) for k, v in raw_weights.items()}
    raise ValueError("mode must be 'balanced', 'sqrt' or 'none'")


def set_seed(seed=SEED):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ------------------------------------------------------------------
# Step 3: build the models
# ------------------------------------------------------------------
def compile_model(model):
    from tensorflow import keras
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_dnn(n_features, n_classes):
    """Deep Neural Network: a stack of fully connected layers."""
    from tensorflow import keras
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ], name="dnn")
    compile_model(model)
    return model


def build_cnn(n_features, n_classes):
    """1D Convolutional Neural Network: looks for patterns across neighbouring features."""
    from tensorflow import keras
    layers = keras.layers
    model = keras.Sequential([
        layers.Input(shape=(n_features, 1)),
        layers.Conv1D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation="softmax"),
    ], name="cnn")
    compile_model(model)
    return model


def reshape_for_cnn(X):
    # (rows, 30) -> (rows, 30, 1): the CNN wants one "channel"
    return X.reshape(X.shape[0], X.shape[1], 1)


def keep_shape(X):
    return X


# ------------------------------------------------------------------
# Step 4: train
# ------------------------------------------------------------------
def train_model(model, X_train, y_train, X_val, y_val,
                class_weight=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
    from tensorflow import keras
    callbacks = [
        # stop when the validation loss has not improved for 6 epochs
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                      restore_best_weights=True),
        # make the learning rate smaller when progress slows down
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    return history.history


# ------------------------------------------------------------------
# Step 5: measure
# ------------------------------------------------------------------
def predict_probs(model, X, batch_size=1024):
    return model.predict(X, batch_size=batch_size, verbose=0)


def compute_metrics(y_true, probs, class_names):
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, precision_recall_fscore_support)
    y_pred = np.argmax(probs, axis=1)
    labels = list(range(len(class_names)))

    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    _, _, f_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "f1_macro": float(f_m),
        "per_class": report,
        "confusion_matrix": cm.tolist(),
    }


def print_summary(name, metrics_val, metrics_test):
    print("\n==== " + name.upper() + " ====")
    print("%-12s %10s %10s" % ("", "validation", "test"))
    for key in ["accuracy", "precision_weighted", "recall_weighted",
                "f1_weighted", "f1_macro"]:
        print("%-20s %8.4f %10.4f" % (key, metrics_val[key], metrics_test[key]))


# ------------------------------------------------------------------
# Graphs
# ------------------------------------------------------------------
def plot_history(history, title, path):
    import matplotlib
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title(title + ": loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title(title + ": accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


def plot_confusion(cm, class_names, title, path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cm = np.array(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums            # each row adds up to 1

    fig = plt.figure(figsize=(9, 7))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title + " (each row = 100% of that real class)")
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    return fig


# ------------------------------------------------------------------
# Step 6: save
# ------------------------------------------------------------------
def _json_default(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def save_results(name, model, history, val_probs, test_probs,
                 metrics_val, metrics_test, class_names):
    for d in (MODEL_DIR, METRIC_DIR, PRED_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_DIR / (name + "_model.h5"))

    np.save(PRED_DIR / (name + "_val_probs.npy"), val_probs.astype("float32"))
    np.save(PRED_DIR / (name + "_test_probs.npy"), test_probs.astype("float32"))

    save_json({
        "model": name,
        "epochs_run": len(history["loss"]),
        "validation": metrics_val,
        "test": metrics_test,
    }, METRIC_DIR / (name + "_metrics.json"))
    save_json(history, METRIC_DIR / (name + "_history.json"))

    plot_history(history, name.upper(), FIG_DIR / (name + "_history.png"))
    plot_confusion(metrics_test["confusion_matrix"], class_names,
                   name.upper() + " test confusion matrix",
                   FIG_DIR / (name + "_confusion_test.png"))


# ------------------------------------------------------------------
# Run everything for one model
# ------------------------------------------------------------------
def run_training(name, epochs=EPOCHS, batch_size=BATCH_SIZE, weight_mode=WEIGHT_MODE):
    data, class_names, raw_weights = load_data()
    n_features = data["X_train"].shape[1]
    n_classes = len(class_names)
    class_weight = make_class_weights(raw_weights, weight_mode)

    set_seed()
    if name == "dnn":
        model = build_dnn(n_features, n_classes)
        prep = keep_shape
    elif name == "cnn":
        model = build_cnn(n_features, n_classes)
        prep = reshape_for_cnn
    else:
        raise ValueError("name must be 'dnn' or 'cnn'")

    history = train_model(model, prep(data["X_train"]), data["y_train"],
                          prep(data["X_val"]), data["y_val"],
                          class_weight, epochs, batch_size)

    val_probs = predict_probs(model, prep(data["X_val"]))
    test_probs = predict_probs(model, prep(data["X_test"]))
    metrics_val = compute_metrics(data["y_val"], val_probs, class_names)
    metrics_test = compute_metrics(data["y_test"], test_probs, class_names)

    save_results(name, model, history, val_probs, test_probs,
                 metrics_val, metrics_test, class_names)
    print_summary(name, metrics_val, metrics_test)
    return metrics_val, metrics_test


if __name__ == "__main__":
    for model_name in ["dnn", "cnn"]:
        run_training(model_name)
    print("\nDone. Models are in", MODEL_DIR)
