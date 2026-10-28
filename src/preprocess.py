"""
Day 2 - preprocessing pipeline for UNSW-NB15 (Member A).

What this file does, in order:
  1. load the official train and test csv files
  2. clean the data (drop id, duplicates, inf / missing values)
  3. choose the target: multi-class (attack_cat) or binary (label)
  4. split the training file into train and validation
  5. label-encode the text columns (proto, service, state)
  6. normalize the numbers with MinMaxScaler
  7. pick the best features with Mutual Information
  8. calculate class weights (and optionally SMOTE) for the imbalance
  9. save everything to disk so Member B never touches the raw data

Run the whole thing from the project folder with:
    python src/preprocess.py
or use the step by step notebook: notebooks/03_preprocessing.ipynb
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------
# Settings you can change
# ------------------------------------------------------------------
MODE = "multiclass"      # "multiclass" (attack_cat) or "binary" (label)
TOP_K = 30               # how many features to keep after Mutual Information
VAL_SIZE = 0.15          # part of the training file kept for validation
USE_SMOTE = False        # True needs: pip install imbalanced-learn (makes data bigger)
RANDOM_STATE = 42

CATEGORICAL_COLS = ["proto", "service", "state"]
DROP_COLS = ["id"]
TARGET_MULTI = "attack_cat"
TARGET_BINARY = "label"

# ------------------------------------------------------------------
# Folders
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "models" / "preprocessing"
FIG_DIR = ROOT / "notebooks" / "figures"


# ------------------------------------------------------------------
# Step 1: load
# ------------------------------------------------------------------
def load_raw(raw_dir=RAW_DIR):
    train_path = Path(raw_dir) / "UNSW_NB15_training-set.csv"
    test_path = Path(raw_dir) / "UNSW_NB15_testing-set.csv"
    train_full = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    return train_full, test


# ------------------------------------------------------------------
# Step 2: clean
# ------------------------------------------------------------------
def basic_clean(df, drop_duplicates=False):
    df = df.copy()

    # remove columns we do not need
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    # tidy the text columns
    for col in CATEGORICAL_COLS + [TARGET_MULTI]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # infinite values become missing, then missing values become 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print("Duplicates removed:", before - len(df))

    return df


# ------------------------------------------------------------------
# Step 3: features (X) and target (y)
# ------------------------------------------------------------------
def make_xy(df, mode=MODE):
    if mode == "multiclass":
        y = df[TARGET_MULTI].astype(str)
    elif mode == "binary":
        y = df[TARGET_BINARY].map({0: "Normal", 1: "Attack"})
    else:
        raise ValueError("mode must be 'multiclass' or 'binary'")

    X = df.drop(columns=[TARGET_MULTI, TARGET_BINARY])
    return X, y


# ------------------------------------------------------------------
# Step 4: train / validation split
# ------------------------------------------------------------------
def split_train_val(X, y, val_size=VAL_SIZE):
    try:
        return train_test_split(X, y, test_size=val_size,
                                stratify=y, random_state=RANDOM_STATE)
    except ValueError:
        # happens only if a class has just 1 row
        print("Warning: stratified split not possible, using a normal split")
        return train_test_split(X, y, test_size=val_size, random_state=RANDOM_STATE)


# ------------------------------------------------------------------
# Step 5: label encoding
# ------------------------------------------------------------------
def fit_encoders(X_train, cols=CATEGORICAL_COLS):
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        le.fit(X_train[col].astype(str))
        encoders[col] = le
    return encoders


def apply_encoders(X, encoders):
    X = X.copy()
    for col, le in encoders.items():
        mapping = {name: i for i, name in enumerate(le.classes_)}
        # a value never seen in training gets -1
        X[col] = X[col].astype(str).map(mapping).fillna(-1).astype(int)
    return X


def encode_target(y_train, y_val, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    return (le, le.transform(y_train), le.transform(y_val), le.transform(y_test))


# ------------------------------------------------------------------
# Step 6: normalization
# ------------------------------------------------------------------
def fit_scaler(X_train):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(X, scaler):
    scaled = scaler.transform(X)
    # test data can fall slightly outside the training range
    scaled = np.clip(scaled, 0, 1)
    return pd.DataFrame(scaled, columns=X.columns, index=X.index)


# ------------------------------------------------------------------
# Step 7: Mutual Information feature selection
# ------------------------------------------------------------------
def mutual_info_ranking(X_train_scaled, y_train, X_train_encoded=None, sample_size=20000):
    # The text columns are whole numbers (0, 1, 2 ...) before scaling.
    # Mutual Information treats them as categories, so we put the whole
    # numbers back for those columns only.
    if X_train_encoded is not None:
        X_train_scaled = X_train_scaled.copy()
        for col in CATEGORICAL_COLS:
            X_train_scaled[col] = X_train_encoded[col].values

    # Mutual Information is slow on 150k rows, so use a random sample
    if len(X_train_scaled) > sample_size:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_train_scaled), sample_size, replace=False)
        X_s = X_train_scaled.iloc[idx]
        y_s = np.asarray(y_train)[idx]
    else:
        X_s = X_train_scaled
        y_s = np.asarray(y_train)

    discrete = [c in CATEGORICAL_COLS for c in X_s.columns]
    scores = mutual_info_classif(X_s, y_s, discrete_features=discrete,
                                 random_state=RANDOM_STATE)
    ranking = pd.Series(scores, index=X_s.columns).sort_values(ascending=False)
    return ranking


def pick_features(ranking, all_columns, top_k=TOP_K):
    top = set(ranking.head(top_k).index)
    # keep the original column order so the model input is stable
    return [c for c in all_columns if c in top]


# ------------------------------------------------------------------
# Step 8: class imbalance
# ------------------------------------------------------------------
def get_class_weights(y_train_encoded):
    classes = np.unique(y_train_encoded)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_encoded)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def apply_smote(X_train, y_train):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("Install it first: pip install imbalanced-learn")
    sm = SMOTE(random_state=RANDOM_STATE)
    return sm.fit_resample(X_train, y_train)


# ------------------------------------------------------------------
# Step 9: save
# ------------------------------------------------------------------
def save_everything(arrays, artifacts, meta):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    for name, arr in arrays.items():
        np.save(PROCESSED_DIR / (name + ".npy"), arr)

    joblib.dump(artifacts["encoders"], ARTIFACT_DIR / "feature_encoders.joblib")
    joblib.dump(artifacts["target_encoder"], ARTIFACT_DIR / "target_encoder.joblib")
    joblib.dump(artifacts["scaler"], ARTIFACT_DIR / "scaler.joblib")

    with open(ARTIFACT_DIR / "feature_columns.json", "w") as f:
        json.dump(artifacts["feature_columns"], f, indent=2)
    with open(ARTIFACT_DIR / "selected_features.json", "w") as f:
        json.dump(artifacts["selected_features"], f, indent=2)
    with open(ARTIFACT_DIR / "class_names.json", "w") as f:
        json.dump(artifacts["class_names"], f, indent=2)
    with open(ARTIFACT_DIR / "class_weights.json", "w") as f:
        json.dump(artifacts["class_weights"], f, indent=2)
    with open(ARTIFACT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------------
# Run everything at once
# ------------------------------------------------------------------
def run_pipeline(mode=MODE, top_k=TOP_K, use_smote=USE_SMOTE, save=True):
    print("Mode:", mode, "| top features:", top_k)

    train_full, test = load_raw()
    train_full = basic_clean(train_full, drop_duplicates=True)
    test = basic_clean(test, drop_duplicates=False)

    X_full, y_full = make_xy(train_full, mode)
    X_test, y_test = make_xy(test, mode)
    feature_columns = list(X_full.columns)

    X_train, X_val, y_train, y_val = split_train_val(X_full, y_full)
    print("Train / val / test rows:", len(X_train), len(X_val), len(X_test))

    encoders = fit_encoders(X_train)
    X_train = apply_encoders(X_train, encoders)
    X_val = apply_encoders(X_val, encoders)
    X_test = apply_encoders(X_test, encoders)

    target_encoder, y_train_e, y_val_e, y_test_e = encode_target(y_train, y_val, y_test)
    class_names = list(target_encoder.classes_)

    scaler = fit_scaler(X_train)
    X_train_s = apply_scaler(X_train, scaler)
    X_val_s = apply_scaler(X_val, scaler)
    X_test_s = apply_scaler(X_test, scaler)

    ranking = mutual_info_ranking(X_train_s, y_train_e, X_train_encoded=X_train)
    selected = pick_features(ranking, feature_columns, top_k)
    print("Selected", len(selected), "features")

    Xtr = X_train_s[selected].values.astype("float32")
    Xva = X_val_s[selected].values.astype("float32")
    Xte = X_test_s[selected].values.astype("float32")

    class_weights = get_class_weights(y_train_e)
    ytr = y_train_e
    if use_smote:
        Xtr, ytr = apply_smote(Xtr, ytr)
        print("After SMOTE, train rows:", len(Xtr))

    arrays = {
        "X_train": Xtr, "y_train": ytr,
        "X_val": Xva, "y_val": y_val_e,
        "X_test": Xte, "y_test": y_test_e,
    }
    artifacts = {
        "encoders": encoders, "target_encoder": target_encoder, "scaler": scaler,
        "feature_columns": feature_columns, "selected_features": selected,
        "class_names": class_names, "class_weights": class_weights,
    }
    meta = {
        "mode": mode, "top_k": top_k, "use_smote": use_smote,
        "n_features": len(selected), "n_classes": len(class_names),
        "train_rows": int(len(Xtr)), "val_rows": int(len(Xva)), "test_rows": int(len(Xte)),
    }
    if save:
        save_everything(arrays, artifacts, meta)
        ranking.to_csv(ARTIFACT_DIR / "mutual_info_ranking.csv", header=["mi_score"])
        print("Saved to", PROCESSED_DIR, "and", ARTIFACT_DIR)
    return arrays, artifacts, meta, ranking


# ------------------------------------------------------------------
# For Member B: turn raw input into model-ready numbers
# ------------------------------------------------------------------
def load_artifacts(artifact_dir=ARTIFACT_DIR):
    artifact_dir = Path(artifact_dir)
    return {
        "encoders": joblib.load(artifact_dir / "feature_encoders.joblib"),
        "target_encoder": joblib.load(artifact_dir / "target_encoder.joblib"),
        "scaler": joblib.load(artifact_dir / "scaler.joblib"),
        "feature_columns": json.load(open(artifact_dir / "feature_columns.json")),
        "selected_features": json.load(open(artifact_dir / "selected_features.json")),
        "class_names": json.load(open(artifact_dir / "class_names.json")),
    }


def preprocess_input(data, artifacts=None):
    """
    data      : a dict (one row) or a DataFrame with the raw column names
                (proto, service, state, dur, sbytes, ...).
    returns   : numpy array of shape (rows, number_of_selected_features)
    Missing columns are filled with 0 and extra columns are ignored.
    """
    if artifacts is None:
        artifacts = load_artifacts()

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    cols = artifacts["feature_columns"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    df = df[cols]

    df = df.replace([np.inf, -np.inf], np.nan)
    for c in cols:
        if c not in CATEGORICAL_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(0)

    df = apply_encoders(df, artifacts["encoders"])
    df = apply_scaler(df, artifacts["scaler"])
    return df[artifacts["selected_features"]].values.astype("float32")


if __name__ == "__main__":
    run_pipeline()
