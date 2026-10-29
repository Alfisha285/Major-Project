import hashlib

from config import FEATURES, FEATURE_NAMES, CLASSES


def validate_row(row):
    """Check one input row. Returns (clean_row, error_message)."""
    clean = {}
    for f in FEATURES:
        name = f["name"]
        value = row.get(name)
        if value is None or str(value).strip() == "":
            return None, f"Missing value for '{name}'."
        if f["type"] == "number":
            try:
                clean[name] = float(value)
            except ValueError:
                return None, f"'{name}' must be a number."
        else:
            clean[name] = str(value).strip()
    return clean, None


def mock_predict(clean_row):
    """FAKE model. Gives a repeatable answer for the same input."""
    key = "|".join(str(clean_row[n]) for n in FEATURE_NAMES)
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    label = CLASSES[h % len(CLASSES)]
    confidence = round(0.80 + (h % 20) / 100, 2)
    return {"prediction": label, "confidence": confidence}


# ---- Day 5: Member A's real model gets plugged in HERE ----
# Only change this one line to swap the mock for the real model.
predict_one = mock_predict


def predict_row(row):
    """Validate then predict. Returns (result, error_message)."""
    clean, error = validate_row(row)
    if error:
        return None, error
    return predict_one(clean), None
