# ------------------------------------------------------------
# THE CONTRACT between Member A (model) and Member B (app).
# When Member A sends the real feature list and class names,
# ONLY edit this file. Do not change the order after Day 3.
# ------------------------------------------------------------

# Feature names IN ORDER. type is "number" or "text".
# These are PLACEHOLDERS until Member A confirms the real ones.
FEATURES = [
    {"name": "dur",    "label": "Duration",  "type": "number"},
    {"name": "proto",  "label": "Protocol",            "type": "text"},
    {"name": "sbytes", "label": "Source Bytes",        "type": "number"},
    {"name": "dbytes", "label": "Destination Bytes",   "type": "number"},
    {"name": "spkts",  "label": "Source Packets",      "type": "number"},
]
# Class names the model can output (placeholders until Member A confirms)
CLASSES = ["Normal", "DDoS", "DoS", "Reconnaissance", "Backdoor"]

FEATURE_NAMES = [f["name"] for f in FEATURES]
