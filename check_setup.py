"""Quick check that the setup and the dataset files are fine.

Run it from the project folder:  python check_setup.py
"""
import os
import sys

print("Python version:", sys.version.split()[0])

# 1. check packages
packages = ["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "flask"]
for name in packages:
    try:
        __import__(name)
        print("[ok]      package", name)
    except ImportError:
        print("[MISSING] package", name, "-> run: pip install -r requirements.txt")

# 2. check folders
for folder in ["data/raw", "data/processed", "models", "notebooks", "app"]:
    if os.path.isdir(folder):
        print("[ok]      folder", folder)
    else:
        print("[MISSING] folder", folder)

# 3. check dataset files
files = [
    "data/raw/UNSW_NB15_training-set.csv",
    "data/raw/UNSW_NB15_testing-set.csv",
    "data/raw/KDDTrain+.txt",
    "data/raw/KDDTest+.txt",
]
for path in files:
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print("[ok]      file", path, "(%.1f MB)" % size_mb)
    else:
        print("[MISSING] file", path)

print("\nIf everything says [ok], you are ready to open the notebooks.")
