"""
Project status check: shows how much of the project is done and what is pending.

Put this file in the main project folder and run:
    python check_project.py

It only looks at your files and at Git. It does not change anything.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

phases = []          # (title, [(label, passed, optional)])


def add_phase(title, checks):
    phases.append((title, checks))


def exists(rel):
    return (ROOT / rel).exists()


def has_files(folder, patterns):
    d = ROOT / folder
    if not d.is_dir():
        return False
    return any(any(d.glob(p)) for p in patterns)


def run_git(*args):
    try:
        r = subprocess.run(["git", "-C", str(ROOT)] + list(args),
                           capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout.strip()
    except Exception:
        return None, ""


# ---------------------------------------------------------------------------
# Phase 1: setup and data exploration
# ---------------------------------------------------------------------------
py_ok = sys.version_info[:2] == (3, 13)
checks = [("Python 3.13 in the virtual environment (now %d.%d)" % sys.version_info[:2], py_ok, False)]
for pkg in ["pandas", "numpy", "sklearn", "matplotlib", "seaborn", "flask", "tensorflow", "joblib"]:
    checks.append(("package " + pkg + " installed", importlib.util.find_spec(pkg) is not None, False))
for f in ["data/raw/UNSW_NB15_training-set.csv", "data/raw/UNSW_NB15_testing-set.csv",
          "data/raw/KDDTrain+.txt", "data/raw/KDDTest+.txt"]:
    checks.append(("dataset file " + f, exists(f), False))
checks.append(("notebooks/01_data_exploration.ipynb", exists("notebooks/01_data_exploration.ipynb"), False))
checks.append(("notebooks/02_nslkdd_exploration.ipynb", exists("notebooks/02_nslkdd_exploration.ipynb"), False))

notes = ROOT / "notebooks" / "data_notes.md"
blank = None
if notes.is_file():
    lines = notes.read_text(encoding="utf-8", errors="ignore").splitlines()
    blank = sum(1 for l in lines if l.strip().startswith("-") and l.rstrip().endswith(":"))
checks.append(("notebooks/data_notes.md filled in (%s blank fields)" %
               ("?" if blank is None else blank), blank == 0, False))
add_phase("1. Setup and data exploration", checks)

# ---------------------------------------------------------------------------
# Phase 2: preprocessing
# ---------------------------------------------------------------------------
checks = [("src/preprocess.py", exists("src/preprocess.py"), False)]
for n in ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]:
    checks.append(("data/processed/" + n + ".npy", exists("data/processed/" + n + ".npy"), False))
for f in ["scaler.joblib", "feature_encoders.joblib", "target_encoder.joblib", "selected_features.json",
          "feature_columns.json", "class_names.json", "class_weights.json", "meta.json"]:
    checks.append(("models/preprocessing/" + f, exists("models/preprocessing/" + f), False))
checks.append(("check_day2.py", exists("check_day2.py"), True))
checks.append(("notebooks/03_preprocessing.ipynb", exists("notebooks/03_preprocessing.ipynb"), True))
add_phase("2. Preprocessing", checks)

# ---------------------------------------------------------------------------
# Phase 3: DNN and CNN
# ---------------------------------------------------------------------------
checks = [("src/train_models.py", exists("src/train_models.py"), False)]
for m in ["dnn", "cnn"]:
    checks += [
        ("models/%s_model.h5" % m, exists("models/%s_model.h5" % m), False),
        ("models/metrics/%s_metrics.json" % m, exists("models/metrics/%s_metrics.json" % m), False),
        ("models/predictions/%s_test_probs.npy" % m, exists("models/predictions/%s_test_probs.npy" % m), False),
        ("notebooks/figures/%s_history.png" % m, exists("notebooks/figures/%s_history.png" % m), False),
    ]
checks.append(("notebooks/04_dnn_cnn_training.ipynb", exists("notebooks/04_dnn_cnn_training.ipynb"), True))
add_phase("3. DNN and CNN", checks)

# ---------------------------------------------------------------------------
# Phase 4: LSTM and ensemble
# ---------------------------------------------------------------------------
checks = [("src/lstm_ensemble.py", exists("src/lstm_ensemble.py"), False),
          ("models/lstm_model.h5", exists("models/lstm_model.h5"), False),
          ("models/metrics/lstm_metrics.json", exists("models/metrics/lstm_metrics.json"), False),
          ("models/ensemble_config.json", exists("models/ensemble_config.json"), False),
          ("models/metrics/ensemble_metrics.json", exists("models/metrics/ensemble_metrics.json"), False),
          ("notebooks/05_lstm_ensemble.ipynb", exists("notebooks/05_lstm_ensemble.ipynb"), True)]
add_phase("4. RNN-LSTM and ensemble", checks)

# ---------------------------------------------------------------------------
# Phase 5: evaluation and report tables
# ---------------------------------------------------------------------------
checks = [("src/evaluate_all.py", exists("src/evaluate_all.py"), False)]
for f in ["comparison_test.png", "comparison_validation.png", "per_class_f1_test.png",
          "dnn_confusion_test.png", "cnn_confusion_test.png", "lstm_confusion_test.png",
          "ensemble_confusion_test.png"]:
    checks.append(("notebooks/figures/" + f, exists("notebooks/figures/" + f), False))
for f in ["results_summary.md", "results_summary.csv", "per_class_test.csv"]:
    checks.append(("reports/" + f, exists("reports/" + f), False))
add_phase("5. Final evaluation and tables", checks)

# ---------------------------------------------------------------------------
# Phase 6: web application
# ---------------------------------------------------------------------------
checks = []
for f in ["app/app.py", "app/predictor.py", "app/test_app.py", "app/static/style.css",
          "app/templates/base.html", "app/templates/index.html", "app/templates/register.html",
          "app/templates/login.html", "app/templates/predict.html", "app/templates/batch_result.html",
          "app/templates/results.html", "app/templates/error.html"]:
    checks.append((f, exists(f), False))

res = ROOT / "reports" / "app_test_results.md"
test_label = "reports/app_test_results.md (tests were run)"
test_ok = False
real_mode = False
if res.is_file():
    text = res.read_text(encoding="utf-8", errors="ignore")
    passes = text.count("| PASS |")
    fails = text.count("| FAIL |")
    real_mode = "real ensemble" in text
    test_ok = passes > 0 and fails == 0
    test_label = "web app tests: %d passed, %d failed" % (passes, fails)
checks.append((test_label, test_ok, False))
checks.append(("web app tests were run with the real model (--real)", real_mode and test_ok, False))
add_phase("6. Web application", checks)

# ---------------------------------------------------------------------------
# Phase 7: Git and GitHub
# ---------------------------------------------------------------------------
git_lines = []
checks = []
code, out = run_git("rev-parse", "--is-inside-work-tree")
if code == 0:
    code, out = run_git("symbolic-ref", "--short", "HEAD")      # works before the first commit too
if code is None:
    checks.append(("Git is installed and available", False, False))
elif code != 0:
    checks.append(("this folder is a Git repository", False, False))
else:
    branch = out
    git_lines.append("Current branch: " + branch)
    checks.append(("this folder is a Git repository", True, False))

    code, out = run_git("status", "--porcelain")
    changed = [l for l in out.splitlines() if l.strip()]
    checks.append(("no uncommitted or untracked files (%d found)" % len(changed), len(changed) == 0, False))
    if changed:
        git_lines.append("Uncommitted files (first 12):")
        git_lines += ["    " + l for l in changed[:12]]

    code, out = run_git("log", "origin/%s..HEAD" % branch, "--oneline")
    if code == 0:
        unpushed = [l for l in out.splitlines() if l.strip()]
        checks.append(("everything committed is pushed to GitHub (%d unpushed commits)" % len(unpushed),
                       len(unpushed) == 0, False))
    else:
        checks.append(("branch '%s' exists on GitHub (push it with: git push origin %s)" % (branch, branch),
                       False, False))

    code, _ = run_git("rev-parse", "--verify", "main")
    code2, _ = run_git("rev-parse", "--verify", "ml")
    if code == 0 and code2 == 0:
        merged, _ = run_git("merge-base", "--is-ancestor", "ml", "main")
        checks.append(("branch ml is merged into main", merged == 0, False))
        code, out = run_git("log", "origin/main..main", "--oneline")
        if code == 0:
            n = len([l for l in out.splitlines() if l.strip()])
            checks.append(("main is pushed to GitHub (%d unpushed commits)" % n, n == 0, False))
    else:
        checks.append(("branches main and ml both exist", False, False))

    code, out = run_git("check-ignore", "-q", "app/users.db")
    checks.append(("app/users.db is ignored by Git (keeps user data off GitHub)", code == 0, True))
add_phase("7. Git and GitHub", checks)

# ---------------------------------------------------------------------------
# Phase 8: report, presentation and demo (files you save yourself)
# ---------------------------------------------------------------------------
shots = has_files("reports/screenshots", ["*.png", "*.jpg", "*.jpeg"])
checks = [
    ("README.md in the main folder", exists("README.md"), False),
    ("final report saved in reports/ (a .docx or .pdf file)", has_files("reports", ["*.docx", "*.pdf"]), False),
    ("presentation saved in reports/ (a .pptx file)", has_files("reports", ["*.pptx"]), False),
    ("app screenshots saved in reports/screenshots/ (png or jpg)", shots, False),
]
add_phase("8. Report, presentation and screenshots", checks)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
def bar(done, total, width=20):
    filled = int(round(width * done / total)) if total else 0
    return "[" + "#" * filled + "-" * (width - filled) + "]"


print("=" * 72)
print("PROJECT STATUS: IoT Intrusion Detection (PN110)")
print("=" * 72)

grand_done = grand_total = 0
pending = []
for title, checks in phases:
    counted = [c for c in checks if not c[2]]
    done = sum(1 for c in counted if c[1])
    total = len(counted)
    grand_done += done
    grand_total += total
    pct = 100.0 * done / total if total else 100.0
    print("\n%-42s %s %3.0f%%  (%d/%d)" % (title, bar(done, total), pct, done, total))
    for label, ok, optional in checks:
        if not ok:
            tag = "[opt] " if optional else "[TODO]"
            print("    " + tag + " " + label)
            if not optional:
                pending.append((title, label))

if git_lines:
    print("\nGit details")
    for l in git_lines:
        print("    " + l)

print("\n" + "=" * 72)
pct = 100.0 * grand_done / grand_total if grand_total else 0
print("OVERALL: %s %.0f%%  (%d of %d checks)" % (bar(grand_done, grand_total, 30), pct,
                                                grand_done, grand_total))
print("=" * 72)

if pending:
    print("\nPENDING WORK (%d items):" % len(pending))
    last = None
    for title, label in pending:
        if title != last:
            print("  " + title)
            last = title
        print("      - " + label)
else:
    print("\nEverything on the checklist is done.")
print("\n[opt] items are optional and are not counted.")
