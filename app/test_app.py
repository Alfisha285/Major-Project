"""
Tests for the web application (the test cases from the project report).

Run from the project folder:
    python app\\test_app.py            fake model, takes a few seconds
    python app\\test_app.py --real     the real ensemble (needs TensorFlow, slower)

It prints PASS or FAIL for every test case and saves the table to
reports\\app_test_results.md so you can paste it into the report.
"""
import io
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
sys.path.insert(0, str(APP_DIR))

import app as webapp          # noqa: E402

REAL = "--real" in sys.argv
rows = []


def check(name, passed, detail=""):
    rows.append((name, "PASS" if passed else "FAIL", detail))
    print(("[PASS] " if passed else "[FAIL] ") + name + (("  ->  " + detail) if detail and not passed else ""))


def skip(name, why):
    rows.append((name, "SKIPPED", why))
    print("[SKIP] " + name + "  ->  " + why)


tmp = tempfile.mkdtemp()
db_path = str(Path(tmp) / "test_users.db")
flask_app = webapp.create_app({
    "TESTING": True,
    "DATABASE": db_path,
    "MODEL_MODE": "real" if REAL else "mock",
})
specs = flask_app.predictor.feature_specs()
class_names = flask_app.predictor.class_names
print("Mode:", "real ensemble" if REAL else "fake model", "| features:", len(specs), "\n")


def get_token(client, url):
    html = client.get(url).get_data(as_text=True)
    return re.search(r'name="csrf_token" value="([^"]+)"', html).group(1)


def register(client, username, password="secret123", confirm=None, email=""):
    return client.post("/register", data={
        "csrf_token": get_token(client, "/register"), "username": username, "email": email,
        "password": password, "confirm": password if confirm is None else confirm,
    }, follow_redirects=True)


def login(client, username, password="secret123"):
    return client.post("/login", data={
        "csrf_token": get_token(client, "/login"), "username": username, "password": password,
    }, follow_redirects=True)


def new_logged_in_client(username):
    client = flask_app.test_client()
    register(client, username)
    login(client, username)
    return client


def valid_form():
    form = {}
    for i, s in enumerate(specs):
        form[s["name"]] = s["choices"][0] if s["kind"] == "categorical" else str(0.5 + i * 0.01)
    return form


def csv_bytes(n_rows=5, columns=None):
    data = {}
    for s in specs:
        if s["kind"] == "categorical":
            data[s["name"]] = [s["choices"][0]] * n_rows
        else:
            data[s["name"]] = [0.5 + r * 0.1 for r in range(n_rows)]
    df = pd.DataFrame(data)
    if columns is not None:
        df = df[columns]
    return df.to_csv(index=False).encode()


# --- 1 and 2: registration -------------------------------------------------
c = flask_app.test_client()
html = register(c, "alice", email="alice@example.com").get_data(as_text=True)
check("Register a new user", "Account created" in html)

with sqlite3.connect(db_path) as db:
    stored = db.execute("SELECT password_hash FROM users WHERE username = 'alice'").fetchone()
ok = stored is not None and stored[0] != "secret123" and ":" in stored[0]
check("Password is stored hashed, not as plain text", ok)

html = register(flask_app.test_client(), "alice").get_data(as_text=True)
check("Register with an existing username shows an error", "already taken" in html)

html = register(flask_app.test_client(), "bob", password="123").get_data(as_text=True)
check("Register with a short password shows an error", "at least 6 characters" in html)

html = register(flask_app.test_client(), "carol", confirm="different").get_data(as_text=True)
check("Register with non-matching passwords shows an error", "do not match" in html)

# --- 3: login ----------------------------------------------------------------
c = flask_app.test_client()
html = login(c, "alice", "wrong-password").get_data(as_text=True)
check("Log in with a wrong password shows an error", "Wrong username or password" in html)
html = login(c, "alice").get_data(as_text=True)
check("Log in with the correct password works", "Welcome, alice" in html)

# --- 4: access control -------------------------------------------------------
r = flask_app.test_client().get("/predict")
check("Predict page without login redirects to the login page",
      r.status_code == 302 and "/login" in r.headers["Location"])

# --- 5: valid prediction -----------------------------------------------------
c = new_logged_in_client("dave")
form = valid_form()
form["csrf_token"] = get_token(c, "/predict")
html = c.post("/predict", data=form).get_data(as_text=True)
shown = [n for n in class_names if n in html]
check("Valid feature row gives a predicted class and confidence",
      "Confidence" in html and len(shown) > 0)

# --- 6: bad input ------------------------------------------------------------
numeric = [s["name"] for s in specs if s["kind"] == "numeric"][0]
categorical = [s["name"] for s in specs if s["kind"] == "categorical"]

form = valid_form()
form[numeric] = ""
form["csrf_token"] = get_token(c, "/predict")
html = c.post("/predict", data=form).get_data(as_text=True)
check("Missing value shows a validation error", "a value is required" in html)

form = valid_form()
form[numeric] = "abc"
form["csrf_token"] = get_token(c, "/predict")
html = c.post("/predict", data=form).get_data(as_text=True)
check("Non-numeric value shows a validation error", "must be a number" in html)

if categorical:
    form = valid_form()
    form[categorical[0]] = "not-a-real-value"
    form["csrf_token"] = get_token(c, "/predict")
    html = c.post("/predict", data=form).get_data(as_text=True)
    check("Unknown category value shows a validation error", "choose one of the listed values" in html)
else:
    skip("Unknown category value shows a validation error", "no categorical feature was selected")

# --- 7: CSV upload ---------------------------------------------------------
def upload(client, content, filename):
    return client.post("/predict/csv", data={
        "csrf_token": get_token(client, "/predict"),
        "file": (io.BytesIO(content), filename),
    }, content_type="multipart/form-data", follow_redirects=True)

html = upload(c, csv_bytes(5), "data.csv").get_data(as_text=True)
check("Upload a valid CSV gives predictions for each row", "<strong>5</strong> rows were predicted" in html)

r = c.get("/download")
lines = r.get_data(as_text=True).strip().splitlines()
check("Predictions can be downloaded as CSV",
      r.status_code == 200 and len(lines) == 6 and "prediction" in lines[0])

html = upload(c, b"some text", "data.txt").get_data(as_text=True)
check("Upload of a wrong file type shows an error", "Only .csv files are accepted" in html)

html = upload(c, b"a,b\n1,2\n", "data.csv").get_data(as_text=True)
check("Upload with wrong columns shows an error", "Missing columns" in html)

df_bad = pd.read_csv(io.BytesIO(csv_bytes(3)))
df_bad[numeric] = df_bad[numeric].astype(object)
df_bad.loc[0, numeric] = "oops"
html = upload(c, df_bad.to_csv(index=False).encode(), "data.csv").get_data(as_text=True)
check("Upload with non-numeric values shows an error", "non-numeric" in html)

html = upload(c, b"", "empty.csv").get_data(as_text=True)
check("Upload of an empty file shows an error", "could not be read" in html or "no rows" in html)

# --- 8: security and other pages -------------------------------------------
r = flask_app.test_client().post("/login", data={"username": "alice", "password": "secret123"})
check("Form without a CSRF token is rejected", r.status_code == 400)

html = c.post("/logout", data={"csrf_token": get_token(c, "/predict")},
              follow_redirects=True).get_data(as_text=True)
check("Log out works", "You are logged out" in html)

check("Home page loads", flask_app.test_client().get("/").status_code == 200)
check("Results page loads", flask_app.test_client().get("/results").status_code == 200)
check("Unknown page shows a 404 page", flask_app.test_client().get("/nothing-here").status_code == 404)

# --- 9: same prediction in the app and in the notebook ---------------------
raw_test = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"
name9 = "Same input gives the same prediction in the app and in the notebook"
if not REAL:
    skip(name9, "needs the real model (run with --real)")
elif not raw_test.is_file():
    skip(name9, "test file not found")
else:
    import preprocess as pp
    import lstm_ensemble as le
    row = pd.read_csv(raw_test, encoding="utf-8-sig", nrows=1)
    c = new_logged_in_client("erin")
    form = {s["name"]: str(row.iloc[0][s["name"]]).strip() for s in specs}
    form["csrf_token"] = get_token(c, "/predict")
    html = c.post("/predict", data=form).get_data(as_text=True)
    direct = le.load_ensemble().predict_labels(pp.preprocess_input(row))[0]
    check(name9, ("Attack detected: " + direct in html) or (direct == "Normal" and "Normal traffic" in html),
          "notebook says " + direct)

# --- summary ---------------------------------------------------------------
passed = sum(1 for r in rows if r[1] == "PASS")
failed = sum(1 for r in rows if r[1] == "FAIL")
print("\n%d passed, %d failed, %d skipped" % (passed, failed, len(rows) - passed - failed))

report_dir = ROOT / "reports"
report_dir.mkdir(exist_ok=True)
with open(report_dir / "app_test_results.md", "w") as f:
    f.write("# Web application test results\n\n")
    f.write("Mode: %s\n\n" % ("real ensemble" if REAL else "fake model (interface tests only)"))
    f.write("| Test case | Result |\n|---|---|\n")
    for name, status, detail in rows:
        f.write("| %s | %s%s |\n" % (name, status, (" (" + detail + ")") if status == "SKIPPED" else ""))
print("Saved the table to", report_dir / "app_test_results.md")
sys.exit(1 if failed else 0)
