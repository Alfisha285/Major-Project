"""
IoT Intrusion Detection - Flask web application.

Pages:  home, register, login, logout, predict (one row or CSV), results.
Run from the project folder:   python app\\app.py
Then open:                     http://127.0.0.1:5000
"""
import io
import json
import os
import random
import re
import secrets
import sqlite3
from functools import wraps
from pathlib import Path

import pandas as pd
from flask import (Flask, abort, flash, g, redirect, render_template, request,
                   send_file, send_from_directory, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import predictor as pred

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
FIG_DIR = ROOT / "notebooks" / "figures"
METRIC_DIR = ROOT / "models" / "metrics"
RAW_TEST = ROOT / "data" / "raw" / "UNSW_NB15_testing-set.csv"

MODEL_LABELS = [("dnn", "DNN"), ("cnn", "CNN"), ("lstm", "RNN-LSTM"), ("ensemble", "Ensemble")]
METRIC_KEYS = [("accuracy", "Accuracy"), ("precision_weighted", "Precision"),
               ("recall_weighted", "Recall"), ("f1_weighted", "F1 (weighted)"),
               ("f1_macro", "F1 (macro)")]


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-key-change-before-sharing"),
        DATABASE=str(APP_DIR / "users.db"),
        MAX_CONTENT_LENGTH=20 * 1024 * 1024,
        SESSION_COOKIE_SAMESITE="Lax",
        MODEL_MODE=pred.MODEL_MODE,
    )
    if test_config:
        app.config.update(test_config)

    os.makedirs(app.instance_path, exist_ok=True)
    app.predictor = pred.Predictor(mode=app.config["MODEL_MODE"])
    app.sample_cache = {}

    # ------------------------------------------------------------------
    # database
    # ------------------------------------------------------------------
    def get_db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"])
            g.db.row_factory = sqlite3.Row
        return g.db

    @app.teardown_appcontext
    def close_db(error=None):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    with app.app_context():
        get_db().execute(
            """CREATE TABLE IF NOT EXISTS users (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   username TEXT UNIQUE NOT NULL,
                   email TEXT,
                   password_hash TEXT NOT NULL,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
        get_db().commit()

    # ------------------------------------------------------------------
    # login and CSRF helpers
    # ------------------------------------------------------------------
    def login_required(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in first.", "error")
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)
        return wrapped

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def check_csrf():
        if request.method == "POST":
            sent = request.form.get("csrf_token", "")
            if not sent or sent != session.get("csrf"):
                abort(400, "The form has expired. Go back, reload the page and try again.")

    def safe_next(target):
        if target and target.startswith("/") and not target.startswith("//"):
            return target
        return url_for("index")

    # ------------------------------------------------------------------
    # pages
    # ------------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html", classes=app.predictor.class_names,
                               n_features=len(app.predictor.selected),
                               mode=app.config["MODEL_MODE"])

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            email = request.form.get("email", "").strip()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm", "")
            errors = []
            if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
                errors.append("Username must be 3 to 30 letters, digits or underscores.")
            if email and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                errors.append("That email address does not look right.")
            if len(password) < 6:
                errors.append("Password must be at least 6 characters.")
            if password != confirm:
                errors.append("The two passwords do not match.")
            if not errors:
                db = get_db()
                if db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone():
                    errors.append("That username is already taken.")
                else:
                    db.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                               (username, email, generate_password_hash(password)))
                    db.commit()
                    flash("Account created. You can log in now.", "success")
                    return redirect(url_for("login"))
            for e in errors:
                flash(e, "error")
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = get_db().execute("SELECT * FROM users WHERE username = ?",
                                    (username,)).fetchone()
            if user is None or not check_password_hash(user["password_hash"], password):
                flash("Wrong username or password.", "error")
            else:
                session.clear()
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                flash("Welcome, " + user["username"] + ".", "success")
                return redirect(safe_next(request.args.get("next")))
        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        flash("You are logged out.", "success")
        return redirect(url_for("index"))

    # ------------------------------------------------------------------
    # prediction
    # ------------------------------------------------------------------
    def load_sample():
        """One random row from the official test file (only for the demo button)."""
        if not RAW_TEST.is_file():
            return None, None
        if "df" not in app.sample_cache:
            app.sample_cache["df"] = pd.read_csv(RAW_TEST, encoding="utf-8-sig")
        df = app.sample_cache["df"]
        row = df.iloc[random.randrange(len(df))]
        values = {s["name"]: str(row[s["name"]]).strip() for s in app.predictor.feature_specs()
                  if s["name"] in df.columns}
        return values, str(row.get("attack_cat", "")).strip()

    def model_note():
        path = METRIC_DIR / "ensemble_metrics.json"
        if path.is_file():
            with open(path) as f:
                return json.load(f)["test"]["accuracy"]
        return None

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        specs = app.predictor.feature_specs()
        values, actual, result = {}, "", None

        if request.method == "GET" and request.args.get("sample"):
            values, actual = load_sample()
            if values is None:
                flash("The test file was not found, so no sample can be loaded.", "error")
                values, actual = {}, ""

        if request.method == "POST":
            values = {s["name"]: request.form.get(s["name"], "") for s in specs}
            actual = request.form.get("actual", "")
            row, errors = app.predictor.validate_form(request.form)
            if errors:
                for e in errors:
                    flash(e, "error")
            else:
                try:
                    result = app.predictor.predict_row(row)
                except Exception as exc:               # model could not be loaded or run
                    flash("The model could not make a prediction: " + str(exc)[:200], "error")

        return render_template("predict.html", specs=specs, values=values, actual=actual,
                               result=result, test_accuracy=model_note(),
                               mode=app.config["MODEL_MODE"])

    @app.route("/predict/csv", methods=["POST"])
    @login_required
    def predict_csv():
        file = request.files.get("file")
        if file is None or file.filename == "":
            flash("Please choose a CSV file.", "error")
            return redirect(url_for("predict"))
        if not file.filename.lower().endswith(".csv"):
            flash("Only .csv files are accepted.", "error")
            return redirect(url_for("predict"))
        try:
            df = pd.read_csv(io.BytesIO(file.read()), encoding="utf-8-sig")
        except Exception:
            flash("The file could not be read as a CSV.", "error")
            return redirect(url_for("predict"))

        errors, warnings = app.predictor.validate_csv(df)
        if errors:
            for e in errors:
                flash(e, "error")
            return redirect(url_for("predict"))
        for w in warnings:
            flash(w, "warning")

        try:
            out, _ = app.predictor.predict_df(df)
        except Exception as exc:
            flash("The model could not make predictions: " + str(exc)[:200], "error")
            return redirect(url_for("predict"))

        full = df.copy()
        full["prediction"] = out["prediction"].values
        full["confidence"] = out["confidence"].round(4).values
        full.to_csv(Path(app.instance_path) / ("predictions_%s.csv" % session["user_id"]),
                    index=False)

        counts = out["prediction"].value_counts()
        summary = [(name, int(n), 100.0 * n / len(out)) for name, n in counts.items()]

        shown = pd.DataFrame({"row": range(1, len(out) + 1),
                              "prediction": out["prediction"], "confidence": out["confidence"]})
        accuracy = None
        if "attack_cat" in df.columns:
            actual = df["attack_cat"].astype(str).str.strip().values
            shown["actual"] = actual
            accuracy = float((out["prediction"].values == actual).mean())
        rows = shown.head(100).to_dict("records")

        return render_template("batch_result.html", n_rows=len(out), summary=summary,
                               rows=rows, accuracy=accuracy, has_actual="actual" in shown.columns)

    @app.route("/download")
    @login_required
    def download():
        path = Path(app.instance_path) / ("predictions_%s.csv" % session["user_id"])
        if not path.is_file():
            flash("No predictions to download yet. Upload a CSV first.", "error")
            return redirect(url_for("predict"))
        return send_file(path, as_attachment=True, download_name="predictions.csv")

    # ------------------------------------------------------------------
    # results (model comparison)
    # ------------------------------------------------------------------
    @app.route("/results")
    def results():
        metrics = {}
        for key, _ in MODEL_LABELS:
            path = METRIC_DIR / (key + "_metrics.json")
            if path.is_file():
                with open(path) as f:
                    metrics[key] = json.load(f)

        tables = {}
        for split in ("test", "validation"):
            rows = []
            for key, label in MODEL_LABELS:
                if key in metrics:
                    rows.append((label, [metrics[key][split][k] for k, _ in METRIC_KEYS]))
            tables[split] = rows

        per_class = []
        for cls in app.predictor.class_names:
            cells = []
            for key, _ in MODEL_LABELS:
                if key in metrics:
                    cells.append(metrics[key]["test"]["per_class"][cls]["f1-score"])
                else:
                    cells.append(None)
            per_class.append((cls, cells))

        wanted = ["model_comparison_test.png", "comparison_test.png", "comparison_validation.png",
                  "per_class_f1_test.png", "ensemble_confusion_test.png", "cnn_confusion_test.png",
                  "dnn_confusion_test.png", "lstm_confusion_test.png",
                  "dnn_history.png", "cnn_history.png", "lstm_history.png"]
        figures = [f for f in wanted if (FIG_DIR / f).is_file()]

        return render_template("results.html", tables=tables, per_class=per_class,
                               metric_labels=[l for _, l in METRIC_KEYS],
                               model_labels=[l for k, l in MODEL_LABELS if k in metrics],
                               figures=figures, have_results=bool(metrics))

    @app.route("/figures/<path:filename>")
    def figure(filename):
        if not filename.lower().endswith(".png"):
            abort(404)
        return send_from_directory(FIG_DIR, filename)

    # ------------------------------------------------------------------
    # errors
    # ------------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(e):
        return render_template("error.html", code=400, message=e.description), 400

    @app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", code=404, message="That page does not exist."), 404

    @app.errorhandler(413)
    def too_large(e):
        return render_template("error.html", code=413,
                               message="That file is too large. The limit is 20 MB."), 413

    @app.errorhandler(500)
    def server_error(e):
        return render_template("error.html", code=500,
                               message="Something went wrong on the server."), 500

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
