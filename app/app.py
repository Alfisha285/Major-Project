import json
import os
import sqlite3
from functools import wraps

import pandas as pd
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from flask import jsonify

from config import FEATURES, FEATURE_NAMES
from model_service import predict_row

app = Flask(__name__)
app.secret_key = "change-this-to-something-random"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
DB_PATH = "users.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            flash("Please log in first.", "error")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"]

        if not username or not email or len(password) < 6:
            flash("Fill all fields. Password must be at least 6 characters.", "error")
            return render_template("register.html")

        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, generate_password_hash(password)),
            )
            conn.commit()
            flash("Registered successfully. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "error")
        finally:
            conn.close()
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = user["username"]
            flash("Logged in.", "success")
            return redirect(url_for("predict"))
        flash("Invalid username or password.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("home"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    table = None
    summary = None

    def page(**kw):
        return render_template("predict.html", features=FEATURES, **kw)

    if request.method == "POST":
        file = request.files.get("csv_file")
        if file and file.filename:
            if not file.filename.lower().endswith(".csv"):
                flash("Only .csv files are allowed.", "error")
                return page()
            try:
                df = pd.read_csv(file, encoding="utf-8-sig")
            except Exception:
                flash("Could not read the CSV file.", "error")
                return page()

            # ignore spaces and capital letters in the column names
            df.columns = [str(c).strip().lower() for c in df.columns]

            if df.empty:
                flash("The CSV file has no data rows.", "error")
                return page()
            missing = [c for c in FEATURE_NAMES if c not in df.columns]
            if missing:
                flash("CSV is missing columns: " + ", ".join(missing), "error")
                return page()

            labels, confs = [], []
            for i, row in df.iterrows():
                res, err = predict_row(row.to_dict())
                if err:
                    flash(f"Row {i + 1}: {err}", "error")
                    return page()
                labels.append(res["prediction"])
                confs.append(res["confidence"])
            df["prediction"] = labels
            df["confidence"] = confs
            summary = {
                "total": len(df),
                "attacks": sum(1 for l in labels if l != "Normal"),
                "normal": sum(1 for l in labels if l == "Normal"),
            }
            table = df.head(50).to_html(classes="table table-sm table-striped data", index=False)
        else:
            data = {k: v for k, v in request.form.items()}
            result, err = predict_row(data)
            if err:
                flash(err, "error")
                result = None

    return page(result=result, table=table, summary=summary)


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    data = request.get_json(silent=True) or {}
    result, err = predict_row(data)
    if err:
        return jsonify({"error": err}), 400
    return jsonify(result)


@app.route("/results")
@login_required
def results():
    # Member A saves model results in app/model_metrics.json (Day 5)
    path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
    metrics = None
    if os.path.exists(path):
        try:
            with open(path) as f:
                metrics = json.load(f).get("models")
        except Exception:
            metrics = None
    return render_template("results.html", metrics=metrics)


@app.errorhandler(413)
def too_large(e):
    flash("File too large. Maximum size is 5 MB.", "error")
    return redirect(url_for("predict"))


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
