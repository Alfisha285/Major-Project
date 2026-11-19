# IoT Intrusion Detection (PN110)

Detection of real-time malicious intrusions and attacks in IoT-empowered cybersecurity
infrastructures. An ensemble of three deep learning models (DNN, CNN and RNN-LSTM) trained
on UNSW-NB15, with a Flask web application for login, prediction and result comparison.

## Folder structure
```
MAJOR PROJECT/
|-- app/                 Flask web application
|   |-- app.py           routes, login, CSV upload, results page
|   |-- predictor.py     connects the app to preprocessing and the ensemble
|   |-- test_app.py      test cases for the application
|   |-- templates/       HTML pages
|   `-- static/          style.css
|-- src/
|   |-- preprocess.py    cleaning, encoding, scaling, feature selection, preprocess_input
|   |-- train_models.py  DNN and CNN training, metrics, plots
|   |-- lstm_ensemble.py RNN-LSTM, soft-voting ensemble, EnsembleModel
|   `-- evaluate_all.py  final comparison charts and report tables
|-- notebooks/           step by step notebooks and figures
|-- models/              trained models, metrics, preprocessing files
|-- data/raw/            datasets (not uploaded to GitHub)
|-- reports/             result tables for the report
`-- check_setup.py, check_day2.py, check_day3.py, check_day4.py, show_results.py
```

## Setup (Windows, VS Code)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_setup.py
```
Put the UNSW-NB15 files `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
(and the NSL-KDD files) in `data/raw/`.

## Run the pipeline (in this order)
1. `notebooks/01_data_exploration.ipynb`, `02_nslkdd_exploration.ipynb`
2. `python src\preprocess.py`  (or `notebooks/03_preprocessing.ipynb`), then `python check_day2.py`
3. `notebooks/04_dnn_cnn_training.ipynb`, then `python check_day3.py`
4. `notebooks/05_lstm_ensemble.ipynb`, then `python check_day4.py`
5. `python src\evaluate_all.py`  (charts and tables for the report)

## Run the web application
```
.venv\Scripts\activate
python app\app.py
```
Open http://127.0.0.1:5000, register, log in and use the Predict page.
The first prediction takes longer because the three models are loaded then.

To use a fake model (no TensorFlow, for quick interface tests), set this before starting:
```
set MODEL_MODE=mock
```
or change the line `MODEL_MODE = "real"` in `app/predictor.py`.

## Test the web application
```
python app\test_app.py           (fake model, a few seconds)
python app\test_app.py --real    (real ensemble, also checks app = notebook predictions)
```
The results are saved to `reports/app_test_results.md`.

## Notes
- Set your own secret key before sharing the app:  `set SECRET_KEY=some-long-random-text`
- The built-in Flask server is for development only.
- `app/users.db` (the user database) and `app/instance/` are created automatically and are not uploaded to GitHub.
- The Results page reads the saved metrics, so its numbers always match the trained models.
