# FinTech Fraud Detection

B.Tech final-year educational prototype for financial transaction fraud and behavioral anomaly detection.

## Modules
- CSV ingestion and synthetic-data generator
- Behavioral feature engineering with prior-transaction features
- Logistic Regression, Random Forest, XGBoost, LightGBM
- PyTorch LSTM sequence classifier and normal-only Autoencoder
- Isolation Forest, validation-scaled anomaly scores, weighted ensemble
- SHAP/LIME optional explanation helpers
- FastAPI prediction, history, alerts, metrics endpoints
- SQLAlchemy persistence (SQLite default; PostgreSQL configurable)
- Kafka simulator/consumer scripts
- Streamlit + Plotly dashboard
- pytest tests

This is a simulated student project. It does not connect to banks/UPI, authorize/block payments, or prove production high-frequency throughput.

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts/generate_sample_data.py
python scripts/train_all.py
uvicorn app.backend.main:app --reload
```
Second terminal: `streamlit run app/frontend/dashboard.py`. API docs: http://127.0.0.1:8000/docs

## Linux/macOS
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/generate_sample_data.py
python scripts/train_all.py
uvicorn app.backend.main:app --reload
```
Second terminal: `streamlit run app/frontend/dashboard.py`.

## Dataset
Quick start generates labelled synthetic data (not real bank data). For PaySim, download the CSV yourself from Kaggle and place it at `data/raw/transactions.csv`. The loader requires `amount` and a target (`isFraud`, `Class`, or `label`). It only uses columns present. Check dataset licensing before redistribution.

## PostgreSQL
SQLite works by default. For PostgreSQL set `DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME` in `.env`.

## Kafka
Kafka is optional for the basic API demo. Start a Kafka broker, then run `python -m app.streaming.simulator` and `python -m app.streaming.consumer`. Alternatively `python -m app.streaming.simulator --api` posts simulated events directly to FastAPI.

## Tests
Run `pytest -q`. Train first for prediction endpoints. Metrics are calculated from the held-out test split; never claim the synthetic data results as real-world performance.

## Methodology
Chronological split; scaler fit on training data only; validation data selects threshold and anomaly score ranges; test data is reserved for reporting. Anomaly scores are not probabilities. LSTM consumes ordered transaction windows. Autoencoder is trained on non-fraud training examples. The demo risk score is an indicative ranking, not a payment decision.
