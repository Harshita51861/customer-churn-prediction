# Retainly — Customer Churn Prediction & Retention Console

An end-to-end machine learning system that predicts telecom customer churn, explains *why* each prediction was made using SHAP, and surfaces the result through a retention-analyst console — not just a notebook.

**Live demo:** `<add your Netlify URL here>`
**API docs:** `<add your Render URL here>/docs`

---

## What this does

Given a customer's account, service, and billing details, the system:

1. Predicts the probability that the customer will churn
2. Classifies them into a risk tier (Low / Medium / High) using a tuned decision threshold
3. Explains the prediction with the top SHAP-driven factors — labeled as **Dr** (increases risk) or **Cr** (decreases risk), mirroring how the factor pushes the customer's "account balance" of risk up or down
4. Suggests a retention action based on the leading risk driver
5. Supports three modes of use:
   - **Single customer** — fill in one customer's details, get an explained reading
   - **Batch file** — upload a CSV of customers, get back a ranked, downloadable risk report for the whole base
   - **What if** — change one field on a customer (e.g. switch to an annual contract) and see the effect on churn probability, holding everything else constant

---

## Why it's built this way

Most churn-prediction portfolios stop at a notebook with an accuracy score. This project treats churn as a decision-support problem, not just a classification exercise:

- **Threshold tuning, not the default 0.5.** Missing an actual churner (false negative) costs more than a false alarm, so the decision threshold (0.40) was chosen by evaluating precision/recall/F1 across a range of cutoffs, not left at the library default.
- **Model comparison, not a single model.** Logistic Regression, Random Forest, and XGBoost were all trained and cross-validated; the best performer (Logistic Regression, by ROC-AUC and recall) was kept — an interesting result, since simpler models can outperform ensembles when the underlying signal is close to linear.
- **Explainability is load-bearing, not decorative.** Every prediction returns its SHAP attribution, not just a probability — so a retention team could act on *why* a customer is flagged, not just that they were.
- **Batch scoring mirrors real usage.** Retention teams score their whole customer base at once, not one customer at a time — the batch endpoint reflects that.

---

## Results

| Model | Test ROC-AUC | Recall (churn class) | Precision (churn class) |
|---|---|---|---|
| **Logistic Regression** (selected) | 0.841 | 0.78 (0.50 threshold) / 0.87 (0.40 threshold) | 0.50 / 0.47 |
| Random Forest | 0.819 | 0.63 | 0.55 |
| XGBoost | 0.814 | 0.65 | 0.52 |

Dataset: [IBM Telco Customer Churn](https://github.com/IBM/watsonx-ai-samples) — 7,043 customers, 19 features, 26.5% baseline churn rate.

Decision threshold was set to **0.40** (rather than the default 0.50) after evaluating the precision/recall/F1 trade-off, prioritizing recall since missing an actual churner is costlier than a false alarm.

---

## Architecture

```
Raw CSV (Telco churn dataset)
        │
        ▼
  Data cleaning (src/data_prep.py)
        │
        ▼
  EDA (notebooks/01_eda.ipynb)
        │
        ▼
  Preprocessing pipeline + model comparison (src/train.py)
  Logistic Regression / Random Forest / XGBoost, 5-fold CV
        │
        ▼
  Threshold tuning (src/threshold_tuning.py)
        │
        ▼
  Saved pipeline + threshold (models/churn_pipeline.pkl)
        │
        ▼
  FastAPI backend (api/main.py)
  /health · /predict · /predict-batch
  SHAP explainer built on the trained pipeline
        │
        ▼
  Frontend console (frontend/index.html)
  Single customer · Batch file · What-if simulator
```

---

## Tech stack

- **Modeling:** Python, pandas, scikit-learn, XGBoost, SHAP
- **Backend:** FastAPI, uvicorn
- **Frontend:** vanilla HTML/CSS/JS (no framework — kept dependency-free for easy static hosting)
- **Deployment:** Render (API), Netlify (frontend)

---

## Project structure

```
customer-churn-prediction/
├── data/
│   ├── raw/                     # original dataset
│   └── processed/               # cleaned dataset
├── notebooks/
│   └── 01_eda.ipynb             # exploratory data analysis
├── src/
│   ├── data_prep.py             # cleaning pipeline
│   ├── train.py                 # model comparison + training
│   └── threshold_tuning.py      # precision/recall/F1 vs threshold
├── models/
│   └── churn_pipeline.pkl       # saved preprocessing + model + threshold
├── api/
│   └── main.py                  # FastAPI app (predict, predict-batch, SHAP)
├── frontend/
│   └── index.html               # single-file console UI
├── requirements.txt
└── README.md
```

---

## Running it locally

**1. Clone and set up the environment**
```bash
git clone https://github.com/Harshita51861/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Prepare the data and train the model**
```bash
python src/data_prep.py
python src/train.py
```

**3. Run the API**
```bash
uvicorn api.main:app --reload
```
API docs available at `http://127.0.0.1:8000/docs`.

**4. Open the frontend**
Open `frontend/index.html` directly in a browser (with the API running locally, or pointed at a deployed API URL inside the file's `API_URL` constant).

---

## API reference

### `GET /health`
Returns `{"status": "healthy"}` — used for uptime checks.

### `POST /predict`
Single customer prediction. Request body: customer fields (gender, tenure, contract type, charges, service subscriptions, etc.). Returns:
```json
{
  "churn_prediction": "Churn",
  "churn_probability": 0.809,
  "risk_level": "High",
  "top_factors": [
    {"factor": "tenure", "impact": 1.001, "direction": "increases risk"}
  ],
  "recommendation": "Early-tenure customer — assign proactive onboarding/check-in."
}
```

### `POST /predict-batch`
Accepts a multipart CSV upload with the same required columns as `/predict`. Returns a downloadable CSV with `churn_probability`, `churn_prediction`, `risk_level`, and `top_factor` appended, sorted by highest risk first.

---

## Future improvements

- Hyperparameter tuning (RandomizedSearchCV) on the selected model
- Model monitoring for feature drift if retrained on live data
- Authentication on the batch endpoint before any production use with real customer data
