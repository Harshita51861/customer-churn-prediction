import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
import io

REQUIRED_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

MODEL_PATH = "models/churn_pipeline.pkl"
DATA_PATH = "data/processed/cleaned_churn.csv"

app = FastAPI(title="Customer Churn Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load model pipeline ---
saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
threshold = saved["threshold"]

preprocessor = pipeline.named_steps["preprocess"]
model = pipeline.named_steps["model"]

# --- Build SHAP explainer using a background sample from training data ---
background_df = pd.read_csv(DATA_PATH).drop("Churn", axis=1).sample(100, random_state=42)
background_transformed = preprocessor.transform(background_df)
explainer = shap.LinearExplainer(model, background_transformed)
feature_names = preprocessor.get_feature_names_out()

# --- Recommendation mapping based on top driving factor ---
RECOMMENDATIONS = {
    "Contract": "Offer a discount for switching to an annual or two-year contract.",
    "TechSupport": "Offer a free trial of technical support services.",
    "OnlineSecurity": "Bundle online security add-on at reduced cost.",
    "InternetService": "Review fiber pricing tier or offer a loyalty discount.",
    "MonthlyCharges": "Propose a lower-cost plan or loyalty discount.",
    "tenure": "Early-tenure customer — assign proactive onboarding/check-in.",
    "PaymentMethod": "Encourage switch to automatic payment for reliability discount.",
    "PaperlessBilling": "No specific action — low-impact factor.",
}

def clean_feature_name(raw_name: str) -> str:
    name = raw_name.split("__", 1)[-1]
    return name.replace("_", " ")

def get_recommendation(top_factor_raw: str) -> str:
    base_feature = top_factor_raw.split("__", 1)[-1].split("_")[0]
    for key in RECOMMENDATIONS:
        if key.lower() in base_feature.lower():
            return RECOMMENDATIONS[key]
    return "Monitor customer — no strong single driver identified."

def risk_level(prob: float) -> str:
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Medium"
    else:
        return "High"

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])

    prob = pipeline.predict_proba(df)[0][1]
    prediction = int(prob >= threshold)

    # SHAP explanation
    X_transformed = preprocessor.transform(df)
    shap_values = explainer.shap_values(X_transformed)[0]

    contributions = list(zip(feature_names, shap_values))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = []

    for name, val in contributions[:3]:
        top_factors.append({
            "factor": clean_feature_name(name),
            "impact": round(float(val), 3),
            "direction": (
                "increases risk" if val > 0
                else "decreases risk"
            )
        })

    recommendation = (
        get_recommendation(contributions[0][0])
        if prediction == 1
        else "No action needed."
    )

    return {
        "churn_prediction": "Churn" if prediction == 1 else "No Churn",
        "churn_probability": round(float(prob), 4),
        "risk_level": risk_level(prob),
        "top_factors": top_factors,
        "recommendation": recommendation
    }


@app.post("/predict-batch")
def predict_batch(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing_cols:
        return {"error": f"Missing required columns: {missing_cols}"}

    df_features = df[REQUIRED_COLUMNS].copy()

    probs = pipeline.predict_proba(df_features)[:, 1]
    preds = (probs >= threshold).astype(int)

    X_transformed = preprocessor.transform(df_features)
    shap_values_batch = explainer.shap_values(X_transformed)

    top_factors_list = []

    for row_shap in shap_values_batch:
        contributions = list(zip(feature_names, row_shap))
        contributions.sort(
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_name = clean_feature_name(contributions[0][0])
        top_factors_list.append(top_name)

    result_df = df.copy()

    result_df["churn_probability"] = [
        round(float(p), 4) for p in probs
    ]

    result_df["churn_prediction"] = [
        "Churn" if p == 1 else "No Churn"
        for p in preds
    ]

    result_df["risk_level"] = [
        risk_level(p) for p in probs
    ]

    result_df["top_factor"] = top_factors_list

    result_df = result_df.sort_values(
        "churn_probability",
        ascending=False
    )

    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=churn_predictions.csv"
        }
    )