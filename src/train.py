import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "data/processed/cleaned_churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"

def build_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42)
    }

    best_model_name, best_auc, best_pipeline = None, 0, None

    for name, model in models.items():
        pipeline = build_pipeline(model, numeric_cols, categorical_cols)

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"\n{name} CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        preds = pipeline.predict(X_test)

        test_auc = roc_auc_score(y_test, probs)
        print(f"{name} Test ROC-AUC: {test_auc:.4f}")
        print(classification_report(y_test, preds))

        if test_auc > best_auc:
            best_auc = test_auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_model_name} (Test ROC-AUC: {best_auc:.4f})")

    THRESHOLD = 0.4
    joblib.dump({"pipeline": best_pipeline, "threshold": THRESHOLD}, MODEL_PATH)
    print(f"Saved to {MODEL_PATH} with threshold={THRESHOLD}")

if __name__ == "__main__":
    main()