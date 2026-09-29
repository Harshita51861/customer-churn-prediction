import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_PATH = "data/processed/cleaned_churn.csv"
MODEL_PATH = "models/churn_pipeline.pkl"

df = pd.read_csv(DATA_PATH)
X = df.drop("Churn", axis=1)
y = df["Churn"]

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

saved = joblib.load(MODEL_PATH)
pipeline = saved["pipeline"]
threshold = saved["threshold"]

probs = pipeline.predict_proba(X_test)[:, 1]

print(f"{'Threshold':<10}{'Precision':<12}{'Recall':<10}{'F1':<10}")
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
    preds = (probs >= t).astype(int)
    p = precision_score(y_test, preds)
    r = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"{t:<10}{p:<12.3f}{r:<10.3f}{f1:<10.3f}")