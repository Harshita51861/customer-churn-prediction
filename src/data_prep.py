import pandas as pd

RAW_PATH = "data/raw/WA_FnUseC_TelcoCustomerChurn.csv"
PROCESSED_PATH = "data/processed/cleaned_churn.csv"

def load_and_clean(path=RAW_PATH):
    df = pd.read_csv(path)

    # TotalCharges has blank strings for new customers -> becomes NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df = df.drop("customerID", axis=1)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df

if __name__ == "__main__":
    df = load_and_clean()
    df.to_csv(PROCESSED_PATH, index=False)

    print("Shape:", df.shape)
    print("\nMissing values:", df.isnull().sum().sum())
    print("\nChurn distribution:\n", df["Churn"].value_counts(normalize=True))