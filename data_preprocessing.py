import pandas as pd
import numpy as np


def load_and_inspect_data(file_path):
    print("\n--- 📂 Loading Dataset ---")
    df = pd.read_csv(file_path)

    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n--- 🎯 Target Class Distribution (Is it imbalanced?) ---")
    churn_counts = df["Churn"].value_counts()
    churn_percentages = df["Churn"].value_counts(normalize=True) * 100

    for class_label in churn_counts.index:
        print(
            f"  {class_label}: {churn_counts[class_label]} customers ({churn_percentages[class_label]:.2f}%)"
        )

    return df


if __name__ == "__main__":
    DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    try:
        df = load_and_inspect_data(DATA_PATH)
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{DATA_PATH}' in this folder.")
        print(
            "Please make sure you drag and drop your downloaded Kaggle CSV file right here!"
        )
