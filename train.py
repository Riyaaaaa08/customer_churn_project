import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Import the data preparation functions we wrote in our other file!
from data_preprocessing import clean_and_preprocess_data, split_and_scale_data

if __name__ == "__main__":
    DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    # 1. Load, clean, split, and scale the data using our preprocessing functions
    cleaned_df = clean_and_preprocess_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_and_scale_data(cleaned_df)
    
    # 2. Train Logistic Regression
    print("\n--- 📈 Training Logistic Regression Model ---")
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
    
    # 3. Train Random Forest Classifier
    print("\n--- 🌲 Training Random Forest Model ---")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    # 4. Evaluate Logistic Regression
    print("\n===== 📈 LOGISTIC REGRESSION EVALUATION =====")
    lr_preds = lr_model.predict(X_test)
    print(classification_report(y_test, lr_preds))
    
    # 5. Evaluate Random Forest
    print("\n===== 🌲 RANDOM FOREST EVALUATION =====")
    rf_preds = rf_model.predict(X_test)
    print(classification_report(y_test, rf_preds))