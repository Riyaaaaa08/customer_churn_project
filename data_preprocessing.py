import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#datacleaning and preprocessing
def clean_and_preprocess_data(file_path):
    print("\n---  Running Preprocessing Pipeline ---")
    df = pd.read_csv(file_path)
    
    # 1. Drop CustomerID (It's a random unique text string, useless for ML patterns)
    df = df.drop(columns=['customerID'])
    
    
# 2. Convert TotalCharges to numeric, turning blank spaces into NaN (Not a Number)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill those NaN blanks with the middle (median) value of the column
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
# 3. Convert Target Variable ('Churn') to integers (No -> 0, Yes -> 1)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 4. Automatically find all remaining text columns and convert them to 0s and 1s
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Post-cleaning shape: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def split_and_scale_data(df):
    print("\n--- ✂️ Splitting & Scaling Data ---")
    # Separate features (inputs) from the target label (output)
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Split into 80% Training data and 20% Testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numeric features so they share the same scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training features shape: {X_train_scaled.shape}")
    print(f"Testing features shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test,scaler

if __name__ == "__main__":
    DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    
    try:
        # Run our data cleaning function
        cleaned_df = clean_and_preprocess_data(DATA_PATH)
        
        # Run our data splitting and scaling function
        X_train, X_test, y_train, y_test = split_and_scale_data(cleaned_df)
        
        print("\n✅ Preprocessing Pipeline Completed Successfully!")
        
    except FileNotFoundError:
        print(f"\n❌ Error: Could not find '{DATA_PATH}' in this folder.")

