import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

INPUT_PATH = "data/processed/train_cleaned.csv"
OUTPUT_PATH = "data/processed/train_final.csv"

FAILURE_THRESHOLD = 30  # RUL threshold

def load_data(path):
    return pd.read_csv(path)

def remove_constant_columns(df):
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index
    df = df.drop(columns=constant_cols)
    print(f"Removed constant columns: {list(constant_cols)}")
    return df

def create_failure_label(df, threshold):
    df["failure"] = np.where(df["RUL"] <= threshold, 1, 0)
    return df

def scale_features(df):
    scaler = StandardScaler()
    
    feature_cols = [col for col in df.columns 
                    if col not in ["engine_id", "cycle", "RUL", "failure"]]
    
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df

def check_class_balance(df):
    print("\nClass Distribution:")
    print(df["failure"].value_counts())
    print("\nClass Ratio:")
    print(df["failure"].value_counts(normalize=True))

def save_data(df, path):
    df.to_csv(path, index=False)

def main():
    print("Loading cleaned dataset...")
    df = load_data(INPUT_PATH)

    print("Removing constant sensors...")
    df = remove_constant_columns(df)

    print("Creating failure label...")
    df = create_failure_label(df, FAILURE_THRESHOLD)

    print("Scaling features...")
    df = scale_features(df)

    check_class_balance(df)

    print("Saving final dataset...")
    save_data(df, OUTPUT_PATH)

    print("Feature engineering completed successfully!")

if __name__ == "__main__":
    main()