import pandas as pd
import numpy as np
import os

# File paths
RAW_DATA_PATH = "data/raw/train_FD001.txt"
PROCESSED_DATA_PATH = "data/processed/train_cleaned.csv"

def load_data(path):
    # Load space-separated file
    df = pd.read_csv(path, sep=" ", header=None)
    
    # Remove empty columns created by extra spaces
    df = df.dropna(axis=1)
    
    return df

def assign_column_names(df):
    columns = ["engine_id", "cycle"]
    columns += [f"op_setting_{i}" for i in range(1, 4)]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    
    df.columns = columns
    return df

def compute_rul(df):
    # Get max cycle for each engine
    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]
    
    # Merge
    df = df.merge(max_cycle, on="engine_id", how="left")
    
    # Calculate RUL
    df["RUL"] = df["max_cycle"] - df["cycle"]
    
    # Drop max_cycle
    df = df.drop(columns=["max_cycle"])
    
    return df

def save_clean_data(df, path):
    df.to_csv(path, index=False)

def main():
    print("Loading raw data...")
    df = load_data(RAW_DATA_PATH)
    
    print("Assigning column names...")
    df = assign_column_names(df)
    
    print("Computing RUL...")
    df = compute_rul(df)
    
    print("Saving cleaned dataset...")
    save_clean_data(df, PROCESSED_DATA_PATH)
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()