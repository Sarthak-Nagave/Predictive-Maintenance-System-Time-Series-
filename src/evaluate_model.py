import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from train_model import create_sequences, WINDOW_SIZE

DATA_PATH = "data/processed/train_final.csv"
MODEL_PATH = "models/lstm_model.h5"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Creating sequences...")
    X, y = create_sequences(df, WINDOW_SIZE)

    print("Loading trained model...")
    model = load_model(MODEL_PATH)

    print("Predicting...")
    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    roc_auc = roc_auc_score(y, y_pred_prob)
    print(f"\nROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    main()