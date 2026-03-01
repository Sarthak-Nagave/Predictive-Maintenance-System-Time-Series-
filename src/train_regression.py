import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

DATA_PATH = "data/processed/train_cleaned.csv"
WINDOW_SIZE = 30

def create_sequences(df, window_size):
    sequences = []
    labels = []

    engines = df["engine_id"].unique()

    for engine in engines:
        engine_df = df[df["engine_id"] == engine]
        engine_df = engine_df.sort_values("cycle")

        features = engine_df.drop(columns=["engine_id", "cycle", "RUL"]).values
        target = engine_df["RUL"].values

        for i in range(len(engine_df) - window_size):
            sequences.append(features[i:i+window_size])
            labels.append(target[i+window_size])

    return np.array(sequences), np.array(labels)

def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Creating sequences...")
    X, y = create_sequences(df, WINDOW_SIZE)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building regression model...")
    model = build_model((X.shape[1], X.shape[2]))

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        callbacks=[early_stop]
    )

    print("Evaluating...")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    model.save("models/lstm_rul_regression.keras")
    print("Regression model saved!")

if __name__ == "__main__":
    main()