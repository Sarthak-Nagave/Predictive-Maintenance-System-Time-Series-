import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

DATA_PATH = "data/processed/train_final.csv"
WINDOW_SIZE = 30

def create_sequences(df, window_size):
    sequences = []
    labels = []
    
    engines = df["engine_id"].unique()
    
    for engine in engines:
        engine_df = df[df["engine_id"] == engine]
        engine_df = engine_df.sort_values("cycle")
        
        features = engine_df.drop(columns=["engine_id", "cycle", "RUL", "failure"]).values
        target = engine_df["failure"].values
        
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
        Dense(1, activation="sigmoid")
    ])
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    print("Creating sequences...")
    X, y = create_sequences(df, WINDOW_SIZE)
    
    print("Train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print("Handling class imbalance...")
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    
    print("Building LSTM model...")
    model = build_model((X.shape[1], X.shape[2]))
    
    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    
    print("Training model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stop]
    )
    
    print("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    model.save("models/lstm_model.keras")
    print("Model saved successfully!")

if __name__ == "__main__":
    main()