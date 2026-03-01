import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI(title="Predictive Maintenance System")

# ---------------- CONFIG ---------------- #

MODEL_PATH = "models/lstm_model.h5"
DATA_PATH = "data/processed/train_final.csv"
WINDOW_SIZE = 30
NUM_FEATURES = 17

# ---------------- LOAD MODEL & DATA ---------------- #

model = load_model(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# ---------------- SERVE FRONTEND ---------------- #

app.mount("/static", StaticFiles(directory="api/static"), name="static")
templates = Jinja2Templates(directory="api/templates")

@app.get("/")
def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- SCHEMAS ---------------- #

class SensorSequence(BaseModel):
    sequence: list

class EngineRequest(BaseModel):
    engine_id: int

# ---------------- MANUAL SEQUENCE PREDICTION ---------------- #

@app.post("/predict")
def predict_failure(data: SensorSequence):

    sequence = np.array(data.sequence)

    if sequence.shape != (WINDOW_SIZE, NUM_FEATURES):
        return {
            "error": f"Expected input shape ({WINDOW_SIZE}, {NUM_FEATURES})"
        }

    sequence = np.expand_dims(sequence, axis=0)

    prob = float(model.predict(sequence)[0][0])
    prediction = int(prob > 0.5)

    return {
        "failure_probability": round(prob, 4),
        "predicted_class": prediction,
        "message": "Failure Likely" if prediction == 1 else "Machine Healthy"
    }

# ---------------- ENGINE ID PREDICTION ---------------- #

@app.post("/predict_by_engine")
def predict_by_engine(request: EngineRequest):

    engine_df = df[df["engine_id"] == request.engine_id].sort_values("cycle")

    if engine_df.empty:
        return {"error": "Engine ID not found"}

    if len(engine_df) < WINDOW_SIZE:
        return {"error": "Not enough cycles for this engine"}

    features = engine_df.drop(
        columns=["engine_id", "cycle", "RUL", "failure"]
    ).values

    sequence = features[-WINDOW_SIZE:]
    sequence = np.expand_dims(sequence, axis=0)

    prob = float(model.predict(sequence)[0][0])
    prediction = int(prob > 0.5)

    return {
        "engine_id": request.engine_id,
        "failure_probability": round(prob, 4),
        "predicted_class": prediction,
        "message": "Failure Likely" if prediction == 1 else "Machine Healthy"
    }

# ---------------- SENSOR TREND API ---------------- #

@app.get("/engine_trend/{engine_id}/{sensor_name}")
def engine_trend(engine_id: int, sensor_name: str):

    engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")

    if engine_df.empty:
        return {"error": "Engine ID not found"}

    if sensor_name not in df.columns:
        return {"error": "Invalid sensor name"}

    return {
        "cycle": engine_df["cycle"].tolist(),
        "values": engine_df[sensor_name].tolist()
    }

# ---------------- HEALTH CHECK ---------------- #

@app.get("/health")
def health():
    return {"status": "API Running Successfully"}