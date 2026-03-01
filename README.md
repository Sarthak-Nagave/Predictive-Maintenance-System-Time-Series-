# 🚀 Predictive Maintenance System

**Team 2**

---

## 📌 Project Overview

This project implements a **time-series based Predictive Maintenance System** to predict equipment failure using multi-sensor historical data.

The system analyzes sequential degradation patterns and estimates the probability of machine failure before it occurs.

The solution follows a modular architecture consisting of:

* Data Processing Layer
* Feature Engineering Layer
* LSTM Model Training Layer
* Inference API Layer
* Real-Time Monitoring Dashboard

---

## 🎯 Objective

To predict equipment failure using time-series sensor data and provide:

* Failure probability estimation
* Binary failure classification
* Risk level categorization
* Real-time monitoring visualization
* Performance evaluation metrics

---

## 🗂️ Project Structure

```
predictive-maintenance/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   └── lstm_model.h5
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── train_regression.py
│   └── evaluate_model.py
│
├── api/
│   ├── main.py
│   ├── templates/
│   └── static/
│
└── README.md
```

---

## 📊 Dataset

The dataset contains multi-engine time-series sensor data with:

* Engine ID
* Operational cycle
* 21 sensor measurements
* Remaining Useful Life (RUL)

Each row represents one cycle of engine operation.

---

## ⚙️ Data Processing

* Removed constant sensors
* Created binary failure label based on RUL threshold
* Normalized sensor values
* Converted data into sliding window sequences (30 cycles)

---

## 🧠 Model Architecture

We implemented a **Stacked LSTM Network**:

* LSTM (128 units, return_sequences=True)
* Dropout (0.3)
* LSTM (64 units)
* Dense (1, Sigmoid activation)

### Why LSTM?

Equipment degradation is sequential.
LSTM captures long-term temporal dependencies better than traditional ML models.

---

## 📈 Model Performance

Classification Metrics:

* Accuracy: ~97%
* F1 Score: > 0.90
* ROC-AUC: ~0.995

Regression Metrics (RUL Prediction):

* MSE
* RMSE
* R² Score

---

## 🔌 Backend Deployment

The trained model is deployed using **FastAPI**.

Available endpoints:

* `POST /predict` → Predict from manual sequence
* `POST /predict_by_engine` → Predict using engine ID
* `GET /engine_trend/{engine_id}/{sensor}` → Sensor trend data
* `GET /health` → API health check

The model loads once at startup for efficiency.

---

## 🖥️ Dashboard Features

The frontend dashboard includes:

* Failure probability meter (animated gauge)
* Risk level classification (LOW / MEDIUM / HIGH)
* Sensor trend visualization
* Real-time simulation (periodic API calls)
* Dark mode support
* PDF report export

---

## 🏗️ System Architecture

The system follows a modular pipeline:

1. Data Cleaning
2. Feature Engineering
3. LSTM Model Training
4. API Inference Layer
5. Real-Time Dashboard

This design ensures scalability and maintainability.

---

## 🔄 Real-Time Simulation

The dashboard simulates real industrial monitoring by periodically calling the prediction API to mimic IoT streaming behavior.

---

## 🚀 Future Enhancements

* Integration with real IoT streaming (Kafka / MQTT)
* Docker containerization
* Cloud deployment (AWS / Azure / GCP)
* Model retraining automation
* Transformer-based time-series models
* Alert notification system

---

## 👥 Team

**Team 2 – Predictive Maintenance System (Time-Series)**

Members:

* Lalit Manesh More
* Atharva Hanumant Admile
* Sarthak Nivrutti Nagave
* Venkatesh Ganesh Gudade
* Ankita Yatish Kakade
* Sinchan Santosh Rao

---

## 📌 Conclusion

This project successfully implements an end-to-end time-series predictive maintenance pipeline, combining deep learning, backend deployment, and real-time monitoring visualization.

The system demonstrates strong predictive performance and production-ready architecture for industrial applications.



