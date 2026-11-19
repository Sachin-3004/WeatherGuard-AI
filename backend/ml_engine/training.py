import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .config import MODEL_PATH, SCALER_PATH, CALIBRATION_PATH, FEATURE_NAMES

def generate_synthetic_training_data(n_samples: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    temp = np.random.normal(30.0, 6.5, n_samples)
    pressure = 1012.0 - (temp - 30.0) * 0.35 + np.random.normal(0, 2.5, n_samples)
    humidity = 65.0 - (temp - 30.0) * 1.6 + np.random.normal(0, 7.0, n_samples)
    humidity = np.clip(humidity, 15.0, 95.0)
    
    return pd.DataFrame({
        "temperature": temp,
        "pressure": pressure,
        "humidity": humidity
    })

def train_and_persist_pipeline():
    df = generate_synthetic_training_data()
    X = df[FEATURE_NAMES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=150,
        contamination=0.03,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    joblib.dump({"trained_at": "2026-09-05", "samples": len(df)}, CALIBRATION_PATH)
    print("Model training completed successfully. Artifacts saved in models/")

if __name__ == "__main__":
    train_and_persist_pipeline()
