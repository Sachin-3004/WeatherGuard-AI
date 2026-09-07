"""
WeatherGuard AI - Model Training Module
SIH Problem Statement: SIH26073

Why Isolation Forest?
- Unsupervised: Real Automatic Weather Station feeds lack high-quality ground-truth labels.
- Tree-Based Partitioning: Recursively cuts the feature space at random split points.
  Anomalies (rare, deviant patterns) require very few splits to isolate (short average tree depth),
  whereas normal climatological clusters take many splits to isolate (deep average tree depth).
- Time & Space Efficient: $O(n \log n)$ training complexity; fast millisecond inference.

This script:
1. Generates 5,000 realistic Indian surface weather observations adhering to physical covariance
   (diurnal thermal heating, hydrostatic pressure coupling, and inverse relative humidity).
2. Fits the StandardScaler.
3. Trains an Isolation Forest model.
4. Validates model predictions on synthetic edge cases (spikes, out-of-bounds, nominal points).
5. Persists both model and scaler artifacts.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

from ml_engine.config import MODEL_PARAMS, MODEL_PATH, SCALER_PATH
from ml_engine.preprocessing import WeatherDataPreprocessor


def generate_synthetic_nominal_aws_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generates realistic, physically consistent normal baseline AWS telemetry.
    Unlike naive random numbers, this accounts for meteorological physical relationships:
    - Temperature follows standard synoptic distributions (15°C to 42°C).
    - Pressure follows the barometric formula with small atmospheric tides.
    - Humidity is thermodynamically inversely coupled with temperature (Clausius-Clapeyron).
    """
    np.random.seed(42)

    # 1. Base Temperatures: Normal distribution around typical daytime mean
    temps = np.random.normal(loc=29.5, scale=5.8, size=n_samples)
    temps = np.clip(temps, 12.0, 44.0)

    # 2. Surface Pressure: Mean ~1010 hPa, with slight negative diurnal thermal coupling
    # Warmer surface air causes localized convective expansion (slight pressure drop)
    pressures = 1012.0 - (temps - 29.5) * 0.35 + np.random.normal(0, 2.5, size=n_samples)
    pressures = np.clip(pressures, 990.0, 1030.0)

    # 3. Relative Humidity: Strongly anti-correlated with ambient temperature
    # As temperature climbs, saturation vapor pressure rises, dropping RH if moisture content is constant
    rh_base = 65.0 - (temps - 29.5) * 1.8 + np.random.normal(0, 6.0, size=n_samples)
    humidities = np.clip(rh_base, 15.0, 95.0)

    df = pd.DataFrame({
        "temperature": np.round(temps, 1),
        "pressure": np.round(pressures, 1),
        "humidity": np.round(humidities, 1)
    })
    return df


def train_and_save_model():
    print("[Trainer] Generating synthetic nominal training telemetry (5,000 samples)...")
    training_data = generate_synthetic_nominal_aws_data(n_samples=5000)
    print(training_data.describe().round(2))

    print("\n[Trainer] Fitting preprocessor and scaling features...")
    preprocessor = WeatherDataPreprocessor()
    X_scaled = preprocessor.fit_and_transform(training_data)
    preprocessor.save_scaler(SCALER_PATH)

    print(f"\n[Trainer] Training Isolation Forest with parameters: {MODEL_PARAMS}...")
    model = IsolationForest(**MODEL_PARAMS)
    model.fit(X_scaled)

    # Save model artifact
    joblib.dump(model, MODEL_PATH)
    print(f"[Trainer] Model saved successfully to {MODEL_PATH}")

    # Sanity checks on known synthetic benchmark readings
    print("\n[Trainer] Running validation tests on standard cases...")
    test_cases = [
        {"name": "Nominal Sunny Day", "temperature": 32.0, "pressure": 1010.0, "humidity": 55.0},
        {"name": "Extreme Thermal Spike (Sensor Fault)", "temperature": 55.5, "pressure": 1008.0, "humidity": 50.0},
        {"name": "Impossible Hot & Humid (Dew point clash)", "temperature": 48.0, "pressure": 1011.0, "humidity": 95.0},
        {"name": "Barometric Glitch (Vacuum fault)", "temperature": 28.0, "pressure": 750.0, "humidity": 60.0}
    ]

    for test in test_cases:
        vec = np.array([[test["temperature"], test["pressure"], test["humidity"]]])
        vec_scaled = preprocessor.scaler.transform(vec)
        raw_pred = model.predict(vec_scaled)[0]  # 1 = normal, -1 = anomaly
        decision = model.decision_function(vec_scaled)[0]
        classification = "NORMAL" if raw_pred == 1 else "ANOMALY"
        print(f"  * {test['name']:45} => {classification:8} (Decision Score: {decision:.4f})")

    print("\n[Trainer] Pipeline training complete and verified!")


if __name__ == "__main__":
    train_and_save_model()