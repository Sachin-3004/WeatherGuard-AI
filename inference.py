"""
WeatherGuard AI - Model Inference Engine
SIH Problem Statement: SIH26073

This module handles:
1. Loading and holding pre-trained Isolation Forest and Scaler in memory.
2. Thread-safe, low-latency prediction for streaming sensor telemetry.
3. Fallback bootstrapping if the model artifacts are not yet generated on disk.
"""

from typing import Dict, Tuple, Any
import numpy as np
import joblib

from ml_engine.config import MODEL_PATH, SCALER_PATH
from ml_engine.preprocessing import WeatherDataPreprocessor
from ml_engine.scoring import evaluate_isolation_forest_result
from ml_engine.training import train_and_save_model


class AnomalyDetector:
    """
    High-performance inference engine for AWS sensor validation.
    """

    def __init__(self, auto_train_if_missing: bool = True):
        self.preprocessor = WeatherDataPreprocessor()
        self.model = None
        self._ensure_artifacts_exist(auto_train_if_missing)
        self.load_artifacts()

    def _ensure_artifacts_exist(self, auto_train: bool):
        """Checks if pre-trained artifacts exist; if not, triggers the training script."""
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            if auto_train:
                print("[Inference] Artifacts missing. Running baseline training pipeline...")
                train_and_save_model()
            else:
                raise FileNotFoundError(
                    f"Model artifacts not found at {MODEL_PATH} and {SCALER_PATH}. "
                    "Run `python -m ml_engine.training` first."
                )

    def load_artifacts(self):
        """Loads both the trained model and fitted scaler from disk."""
        print(f"[Inference] Loading model from {MODEL_PATH}...")
        self.model = joblib.load(MODEL_PATH)
        self.preprocessor.load_scaler(SCALER_PATH)
        print("[Inference] Model and preprocessor successfully loaded into memory.")

    def predict(self, reading: Dict[str, float]) -> Tuple[bool, float]:
        """
        Takes raw telemetry reading:
            {"temperature": float, "pressure": float, "humidity": float}
        Returns:
            (is_anomaly: bool, anomaly_score: float)
        """
        # Step 1: Physical domain and structure validation
        is_valid, err_msg = self.preprocessor.validate_reading(reading)
        if not is_valid:
            # If reading is physically out-of-bounds or non-numeric, it is an undeniable anomaly
            print(f"[Inference] Physical bounds validation failed: {err_msg}")
            return True, 99.9

        # Step 2: Feature scaling via fitted StandardScaler
        X_scaled = self.preprocessor.transform_single(reading)

        # Step 3: Raw Isolation Forest prediction
        # predict() returns 1 (normal) or -1 (anomaly)
        raw_pred = self.model.predict(X_scaled)[0]
        # decision_function() returns raw anomaly score
        raw_decision = self.model.decision_function(X_scaled)[0]

        # Step 4: Calibrate raw decision to interpretable [0-100] scale
        is_anomaly, anomaly_score = evaluate_isolation_forest_result(raw_pred, raw_decision)
        return is_anomaly, anomaly_score


# Instantiate a singleton detector for reuse by the API and worker threads
detector = AnomalyDetector()