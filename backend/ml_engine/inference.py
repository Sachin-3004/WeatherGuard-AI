import joblib
from typing import Dict, Any, Tuple
from .config import MODEL_PATH, SCALER_PATH
from .preprocessing import extract_features, validate_raw_input
from .scoring import calculate_calibrated_score
from .training import train_and_persist_pipeline

class AnomalyInferenceEngine:
    def __init__(self):
        if not MODEL_PATH.exists() or not SCALER_PATH.exists():
            train_and_persist_pipeline()
        self.scaler = joblib.load(SCALER_PATH)
        self.model = joblib.load(MODEL_PATH)

    def predict(self, reading: Dict[str, Any]) -> Tuple[bool, float, float]:
        valid, msg = validate_raw_input(reading)
        if not valid:
            return True, 99.0, -1.0

        X = extract_features(reading)
        X_scaled = self.scaler.transform(X)
        
        raw_decision = float(self.model.decision_function(X_scaled)[0])
        score = calculate_calibrated_score(raw_decision)
        is_anomaly = bool(score >= 60.0 or raw_decision < 0)
        
        return is_anomaly, score, raw_decision
