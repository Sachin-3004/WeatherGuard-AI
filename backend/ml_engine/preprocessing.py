import numpy as np
from typing import Dict, Any, Tuple
from .config import FEATURE_NAMES

def validate_raw_input(reading: Dict[str, Any]) -> Tuple[bool, str]:
    for feat in FEATURE_NAMES:
        if feat not in reading or reading[feat] is None:
            return False, f"Missing required telemetry channel: {feat}"
        try:
            val = float(reading[feat])
            if np.isnan(val) or np.isinf(val):
                return False, f"Non-numeric or NaN value encountered on {feat}"
        except (ValueError, TypeError):
            return False, f"Invalid float value on {feat}"
    return True, "Valid"

def extract_features(data: Dict[str, Any]) -> np.ndarray:
    return np.array([[float(data[f]) for f in FEATURE_NAMES]], dtype=np.float64)
