"""
WeatherGuard AI - Data Preprocessing Pipeline
SIH Problem Statement: SIH26073

Why Preprocessing Matters:
1. Feature Scaling (StandardScaler):
   - Temperature has ranges like ~10 to 45°C.
   - Atmospheric Pressure has ranges like ~850 to 1020 hPa.
   - Humidity has ranges like ~10 to 95%.
   Although decision tree splits in Isolation Forest are scale-invariant along single axes,
   standardization ($z = (x - \mu) / \sigma$) guarantees consistent numerical stability,
   enables seamless distance-based extensions (like PCA or k-NN later), and prevents numerical overflows.

2. Physical Bounds Validation:
   - Validates that incoming values are finite numbers and within physical limits.
"""

from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from ml_engine.config import FEATURE_NAMES, PHYSICAL_BOUNDS, SCALER_PATH


class WeatherDataPreprocessor:
    """
    Handles feature validation, structural organization, and standard scaling
    for weather station telemetry vectors.
    """

    def __init__(self, scaler: Union[StandardScaler, None] = None):
        self.scaler = scaler if scaler is not None else StandardScaler()
        self.is_fitted = False

    def validate_reading(self, reading: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validates that required features exist, are numerical, and are within plausible physical limits.
        Returns (is_valid, error_message).
        """
        for feature in FEATURE_NAMES:
            if feature not in reading:
                return False, f"Missing required telemetry feature: '{feature}'"
            val = reading[feature]
            if val is None or not isinstance(val, (int, float)) or np.isnan(val) or np.isinf(val):
                return False, f"Feature '{feature}' has invalid numerical value: {val}"
            
            # Check sanity limits
            low, high = PHYSICAL_BOUNDS[feature]
            if val < low or val > high:
                return False, (
                    f"Feature '{feature}' with value {val} is outside terrestrial physical bounds "
                    f"[{low}, {high}]."
                )

        return True, ""

    def transform_single(self, reading: Dict[str, float]) -> np.ndarray:
        """
        Converts a single dictionary reading {"temperature": T, "pressure": P, "humidity": RH}
        into a scaled 2D numpy array shaped (1, 3) ready for model scoring.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted or loaded with a trained scaler.")
        
        # Extract features in strict canonical order
        raw_vector = np.array([[reading[col] for col in FEATURE_NAMES]], dtype=np.float64)
        scaled_vector = self.scaler.transform(raw_vector)
        return scaled_vector

    def fit_and_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits the StandardScaler on training historical baseline data and returns the scaled matrix.
        """
        X = df[FEATURE_NAMES].values.astype(np.float64)
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def save_scaler(self, path=SCALER_PATH):
        """Persists the fitted scaler object to disk."""
        joblib.dump(self.scaler, path)
        print(f"[Preprocessor] Scaler saved successfully to {path}")

    def load_scaler(self, path=SCALER_PATH):
        """Loads a pre-fitted scaler object from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found at {path}. Run training first.")
        self.scaler = joblib.load(path)
        self.is_fitted = True
        print(f"[Preprocessor] Scaler loaded successfully from {path}")