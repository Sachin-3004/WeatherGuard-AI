"""
WeatherGuard AI - Configuration & Constants
SIH Problem Statement: SIH26073

This file centralizes model hyperparameters, artifact storage paths,
and feature definitions so the team doesn't hardcode magic values across scripts.
"""

from pathlib import Path

# Base directories for artifacts
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Saved artifact file paths
MODEL_PATH = ARTIFACTS_DIR / "isolation_forest.joblib"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"

# The three core telemetry features specified by the SIH problem statement
FEATURE_NAMES = ["temperature", "pressure", "humidity"]

# Hyperparameters for Isolation Forest:
# - n_estimators: Number of isolation trees. 150 gives robust stability without latency penalties.
# - contamination: Estimated proportion of outliers in the clean baseline dataset (~2.5%).
# - random_state: Fixed seed ensures repeatable results during jury demonstrations.
MODEL_PARAMS = {
    "n_estimators": 150,
    "max_samples": "auto",
    "contamination": 0.025,
    "random_state": 42,
    "n_jobs": -1  # Use all available CPU cores for training
}

# Physical bounds for sanity checking during preprocessing
PHYSICAL_BOUNDS = {
    "temperature": (-30.0, 60.0),  # °C: Indian meteorological envelope
    "pressure": (600.0, 1080.0),   # hPa: Sea level to high mountain passes
    "humidity": (0.0, 100.0)       # %: Relative humidity domain
}