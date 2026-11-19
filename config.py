from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
CALIBRATION_PATH = MODEL_DIR / "calibration.joblib"
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_aws_telemetry.csv"

FEATURE_NAMES = ["temperature", "pressure", "humidity"]

WMO_BOUNDS = {
    "temperature": {"min": -20.0, "max": 60.0, "max_step": 5.0},
    "pressure": {"min": 600.0, "max": 1080.0, "max_step": 6.0},
    "humidity": {"min": 0.0, "max": 100.0, "max_step": 30.0},
}

DEFAULT_STATIONS = {
    "AWS-101": {"name": "Delhi Ridge", "lat": 28.67, "lon": 77.22, "elev": 216},
    "AWS-102": {"name": "Gurugram South", "lat": 28.45, "lon": 77.02, "elev": 220},
    "AWS-103": {"name": "Noida East", "lat": 28.53, "lon": 77.39, "elev": 200},
    "AWS-104": {"name": "Faridabad", "lat": 28.40, "lon": 77.31, "elev": 205},
    "AWS-105": {"name": "Rohtak Ridge", "lat": 28.89, "lon": 76.60, "elev": 220},
}
