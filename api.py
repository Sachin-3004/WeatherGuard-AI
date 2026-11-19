from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from .inference import AnomalyInferenceEngine
from .root_cause import diagnose_root_cause
from .severity import compute_severity_and_confidence
from .xai import xai_engine
from .spatial_analysis import spatial_engine
from .sensor_health import station_health_trackers
from .correction import correction_engine

app = FastAPI(
    title="WeatherGuard AI Backend",
    version="2.0.0",
    description="Intelligent AWS Anomaly Detection and Diagnostic Engine (SIH26073)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference_engine = AnomalyInferenceEngine()

class TelemetryPayload(BaseModel):
    station_id: str = "AWS-101"
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    humidity: Optional[float] = None
    history: List[Dict[str, Any]] = []
    peer_stations: List[Dict[str, Any]] = []

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "online",
        "engine": "WeatherGuard AI Scikit-Learn Pipeline",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/detect")
def detect_telemetry_anomaly(payload: TelemetryPayload):
    current = {
        "temperature": payload.temperature,
        "pressure": payload.pressure,
        "humidity": payload.humidity
    }

    is_ml_anomaly, ml_score, _ = inference_engine.predict(current)
    spatial = spatial_engine.analyze(payload.station_id, current, payload.peer_stations)
    is_spatial_inconsistent = (spatial["status"] == "INCONSISTENT")

    rca = diagnose_root_cause(current, payload.history, is_spatial_inconsistent)
    is_anomaly = (
        is_ml_anomaly
        or (rca["root_cause"] != "Unknown anomaly" and rca["root_cause"] != "Possible genuine meteorological event")
        or is_spatial_inconsistent
    )

    affected = []
    if current["temperature"] is None or current["temperature"] < -20 or current["temperature"] > 60 or "temperature" in rca["root_cause"].lower():
        affected.append("temperature")
    if current["pressure"] is None or current["pressure"] < 600 or current["pressure"] > 1080 or "pressure" in rca["root_cause"].lower():
        affected.append("pressure")
    if current["humidity"] is None or current["humidity"] < 0 or current["humidity"] > 100 or "humidity" in rca["root_cause"].lower():
        affected.append("humidity")

    sev_tier, severity, confidence = compute_severity_and_confidence(
        ml_score, rca["root_cause"], affected, is_anomaly
    )

    xai_info = xai_engine.explain(current)
    tracker = station_health_trackers[payload.station_id]
    tracker.record_reading(current, is_anomaly, rca["root_cause"])

    correction_info = None
    if is_anomaly and affected:
        correction_info = correction_engine.estimate(affected[0], payload.history, payload.peer_stations)

    return {
        "station_id": payload.station_id,
        "timestamp": datetime.utcnow().isoformat(),
        "is_anomaly": is_anomaly,
        "anomaly_score": round(ml_score, 1),
        "status": "ANOMALY DETECTED" if is_anomaly else "NORMAL",
        "severity": sev_tier,
        "severity_score": severity,
        "confidence": confidence,
        "confidence_type": "Heuristic Evidence Calculation",
        "root_cause": rca["root_cause"],
        "evidence": rca["evidence"],
        "affected_sensors": affected,
        "spatial_consistency": spatial,
        "xai": xai_info,
        "health": tracker.get_all_tiers(),
        "correction": correction_info
    }
