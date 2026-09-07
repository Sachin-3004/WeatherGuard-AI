"""
WeatherGuard AI - Complete ML & Meteorological Diagnostic API
SIH Problem Statement: SIH26073

Pipeline Architecture:
1. Ingestion & Preprocessing (StandardScaler & Physical Domain Gates)
2. Stage 1: Unsupervised Isolation Forest Anomaly Scoring
3. Stage 2: Deterministic Root-Cause Analysis (RCA) & Thermodynamic Checks
4. Measurable 5-Factor Severity Calculation & Heuristic Evidence Confidence
5. Feature-Deviation Attribution (XAI)
6. Spatial Peer Consistency across Neighboring AWS Nodes
7. Evidence-Gated Spatial-Temporal Correction Estimation (Non-destructive)
8. Cumulative Sensor Health & Maintenance Tracking
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

# Core internal engine modules
from ml_engine.inference import detector
from ml_engine.root_cause import root_cause_engine
from ml_engine.severity import severity_engine
from ml_engine.xai import xai_engine
from ml_engine.spatial_analysis import spatial_engine, AWS_STATION_NETWORK
from ml_engine.sensor_health import station_health_trackers
from ml_engine.correction import correction_engine

app = FastAPI(
    title="WeatherGuard AI - Unified AWS Diagnostic Engine",
    version="3.0.0",
    description="Full-stack AI/ML telemetry validator for Automatic Weather Stations (SIH26073)"
)

# Enable CORS for local and hosted frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Pydantic Schemas
# ----------------------------------------------------------------------

class StationTelemetryInput(BaseModel):
    station_id: str = Field(default="AWS-01", description="Identifier of target AWS node")
    timestamp: Optional[str] = Field(None, description="ISO timestamp of observation")
    temperature: Optional[float] = Field(None, description="Air Temperature in °C", example=31.8)
    pressure: Optional[float] = Field(None, description="Barometric Pressure in hPa", example=1008.4)
    humidity: Optional[float] = Field(None, description="Relative Humidity in %", example=55.0)


class FullAnalysisRequest(BaseModel):
    target: StationTelemetryInput
    history: List[StationTelemetryInput] = Field(
        default_factory=list,
        description="Previous rolling observations for temporal rate and frozen-value checks"
    )
    network_snapshot: Dict[str, Dict[str, Optional[float]]] = Field(
        default_factory=dict,
        description="Current observations from peer stations in the regional synoptic mesh"
    )


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Provides backend status, model readiness, and timestamp."""
    return {
        "status": "online",
        "service": "WeatherGuard AI Engine",
        "version": "3.0.0",
        "sih_problem_code": "SIH26073",
        "stage_1_model": "Isolation Forest (scikit-learn)",
        "stage_2_engine": "Deterministic Meteorological Physics & Spatial Consensus",
        "model_loaded": detector.model is not None,
        "supported_stations": list(AWS_STATION_NETWORK.keys()),
        "server_time_utc": datetime.utcnow().isoformat() + "Z"
    }


@app.post(
    "/api/v1/detect",
    status_code=status.HTTP_200_OK,
    summary="Minimal detection endpoint matching SIH input/output spec"
)
def detect_minimal(payload: StationTelemetryInput) -> Dict[str, Any]:
    """
    Standard SIH minimal endpoint: accepts temperature, pressure, humidity;
    returns is_anomaly and anomaly_score.
    """
    try:
        reading = {
            "temperature": payload.temperature,
            "pressure": payload.pressure,
            "humidity": payload.humidity
        }
        is_anomaly, score = detector.predict(reading)
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 2),
            "station_id": payload.station_id,
            "evaluated_at": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )


@app.post(
    "/api/v1/analyze",
    status_code=status.HTTP_200_OK,
    summary="Comprehensive pipeline: ML, RCA, Severity, XAI, Spatial, and Imputation"
)
def analyze_full_telemetry(payload: FullAnalysisRequest) -> Dict[str, Any]:
    start_time = time.time()
    try:
        target_id = payload.target.station_id
        if target_id not in AWS_STATION_NETWORK:
            target_id = "AWS-01"

        curr_dict = {
            "temperature": payload.target.temperature,
            "pressure": payload.target.pressure,
            "humidity": payload.target.humidity
        }
        history_dicts = [
            {"temperature": h.temperature, "pressure": h.pressure, "humidity": h.humidity}
            for h in payload.history
        ]

        # 1. Stage 1: Isolation Forest Inference
        is_anomaly_ml, ml_score = detector.predict(curr_dict)

        # 2. Stage 2: Deterministic Root-Cause Analysis (RCA)
        rca_report = root_cause_engine.analyze(
            current=curr_dict,
            history=history_dicts,
            ml_is_anomaly=is_anomaly_ml,
            ml_anomaly_score=ml_score
        )

        final_is_anomaly = is_anomaly_ml and not rca_report.is_genuine_weather

        # 3. Transparent 5-Factor Severity & Heuristic Confidence
        severity_result = severity_engine.compute(
            current=curr_dict,
            history=history_dicts,
            ml_anomaly_score=ml_score,
            is_anomaly=final_is_anomaly,
            persistence_streak=1 if final_is_anomaly else 0
        )

        # 4. Explainable AI (XAI) Attribution
        xai_report = xai_engine.explain(
            current=curr_dict,
            history=history_dicts,
            anomaly_score=ml_score,
            is_anomaly=final_is_anomaly,
            root_cause=rca_report.root_cause,
            evidence=rca_report.evidence
        )

        # 5. Spatial Consistency Analysis across peer stations
        network = payload.network_snapshot or {}
        if target_id not in network:
            network[target_id] = curr_dict

        spatial_report = spatial_engine.evaluate_station(target_id, network)

        # 6. Cumulative Sensor Health Tracking
        tracker = station_health_trackers.get(target_id)
        if tracker:
            tracker.record_tick(
                telemetry=curr_dict,
                is_anomaly=final_is_anomaly,
                affected_sensors=severity_result.affected_sensors,
                root_cause=rca_report.root_cause
            )
            health_report = tracker.calculate_health()
        else:
            health_report = None

        # 7. Optional Anomaly Correction Imputation
        clean_history = {
            "temperature": [h["temperature"] for h in history_dicts if h.get("temperature") is not None],
            "pressure": [h["pressure"] for h in history_dicts if h.get("pressure") is not None],
            "humidity": [h["humidity"] for h in history_dicts if h.get("humidity") is not None]
        }
        peer_telemetry = {
            "temperature": [(p.temperature, p.distance_km) for p in spatial_report.peers_evaluated if p.temperature is not None],
            "pressure": [(p.pressure, p.distance_km) for p in spatial_report.peers_evaluated if p.pressure is not None],
            "humidity": [(p.humidity, p.distance_km) for p in spatial_report.peers_evaluated if p.humidity is not None]
        }

        correction_report = correction_engine.generate_report(
            station_id=target_id,
            current_telemetry=curr_dict,
            is_anomaly=final_is_anomaly,
            affected_channels=severity_result.affected_sensors,
            history_clean=clean_history,
            peer_telemetry=peer_telemetry,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        processing_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "station_id": target_id,
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "processing_latency_ms": processing_ms,
            "detection": {
                "is_anomaly": final_is_anomaly,
                "anomaly_score": ml_score,
                "system_state": "GENUINE_EVENT" if rca_report.is_genuine_weather else (
                    severity_result.severity.severity_level if final_is_anomaly else "NORMAL"
                ),
            },
            "root_cause_analysis": {
                "root_cause": rca_report.root_cause,
                "confidence": rca_report.confidence,
                "evidence": rca_report.evidence,
                "classification_category": rca_report.classification_category,
                "is_genuine_weather": rca_report.is_genuine_weather
            },
            "severity": {
                "score": severity_result.severity.total_severity_score,
                "tier": severity_result.severity.severity_level,
                "factors": {
                    "ml_factor": severity_result.severity.ml_score_factor,
                    "deviation_factor": severity_result.severity.deviation_magnitude_factor,
                    "affected_factor": severity_result.severity.affected_sensors_factor,
                    "persistence_factor": severity_result.severity.persistence_factor,
                    "rate_factor": severity_result.severity.rate_of_change_factor
                },
                "affected_sensors": severity_result.affected_sensors,
                "heuristic_confidence": severity_result.heuristic_confidence,
                "confidence_justification": severity_result.confidence_justification
            },
            "explainability": {
                "what_happened": xai_report.what_happened,
                "primary_contributor": xai_report.primary_sensor_contributor,
                "how_unusual": xai_report.how_unusual,
                "inconsistent_sensors": xai_report.inconsistent_sensors,
                "why_classified": xai_report.why_classified,
                "probable_cause": xai_report.probable_cause,
                "operator_actions": xai_report.operator_actions,
                "attributions": [
                    {
                        "feature": a.feature_name,
                        "observed": a.observed_value,
                        "baseline": a.expected_baseline,
                        "delta": a.deviation_delta,
                        "z_score": a.z_score,
                        "contribution_percent": a.contribution_percentage,
                        "unit": a.unit
                    } for a in xai_report.attributions
                ]
            },
            "spatial_consistency": {
                "status": spatial_report.status,
                "classification": spatial_report.classification,
                "confidence": spatial_report.confidence,
                "explanation": spatial_report.explanation,
                "peer_consensus": {
                    "temperature": spatial_report.peer_consensus_temp,
                    "pressure": spatial_report.peer_consensus_press,
                    "humidity": spatial_report.peer_consensus_humidity
                },
                "deltas": {
                    "temp_delta": spatial_report.temp_spatial_delta,
                    "press_delta": spatial_report.press_spatial_delta,
                    "humidity_delta": spatial_report.humidity_spatial_delta
                },
                "peers": [
                    {
                        "station_id": p.station_id,
                        "name": p.name,
                        "distance_km": p.distance_km,
                        "temperature": p.temperature,
                        "pressure": p.pressure,
                        "humidity": p.humidity
                    } for p in spatial_report.peers_evaluated
                ]
            },
            "optional_correction": {
                channel: {
                    "original_value": est.original_value,
                    "suggested_value": est.suggested_value,
                    "unit": est.unit,
                    "confidence": est.confidence,
                    "is_available": est.is_available,
                    "status_message": est.status_message,
                    "methodology": est.estimation_method
                } for channel, est in correction_report.estimates.items()
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive diagnostic error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("[WeatherGuard AI] Starting FastAPI Service on http://127.0.0.1:8000...")
    uvicorn.run("ml_engine.api:app", host="0.0.0.0", port=8000, reload=True)