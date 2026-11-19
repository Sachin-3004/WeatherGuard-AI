# WeatherGuard AI — SIH26073

**AI/ML-Based Intelligent Anomaly Detection & Quality Control for Automatic Weather Stations (AWS)**

WeatherGuard AI is a full-stack meteorological telemetry validation engine built for the Smart India Hackathon (SIH26073). It combines **unsupervised machine learning (`scikit-learn` Isolation Forest)** with **WMO thermodynamic physics** and **mesoscale spatial consensus** to detect sensor anomalies in real time, distinguish hardware failures from genuine extreme weather events, and provide explainable root-cause diagnostics.

---

## System Architecture

```text
[AWS Telemetry Stream] 
       │
       ▼
[Data Preprocessing & WMO Bounds Validation]
       │
       ▼
[Stage 1: scikit-learn Isolation Forest (Unsupervised Scoring)]
       │
       ▼
[Stage 2: Deterministic Physics & Root-Cause Engine]
  ├── Magnus-Tetens Dew Point Verification
  ├── Zero-Variance Frozen ADC Detection
  ├── Rate-of-Change Step Shock Analysis
  └── Mesoscale Spatial Peer Consensus (Genuine Storm vs. Isolated Fault)
       │
       ▼
[5-Factor Severity & Heuristic Evidence Confidence]
       │
       ▼
[Explainable AI (XAI) & Non-Destructive Imputation] ──► [React Operations Dashboard]
```
Quick Start Guide

​1. Start the Python ML Backend
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
python -m ml_engine.training
python -m uvicorn ml_engine.api:app --host 127.0.0.1 --port 8000 --reload
```

2. Start the React Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173/ in your browser.
​API Reference
​POST /api/v1/detect
​Evaluates an incoming AWS telemetry packet.

​Request Payload:
```json
{
  "station_id": "AWS-101",
  "temperature": 55.0,
  "pressure": 1012.4,
  "humidity": 54.0,
  "history": [],
  "peer_stations": []
}
```

Response:
```json
{
  "station_id": "AWS-101",
  "is_anomaly": true,
  "anomaly_score": 94.0,
  "status": "ANOMALY DETECTED",
  "severity": "CRITICAL",
  "confidence": 89.0,
  "root_cause": "Temperature sensor spike",
  "evidence": "Temperature of 55.0°C breaches physical terrestrial limits (-20°C to 60°C)."
}
```
