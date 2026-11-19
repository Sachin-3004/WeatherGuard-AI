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
