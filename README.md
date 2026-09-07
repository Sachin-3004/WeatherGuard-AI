WeatherGuard AI — Machine Learning Architecture GuideSIH Problem Statement: SIH26073Target: Automatic Weather Station (AWS) Telemetry Anomaly Detection1. Why Isolation Forest?In weather station networks (like India Meteorological Department AWS nodes), sensors send readings every few minutes. We choose Isolation Forest for this initial model because:Unsupervised Formulation: Field telemetry lacks labeled historical ground truth for hardware failures, bird droppings, or transient EMI glitches.Isolation Principle: Outliers in weather data represent extreme or isolated states in 3D parameter space $(T, P, RH)$. Isolation trees cut features randomly; sparse outliers are isolated near the root of the tree (short average path length $h(x)$), whereas dense normal patterns take many random cuts to isolate.Low Latency: Once trees are constructed, scoring a 3D vector takes $< 2 \text{ ms}$, ideal for streaming IoT feeds.2. File Organization & Architectureml_engine/
├── __init__.py
├── config.py           # Hyperparameters, physical limits, and file paths
├── preprocessing.py    # StandardScaler pipeline & physical boundary validation
├── training.py         # Climatological data generation & Isolation Forest training
├── scoring.py          # Calibrates raw decision function into a 0–100 risk score
├── inference.py        # Model loading & singleton prediction engine
├── api.py              # FastAPI endpoint matching required input/output JSON
└── README.md           # This educational guide
3. How the Mathematical Transformation WorksDecision Function:In scikit-learn, IsolationForest.decision_function(X) returns:$$\text{score}(x) = -2^{-\frac{E(h(x))}{c(n)}} - \text{offset}$$Positive score $\implies$ Inlier (Normal weather)Negative score $\implies$ Outlier (Sensor Spike / Glitch)Anomaly Score Normalization:Judges and operators cannot easily interpret raw floats like -0.142. We pass the decision score $s$ through an inverted, scaled sigmoid:$$\text{Anomaly Score} = \frac{100}{1 + e^{k \cdot s}}$$With sensitivity $k = 18$:If $s = +0.15$ (deep inside normal cluster), Score $\approx 6\%$ (Very Low Risk).If $s = 0.00$ (on the decision threshold), Score $= 50\%$ (Borderline).If $s = -0.15$ (abnormal outlier), Score $\approx 94\%$ (Critical Anomaly).4. How to Test and RunStep 1: Install Dependenciespip install fastapi uvicorn scikit-learn numpy pandas joblib pydantic
Step 2: Train the Model Artifactspython -m ml_engine.training
This outputs artifacts/isolation_forest.joblib and artifacts/scaler.joblib.Step 3: Launch the APIpython -m ml_engine.api
Step 4: Test with curl or Postman1. Nominal Reading (Should return is_anomaly: false):curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
     -H "Content-Type: application/json" \
     -d '{"temperature": 31.5, "pressure": 1008.2, "humidity": 56.0}'
Response:{
  "is_anomaly": false,
  "anomaly_score": 5.42
}
2. Thermal Spike Anomaly (Should return is_anomaly: true):curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
     -H "Content-Type: application/json" \
     -d '{"temperature": 56.8, "pressure": 1008.0, "humidity": 50.0}'
Response:{
  "is_anomaly": true,
  "anomaly_score": 99.9
}
3. Dew Point / Physics Conflict Anomaly:curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
     -H "Content-Type: application/json" \
     -d '{"temperature": 48.0, "pressure": 1010.0, "humidity": 96.0}'
Response:{
  "is_anomaly": true,
  "anomaly_score": 88.75
}
