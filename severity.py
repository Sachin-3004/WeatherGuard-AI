from typing import Tuple

def compute_severity_and_confidence(
    anomaly_score: float,
    root_cause: str,
    affected_sensors: list,
    is_anomaly: bool
) -> Tuple[str, float, float]:
    if not is_anomaly or root_cause == "Possible genuine meteorological event":
        return "LOW", 8.0, 95.0

    sev = anomaly_score * 0.4
    sev += len(affected_sensors) * 15.0
    
    if "spike" in root_cause.lower() or "bounds" in root_cause.lower():
        sev += 25.0
    elif "inconsistency" in root_cause.lower():
        sev += 20.0
    elif "frozen" in root_cause.lower():
        sev += 15.0

    sev = min(100.0, max(0.0, sev))
    
    if sev >= 80.0:
        tier = "CRITICAL"
    elif sev >= 55.0:
        tier = "HIGH"
    elif sev >= 30.0:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    confidence = min(99.0, 85.0 + (len(affected_sensors) * 4.0))
    return tier, round(sev, 1), round(confidence, 1)
