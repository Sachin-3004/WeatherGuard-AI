import math
from typing import Dict, Any, List

def calculate_dew_point(t: float, rh: float) -> float:
    if rh <= 0:
        return -50.0
    a, b = 17.27, 237.7
    alpha = ((a * t) / (b + t)) + math.log(rh / 100.0)
    return (b * alpha) / (a - alpha)

def diagnose_root_cause(
    current: Dict[str, float],
    history: List[Dict[str, float]],
    spatial_inconsistent: bool = False
) -> Dict[str, Any]:
    t, p, rh = current.get("temperature"), current.get("pressure"), current.get("humidity")
    
    if t is None or p is None or rh is None:
        return {
            "root_cause": "Missing data",
            "confidence": 99.0,
            "evidence": "One or more telemetry packets were null or dropped during transmission."
        }
    
    if t < -20.0 or t > 60.0:
        return {
            "root_cause": "Temperature sensor spike",
            "confidence": 98.0,
            "evidence": f"Temperature of {t}°C breaches physical terrestrial limits (-20°C to 60°C)."
        }
    if rh < 0.0 or rh > 100.0:
        return {
            "root_cause": "Humidity sensor anomaly",
            "confidence": 97.0,
            "evidence": f"Relative humidity {rh}% is outside the physical [0, 100%] saturation scale."
        }
    if p < 600.0 or p > 1080.0:
        return {
            "root_cause": "Pressure sensor anomaly",
            "confidence": 96.0,
            "evidence": f"Barometric pressure {p} hPa breaches terrestrial bounds."
        }

    if len(history) >= 4:
        recent_t = [h["temperature"] for h in history[-4:] if h.get("temperature") is not None] + [t]
        recent_p = [h["pressure"] for h in history[-4:] if h.get("pressure") is not None] + [p]
        recent_rh = [h["humidity"] for h in history[-4:] if h.get("humidity") is not None] + [rh]

        if len(recent_t) >= 4 and max(recent_t) == min(recent_t):
            return {
                "root_cause": "Frozen/stuck sensor",
                "confidence": 94.0,
                "evidence": "Temperature sensor reported 4 consecutive identical readings with zero variance."
            }
        if len(recent_p) >= 4 and max(recent_p) == min(recent_p):
            return {
                "root_cause": "Frozen/stuck sensor",
                "confidence": 94.0,
                "evidence": "Barometric pressure transducer register locked with zero variance."
            }
        if len(recent_rh) >= 4 and max(recent_rh) == min(recent_rh):
            return {
                "root_cause": "Frozen/stuck sensor",
                "confidence": 93.0,
                "evidence": "Hygrometer output buffer flatlined across consecutive polling cycles."
            }

    if history:
        prev = history[-1]
        if prev.get("temperature") is not None:
            dt = abs(t - prev["temperature"])
            if dt > 5.5:
                return {
                    "root_cause": "Temperature sensor spike",
                    "confidence": 93.0,
                    "evidence": f"Temperature jumped by {dt:.1f}°C in a single cycle. Exceeds max physical thermal rate."
                }
        if prev.get("pressure") is not None:
            dp = abs(p - prev["pressure"])
            if dp > 6.0:
                return {
                    "root_cause": "Pressure sensor anomaly",
                    "confidence": 92.0,
                    "evidence": f"Barometric shift of {dp:.1f} hPa occurred without an atmospheric pressure wave."
                }

    dew_pt = calculate_dew_point(t, rh)
    if dew_pt > t + 0.8:
        return {
            "root_cause": "Multivariate inconsistency",
            "confidence": 95.0,
            "evidence": f"Calculated dew point ({dew_pt:.1f}°C) exceeds dry-bulb temperature ({t:.1f}°C). Thermodynamic impossibility."
        }

    if len(history) >= 2:
        ref = history[-2]
        if ref.get("temperature") is not None and ref.get("humidity") is not None and ref.get("pressure") is not None:
            dt = t - ref["temperature"]
            drh = rh - ref["humidity"]
            dp = abs(p - ref["pressure"])
            if dt <= -3.5 and drh >= 14.0 and dp >= 1.0:
                return {
                    "root_cause": "Possible genuine meteorological event",
                    "confidence": 92.0,
                    "evidence": f"Simultaneous temperature plunge ({dt:.1f}°C), humidity surge (+{drh:.1f}%), and pressure wave indicates a convective gust front."
                }

    return {
        "root_cause": "Unknown anomaly",
        "confidence": 75.0,
        "evidence": "Multivariate vector fell into sparse isolation hyperplanes without a single deterministic boundary violation."
    }
