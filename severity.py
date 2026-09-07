"""
WeatherGuard AI - Deterministic Severity & Heuristic Confidence Engine
SIH Problem Statement: SIH26073

DESIGN PRINCIPLE:
Severity and confidence must NEVER be arbitrary numbers.
This module uses measurable physical and statistical factors to compute:
1. Severity Score [0-100] categorized into: LOW, MEDIUM, HIGH, CRITICAL.
2. Heuristic Diagnostic Confidence [0-100] strictly labeled as heuristic/rule-derived,
   distinct from the statistical ML Isolation Forest certainty.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class SeverityFactorBreakdown:
    # 5 Measurable Factors
    ml_score_factor: float          # Weight: 0.25 (Isolation Forest score)
    deviation_magnitude_factor: float # Weight: 0.25 (Max z-score / relative delta)
    affected_sensors_factor: float  # Weight: 0.20 (Count of compromised channels)
    persistence_factor: float       # Weight: 0.15 (Consecutive ticks anomaly has persisted)
    rate_of_change_factor: float    # Weight: 0.15 (Instantaneous derivative dX/dt)

    total_severity_score: float     # 0 to 100 composite
    severity_level: str             # "LOW", "MEDIUM", "HIGH", "CRITICAL"


@dataclass
class TransparentDiagnosticMetric:
    severity: SeverityFactorBreakdown
    affected_sensors: List[str]     # e.g., ["Temperature", "Pressure"]
    
    # Confidence breakdown
    heuristic_confidence: float     # 0 to 100 percentage
    confidence_type: str            # "HEURISTIC_RULE_EVIDENCE" (strictly labeled)
    confidence_justification: str   # Explicit mathematical reason for confidence level
    ml_model_certainty: float       # Isolation Forest margin certainty [0 to 100]


class TransparentSeverityEngine:
    """
    Computes deterministic severity without arbitrary assignments:
    Severity Formula:
      S = 0.25 * F_ml + 0.25 * F_dev + 0.20 * F_sensors + 0.15 * F_persist + 0.15 * F_rate
    """

    # Physical standard baseline rates of change per tick (approx 2s interval)
    MAX_ALLOWABLE_DELTA_T = 3.0    # °C / tick
    MAX_ALLOWABLE_DELTA_P = 3.5    # hPa / tick
    MAX_ALLOWABLE_DELTA_RH = 15.0  # % / tick

    # Standard terrestrial physical thresholds
    PHYSICAL_BOUNDS = {
        "temperature": (-30.0, 58.0),
        "pressure": (600.0, 1070.0),
        "humidity": (0.0, 100.0)
    }

    def compute(
        self,
        current: Dict[str, Optional[float]],
        history: List[Dict[str, Optional[float]]],
        ml_anomaly_score: float,
        is_anomaly: bool,
        persistence_streak: int = 1
    ) -> TransparentDiagnosticMetric:
        """
        Executes deterministic multi-factor severity and confidence evaluation.
        """
        t = current.get("temperature")
        p = current.get("pressure")
        rh = current.get("humidity")

        affected_sensors: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Identify Affected Sensors & Deviation Magnitude
        # ---------------------------------------------------------------------
        z_scores = []
        recent_window = [h for h in history[-10:] if h is not None]

        # Temperature check
        if t is None or np.isnan(t):
            affected_sensors.append("Temperature (Missing/Null)")
            z_scores.append(4.5)
        else:
            if t < self.PHYSICAL_BOUNDS["temperature"][0] or t > self.PHYSICAL_BOUNDS["temperature"][1]:
                affected_sensors.append("Temperature (Out of Bounds)")
                z_scores.append(5.0)
            elif recent_window:
                temps = [float(h["temperature"]) for h in recent_window if h.get("temperature") is not None]
                if temps:
                    std_t = max(0.4, float(np.std(temps)))
                    mean_t = float(np.mean(temps))
                    z_t = abs(t - mean_t) / std_t
                    z_scores.append(z_t)
                    if z_t >= 2.8:
                        affected_sensors.append("Temperature (Statistical Outlier)")

        # Pressure check
        if p is None or np.isnan(p):
            affected_sensors.append("Pressure (Missing/Null)")
            z_scores.append(4.5)
        else:
            if p < self.PHYSICAL_BOUNDS["pressure"][0] or p > self.PHYSICAL_BOUNDS["pressure"][1]:
                affected_sensors.append("Pressure (Out of Bounds)")
                z_scores.append(5.0)
            elif recent_window:
                press = [float(h["pressure"]) for h in recent_window if h.get("pressure") is not None]
                if press:
                    std_p = max(0.3, float(np.std(press)))
                    mean_p = float(np.mean(press))
                    z_p = abs(p - mean_p) / std_p
                    z_scores.append(z_p)
                    if z_p >= 2.8:
                        affected_sensors.append("Pressure (Statistical Outlier)")

        # Humidity check
        if rh is None or np.isnan(rh):
            affected_sensors.append("Humidity (Missing/Null)")
            z_scores.append(4.5)
        else:
            if rh < self.PHYSICAL_BOUNDS["humidity"][0] or rh > self.PHYSICAL_BOUNDS["humidity"][1]:
                affected_sensors.append("Humidity (Out of Bounds)")
                z_scores.append(5.0)
            elif recent_window:
                hums = [float(h["humidity"]) for h in recent_window if h.get("humidity") is not None]
                if hums:
                    std_rh = max(1.0, float(np.std(hums)))
                    mean_rh = float(np.mean(hums))
                    z_rh = abs(rh - mean_rh) / std_rh
                    z_scores.append(z_rh)
                    if z_rh >= 2.8:
                        affected_sensors.append("Humidity (Statistical Outlier)")

        # ---------------------------------------------------------------------
        # Factor 1: ML Anomaly Score (0 to 100)
        # ---------------------------------------------------------------------
        f_ml = float(np.clip(ml_anomaly_score, 0.0, 100.0))

        # ---------------------------------------------------------------------
        # Factor 2: Deviation Magnitude Factor (0 to 100)
        # A maximum z-score of 5.0 sigma scales to 100%
        # ---------------------------------------------------------------------
        max_z = max(z_scores) if z_scores else 0.0
        f_dev = float(np.clip((max_z / 4.5) * 100.0, 0.0, 100.0))

        # ---------------------------------------------------------------------
        # Factor 3: Affected Sensors Factor (0 to 100)
        # 1 sensor = 33.3%, 2 sensors = 66.7%, 3 sensors = 100%
        # ---------------------------------------------------------------------
        unique_affected_count = len(set([s.split(" ")[0] for s in affected_sensors]))
        f_sensors = float(min(100.0, (unique_affected_count / 3.0) * 100.0))

        # ---------------------------------------------------------------------
        # Factor 4: Persistence Factor (0 to 100)
        # 1 tick = 15%, 4 ticks = 60%, 7+ ticks = 100%
        # ---------------------------------------------------------------------
        if not is_anomaly:
            f_persist = 0.0
        else:
            f_persist = float(min(100.0, max(15.0, persistence_streak * 15.0)))

        # ---------------------------------------------------------------------
        # Factor 5: Rate of Change Factor (0 to 100)
        # ---------------------------------------------------------------------
        rate_ratios = []
        if len(history) >= 1 and history[-1] is not None:
            prev = history[-1]
            if t is not None and prev.get("temperature") is not None:
                rate_ratios.append(abs(t - float(prev["temperature"])) / self.MAX_ALLOWABLE_DELTA_T)
            if p is not None and prev.get("pressure") is not None:
                rate_ratios.append(abs(p - float(prev["pressure"])) / self.MAX_ALLOWABLE_DELTA_P)
            if rh is not None and prev.get("humidity") is not None:
                rate_ratios.append(abs(rh - float(prev["humidity"])) / self.MAX_ALLOWABLE_DELTA_RH)

        max_rate_ratio = max(rate_ratios) if rate_ratios else 0.0
        f_rate = float(np.clip(max_rate_ratio * 50.0, 0.0, 100.0))

        # ---------------------------------------------------------------------
        # Composite Severity Score
        # ---------------------------------------------------------------------
        if not is_anomaly:
            total_severity = float(np.clip(f_ml * 0.15, 0.0, 24.0))
        else:
            total_severity = float(np.clip(
                0.25 * f_ml +
                0.25 * f_dev +
                0.20 * f_sensors +
                0.15 * f_persist +
                0.15 * f_rate,
                10.0,
                100.0
            ))

        # Tier classification
        if total_severity >= 80.0:
            severity_level = "CRITICAL"
        elif total_severity >= 55.0:
            severity_level = "HIGH"
        elif total_severity >= 30.0:
            severity_level = "MEDIUM"
        else:
            severity_level = "LOW"

        breakdown = SeverityFactorBreakdown(
            ml_score_factor=round(f_ml, 1),
            deviation_magnitude_factor=round(f_dev, 1),
            affected_sensors_factor=round(f_sensors, 1),
            persistence_factor=round(f_persist, 1),
            rate_of_change_factor=round(f_rate, 1),
            total_severity_score=round(total_severity, 1),
            severity_level=severity_level
        )

        # ---------------------------------------------------------------------
        # Heuristic Diagnostic Confidence vs ML Statistical Certainty
        # ---------------------------------------------------------------------
        evidence_points = 0
        justifications = []

        if len(recent_window) >= 8:
            evidence_points += 25
            justifications.append("Historical baseline window full (>= 8 samples)")
        else:
            justifications.append(f"Small historical baseline window ({len(recent_window)} samples)")

        if unique_affected_count >= 1:
            evidence_points += 30
            justifications.append(f"Direct sensor deviation verified on {unique_affected_count} channel(s)")

        if max_z >= 3.0:
            evidence_points += 25
            justifications.append(f"Extreme statistical departure confirmed (z = {max_z:.1f})")

        if persistence_streak >= 2:
            evidence_points += 20
            justifications.append(f"Persistence confirmed across {persistence_streak} consecutive ticks")

        heuristic_confidence = float(min(98.0, max(45.0, evidence_points)))
        ml_certainty = float(abs(ml_anomaly_score - 50.0) * 2.0)  # Distance from decision boundary

        clean_sensor_names = list(set([s.split(" ")[0] for s in affected_sensors])) or ["None (Nominal)"]

        return TransparentDiagnosticMetric(
            severity=breakdown,
            affected_sensors=clean_sensor_names,
            heuristic_confidence=round(heuristic_confidence, 1),
            confidence_type="HEURISTIC_RULE_EVIDENCE",
            confidence_justification="; ".join(justifications),
            ml_model_certainty=round(ml_certainty, 1)
        )


severity_engine = TransparentSeverityEngine()