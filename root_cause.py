"""
WeatherGuard AI - Root-Cause Analysis (RCA) & Diagnostic Engine
SIH Problem Statement: SIH26073

ARCHITECTURAL PRINCIPLE:
This is a dedicated, post-detection diagnostic layer.
The Isolation Forest (Stage 1) determines IF a sample is an anomaly.
This Diagnostic Engine (Stage 2) determines WHY the anomaly occurred,
classifying it into deterministic categories with physical evidence and confidence.

Classifications:
1. Temperature sensor spike
2. Pressure sensor anomaly
3. Humidity sensor anomaly
4. Frozen/stuck sensor
5. Missing data
6. Communication/data transmission issue
7. Multivariate inconsistency
8. Possible genuine meteorological event
9. Unknown anomaly
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class DiagnosticResult:
    root_cause: str
    confidence: int       # 0 to 100 percentage
    evidence: str
    classification_category: str
    is_genuine_weather: bool = False


def calculate_magnus_dew_point(temp_c: float, rh_pct: float) -> float:
    """
    Magnus-Tetens empirical approximation for dew point temperature ($T_d$).
    Physical Law: $T_d \le T_{dry}$ under all non-supersaturated atmospheric conditions.
    """
    if rh_pct <= 0:
        return -50.0
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + math.log(rh_pct / 100.0)
    return (b * alpha) / (a - alpha)


class RootCauseAnalyzer:
    """
    Deterministic inference engine for telemetry fault identification.
    Uses physics bounds, temporal gradients ($\Delta/ \Delta t$), rolling window variances,
    and multi-channel covariance signatures.
    """

    # World Meteorological Organization (WMO) No. 8 surface physical boundaries
    TEMP_BOUNDS = (-30.0, 58.0)       # Celsius
    PRESSURE_BOUNDS = (600.0, 1070.0)  # hPa (surface to high-elevation)
    HUMIDITY_BOUNDS = (0.0, 100.0)     # %

    # Maximum permissible natural rates of change between consecutive readings (~1-2 min polling)
    MAX_NATURAL_DELTA_TEMP = 3.0       # °C / step
    MAX_NATURAL_DELTA_PRESSURE = 3.5   # hPa / step
    MAX_NATURAL_DELTA_HUMIDITY = 18.0  # % / step

    def analyze(
        self,
        current: Dict[str, Optional[float]],
        history: List[Dict[str, Optional[float]]],
        ml_is_anomaly: bool,
        ml_anomaly_score: float
    ) -> DiagnosticResult:
        """
        Executes deterministic root-cause analysis on an observation.
        """
        t = current.get("temperature")
        p = current.get("pressure")
        rh = current.get("humidity")

        # ---------------------------------------------------------------------
        # Category A: Missing Data / Communication Transmission Issue
        # ---------------------------------------------------------------------
        missing_fields = []
        if t is None or (isinstance(t, float) and math.isnan(t)):
            missing_fields.append("Temperature")
        if p is None or (isinstance(p, float) and math.isnan(p)):
            missing_fields.append("Pressure")
        if rh is None or (isinstance(rh, float) and math.isnan(rh)):
            missing_fields.append("Humidity")

        if len(missing_fields) >= 2:
            return DiagnosticResult(
                root_cause="Communication/data transmission issue",
                confidence=98,
                evidence=(
                    f"Simultaneous loss of {len(missing_fields)} channels ({', '.join(missing_fields)}). "
                    f"Indicates telemetry packet drop, ADC transmission CRC mismatch, or RS-485 bus power loss."
                ),
                classification_category="Communication/data transmission issue"
            )
        elif len(missing_fields) == 1:
            return DiagnosticResult(
                root_cause="Missing data",
                confidence=96,
                evidence=(
                    f"Field '{missing_fields[0]}' missing or unreadable in telemetry frame while other channels "
                    f"transmitted valid packets. Suspected transducer cable disconnect or isolated ADC channel failure."
                ),
                classification_category="Missing data"
            )

        # Ensure values are typed as floats for downstream mathematical rules
        temp_val: float = float(t)
        press_val: float = float(p)
        rh_val: float = float(rh)

        # ---------------------------------------------------------------------
        # Category B: Genuine Meteorological Event Discrimination
        # Thunderstorm / Convective Squall Line / Gust Front (Cold Pool Dynamics):
        # A sharp temperature drop accompanied by a coherent humidity jump and barometric jump.
        # ---------------------------------------------------------------------
        if len(history) >= 2:
            prev_recent = history[-2] if len(history) >= 3 else history[-1]
            if (prev_recent.get("temperature") is not None and 
                prev_recent.get("pressure") is not None and 
                prev_recent.get("humidity") is not None):
                
                dt = temp_val - float(prev_recent["temperature"])
                dp = press_val - float(prev_recent["pressure"])
                drh = rh_val - float(prev_recent["humidity"])

                # Squall signature: Temp plunges >= 3.5°C, RH surges >= 15%, pressure shifts coherently
                if dt <= -3.2 and drh >= 14.0 and abs(dp) >= 1.0:
                    return DiagnosticResult(
                        root_cause="Possible genuine meteorological event",
                        confidence=93,
                        evidence=(
                            f"Multi-variable covariance: Ambient temperature plummeted by {dt:.1f}°C while humidity "
                            f"surged by +{drh:.1f}% and barometric pressure shifted by {dp:+.1f} hPa. "
                            f"Correlates with an active convective downdraft / gust front (cold pool) rather than sensor failure."
                        ),
                        classification_category="Possible genuine meteorological event",
                        is_genuine_weather=True
                    )

        # ---------------------------------------------------------------------
        # Category C: Frozen / Stuck Sensor Detection
        # Check rolling variance across the last 5-8 valid consecutive samples
        # ---------------------------------------------------------------------
        if len(history) >= 5:
            recent_window = history[-5:] + [current]
            valid_temps = [float(h["temperature"]) for h in recent_window if h.get("temperature") is not None]
            valid_press = [float(h["pressure"]) for h in recent_window if h.get("pressure") is not None]
            valid_rh = [float(h["humidity"]) for h in recent_window if h.get("humidity") is not None]

            t_var = max(valid_temps) - min(valid_temps) if len(valid_temps) >= 5 else 1.0
            p_var = max(valid_press) - min(valid_press) if len(valid_press) >= 5 else 1.0
            rh_var = max(valid_rh) - min(valid_rh) if len(valid_rh) >= 5 else 1.0

            if t_var < 0.0001:
                return DiagnosticResult(
                    root_cause="Frozen/stuck sensor",
                    confidence=95,
                    evidence=(
                        f"Temperature sensor output flatlined at exactly {temp_val:.2f}°C across "
                        f"{len(valid_temps)} consecutive polling ticks with zero Brownian micro-noise variance "
                        f"(ΔT = {t_var:.5f}°C). Thermistor ADC register locked."
                    ),
                    classification_category="Frozen/stuck sensor"
                )
            if p_var < 0.0001:
                return DiagnosticResult(
                    root_cause="Frozen/stuck sensor",
                    confidence=96,
                    evidence=(
                        f"Barometric pressure transducer output constant at {press_val:.2f} hPa with 0.00 variance "
                        f"across {len(valid_press)} samples. Barometric diaphragm or I2C buffer hang suspected."
                    ),
                    classification_category="Frozen/stuck sensor"
                )
            if rh_var < 0.0001:
                return DiagnosticResult(
                    root_cause="Frozen/stuck sensor",
                    confidence=94,
                    evidence=(
                        f"Relative humidity transducer locked at {rh_val:.2f}% across {len(valid_rh)} samples. "
                        f"Hygrometer microcontroller communication deadlock detected."
                    ),
                    classification_category="Frozen/stuck sensor"
                )

        # ---------------------------------------------------------------------
        # Category D: Multivariate Inconsistency (Thermodynamic Enthalpy Violations)
        # ---------------------------------------------------------------------
        dew_point = calculate_magnus_dew_point(temp_val, rh_val)
        # Dew point cannot exceed dry-bulb temperature (Magnus Law)
        if dew_point > (temp_val + 0.5) and rh_val <= 100.0:
            return DiagnosticResult(
                root_cause="Multivariate inconsistency",
                confidence=96,
                evidence=(
                    f"Calculated Dew Point ({dew_point:.1f}°C) exceeds ambient Dry Bulb Temperature ({temp_val:.1f}°C) "
                    f"by {(dew_point - temp_val):.1f}°C at {rh_val:.1f}% RH. "
                    f"Violates Magnus-Tetens thermodynamic vapor pressure equilibrium."
                ),
                classification_category="Multivariate inconsistency"
            )

        # Extreme enthalpy clash: High ambient heat with tropical saturation (Impossible wet-bulb > 38°C)
        if temp_val > 44.0 and rh_val > 80.0:
            return DiagnosticResult(
                root_cause="Multivariate inconsistency",
                confidence=94,
                evidence=(
                    f"Unphysical thermodynamic enthalpy pairing: Temperature of {temp_val:.1f}°C combined with "
                    f"{rh_val:.1f}% Relative Humidity yields an impossible surface wet-bulb temperature (>39°C)."
                ),
                classification_category="Multivariate inconsistency"
            )

        # ---------------------------------------------------------------------
        # Category E: Step Spikes and Channel-Isolated Deviations
        # ---------------------------------------------------------------------
        prev_reading = history[-1] if history else None
        if prev_reading and prev_reading.get("temperature") is not None and prev_reading.get("pressure") is not None and prev_reading.get("humidity") is not None:
            delta_t = temp_val - float(prev_reading["temperature"])
            delta_p = press_val - float(prev_reading["pressure"])
            delta_rh = rh_val - float(prev_reading["humidity"])

            # 1. Temperature Sensor Spike
            if abs(delta_t) >= self.MAX_NATURAL_DELTA_TEMP and abs(delta_p) < 2.0 and abs(delta_rh) < 10.0:
                return DiagnosticResult(
                    root_cause="Temperature sensor spike",
                    confidence=94,
                    evidence=(
                        f"Temperature changed unusually quickly by {delta_t:+.1f}°C in one polling cycle "
                        f"while humidity (ΔRH = {delta_rh:+.1f}%) and pressure (ΔP = {delta_p:+.1f} hPa) "
                        f"remained relatively stable. Indicates an electrical transient or thermistor short circuit."
                    ),
                    classification_category="Temperature sensor spike"
                )

            # 2. Pressure Sensor Anomaly
            if abs(delta_p) >= self.MAX_NATURAL_DELTA_PRESSURE and abs(delta_t) < 1.5:
                return DiagnosticResult(
                    root_cause="Pressure sensor anomaly",
                    confidence=92,
                    evidence=(
                        f"Barometric pressure shifted abruptly by {delta_p:+.1f} hPa in a single step "
                        f"while ambient temperature remained steady (ΔT = {delta_t:+.1f}°C). "
                        f"Baroclinic atmospheric fronts cannot cause instantaneous step jumps of this magnitude."
                    ),
                    classification_category="Pressure sensor anomaly"
                )

            # 3. Humidity Sensor Anomaly
            if abs(delta_rh) >= self.MAX_NATURAL_DELTA_HUMIDITY and abs(delta_t) < 1.5:
                return DiagnosticResult(
                    root_cause="Humidity sensor anomaly",
                    confidence=91,
                    evidence=(
                        f"Relative humidity jumped abruptly by {delta_rh:+.1f}% without any corresponding "
                        f"temperature gradient or barometric trough. Suspected water droplet splash on capacitive substrate."
                    ),
                    classification_category="Humidity sensor anomaly"
                )

        # ---------------------------------------------------------------------
        # Category F: Out-of-Bounds Physical Saturation
        # ---------------------------------------------------------------------
        if temp_val < self.TEMP_BOUNDS[0] or temp_val > self.TEMP_BOUNDS[1]:
            return DiagnosticResult(
                root_cause="Temperature sensor spike",
                confidence=97,
                evidence=(
                    f"Observed temperature of {temp_val:.1f}°C breaches planetary terrestrial bounds "
                    f"[{self.TEMP_BOUNDS[0]}°C, {self.TEMP_BOUNDS[1]}°C]. Hardware amplifier saturation."
                ),
                classification_category="Temperature sensor spike"
            )

        if rh_val < self.HUMIDITY_BOUNDS[0] or rh_val > self.HUMIDITY_BOUNDS[1]:
            return DiagnosticResult(
                root_cause="Humidity sensor anomaly",
                confidence=96,
                evidence=(
                    f"Relative humidity reading of {rh_val:.1f}% exceeds physical saturation range [0%, 100%]. "
                    f"Capacitive hygrometer open/short circuit."
                ),
                classification_category="Humidity sensor anomaly"
            )

        if press_val < self.PRESSURE_BOUNDS[0] or press_val > self.PRESSURE_BOUNDS[1]:
            return DiagnosticResult(
                root_cause="Pressure sensor anomaly",
                confidence=95,
                evidence=(
                    f"Barometric pressure reading of {press_val:.1f} hPa is outside valid ground station physical limits "
                    f"[{self.PRESSURE_BOUNDS[0]} hPa, {self.PRESSURE_BOUNDS[1]} hPa]."
                ),
                classification_category="Pressure sensor anomaly"
            )

        # ---------------------------------------------------------------------
        # Category G: Isolation Forest Outlier / Unknown Anomaly
        # ---------------------------------------------------------------------
        if ml_is_anomaly:
            return DiagnosticResult(
                root_cause="Unknown anomaly",
                confidence=max(60, int(ml_anomaly_score)),
                evidence=(
                    f"Isolation Forest flagged a multi-dimensional statistical outlier (anomaly score {ml_anomaly_score:.1f}/100) "
                    f"at coordinates (T={temp_val:.1f}°C, P={press_val:.1f} hPa, RH={rh_val:.1f}%). "
                    f"No single deterministic hardware rule was triggered; combination exhibits abnormal joint distribution."
                ),
                classification_category="Unknown anomaly"
            )

        # Nominal State
        return DiagnosticResult(
            root_cause="None (Nominal Operation)",
            confidence=98,
            evidence="All parameters adhere to physical bounds, rate-of-change limits, and thermodynamic dew-point equations.",
            classification_category="Nominal"
        )


# Singleton instance for downstream API endpoints
root_cause_engine = RootCauseAnalyzer()