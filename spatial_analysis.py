from typing import Dict, Any, List
import numpy as np

# Station network definitions expected by api.py
AWS_STATION_NETWORK = {
    "AWS-101": {"name": "Delhi Ridge", "lat": 28.67, "lon": 77.22, "elev": 216},
    "AWS-102": {"name": "Gurugram South", "lat": 28.45, "lon": 77.02, "elev": 220},
    "AWS-103": {"name": "Noida East", "lat": 28.53, "lon": 77.39, "elev": 200},
    "AWS-104": {"name": "Faridabad", "lat": 28.40, "lon": 77.31, "elev": 205},
    "AWS-105": {"name": "Rohtak Ridge", "lat": 28.89, "lon": 76.60, "elev": 220},
}

class SpatialAnalysisEngine:
    def analyze_spatial_consistency(
        self,
        target_station_id: str,
        target_reading: Dict[str, float],
        peer_readings: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if peer_readings is None:
            peer_readings = []

        t = target_reading.get("temperature")
        if t is None or not peer_readings:
            return {
                "status": "CONSISTENT",
                "classification": "INSUFFICIENT_PEERS",
                "explanation": "Not enough peer stations active for spatial consensus."
            }

        peer_temps = [
            p["reading"]["temperature"]
            for p in peer_readings
            if isinstance(p, dict) and "reading" in p and p["reading"].get("temperature") is not None
        ]

        if len(peer_temps) < 2:
            return {
                "status": "CONSISTENT",
                "classification": "INSUFFICIENT_PEERS",
                "explanation": "Fewer than 2 valid neighboring stations available."
            }

        median_t = float(np.median(peer_temps))
        delta_t = abs(t - median_t)

        if delta_t > 8.0:
            supporting_peers = sum(1 for pt in peer_temps if abs(pt - t) < 3.0)
            if supporting_peers >= 1:
                return {
                    "status": "REGIONAL_EVENT",
                    "classification": "REGIONAL_WEATHER_EVENT",
                    "explanation": f"Station {target_station_id} ({t}°C) deviates from spatial median ({median_t:.1f}°C), but is corroborated by nearby peer stations. Regional gradient detected."
                }
            else:
                return {
                    "status": "INCONSISTENT",
                    "classification": "ISOLATED_SENSOR_FAULT",
                    "explanation": f"Station {target_station_id} reports {t}°C while neighboring stations report median {median_t:.1f}°C (Δ={delta_t:+.1f}°C). No peer corroboration found."
                }

        return {
            "status": "CONSISTENT",
            "classification": "NOMINAL_SPATIAL_COHERENCE",
            "explanation": f"Station matches spatial consensus (peer median: {median_t:.1f}°C)."
        }

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Wrapper method to handle both .analyze() and .analyze_spatial_consistency() calls."""
        return self.analyze_spatial_consistency(*args, **kwargs)

# Export the singleton instance expected by api.py
spatial_engine = SpatialAnalysisEngine()

# Export standalone function for compatibility
analyze_spatial_consistency = spatial_engine.analyze_spatial_consistency