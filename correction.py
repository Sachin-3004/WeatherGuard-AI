from typing import Dict, Any, List, Optional
import numpy as np

class AnomalyCorrectionEngine:
    def estimate_correction(
        self,
        target_sensor: str,
        history: List[Dict[str, float]],
        peer_readings: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if peer_readings is None:
            peer_readings = []

        valid_temporal = [
            h[target_sensor] for h in history 
            if isinstance(h, dict) and h.get(target_sensor) is not None
        ]
        valid_spatial = [
            p["reading"][target_sensor] for p in peer_readings 
            if isinstance(p, dict) and "reading" in p and p["reading"].get(target_sensor) is not None
        ]

        if len(valid_temporal) < 2 and len(valid_spatial) < 2:
            return {
                "available": False,
                "estimated_value": None,
                "confidence": 0.0,
                "reason": "Correction unavailable – insufficient evidence."
            }

        estimates = []
        if valid_spatial:
            estimates.append(float(np.median(valid_spatial)))
        if valid_temporal:
            estimates.append(float(np.median(valid_temporal[-5:])))

        suggested = float(np.mean(estimates))
        confidence = 90.0 if (valid_spatial and valid_temporal) else 75.0

        return {
            "available": True,
            "estimated_value": round(suggested, 1),
            "confidence": confidence,
            "basis": "Spatial Peer Consensus + Temporal Rolling Median"
        }

    def estimate(self, *args, **kwargs) -> Dict[str, Any]:
        return self.estimate_correction(*args, **kwargs)

# Export both the singleton instance and function name
correction_engine = AnomalyCorrectionEngine()
estimate_correction = correction_engine.estimate_correction