from typing import Dict, Any
from collections import defaultdict

class SensorHealthTracker:
    def __init__(self):
        self.health_scores = {"temperature": 100.0, "pressure": 100.0, "humidity": 100.0}
        self.stats = {
            "temperature": {"anomalies": 0, "spikes": 0, "frozen": 0, "missing": 0},
            "pressure": {"anomalies": 0, "spikes": 0, "frozen": 0, "missing": 0},
            "humidity": {"anomalies": 0, "spikes": 0, "frozen": 0, "missing": 0}
        }

    def record_reading(self, reading: Dict[str, Any], is_anomaly: bool, fault_type: str):
        for sensor in ["temperature", "pressure", "humidity"]:
            if reading.get(sensor) is None:
                self.stats[sensor]["missing"] += 1
                self.health_scores[sensor] = max(10.0, self.health_scores[sensor] - 15.0)
            elif is_anomaly and sensor in fault_type.lower():
                self.stats[sensor]["anomalies"] += 1
                if "spike" in fault_type.lower():
                    self.stats[sensor]["spikes"] += 1
                    self.health_scores[sensor] = max(15.0, self.health_scores[sensor] - 12.0)
                elif "frozen" in fault_type.lower():
                    self.stats[sensor]["frozen"] += 1
                    self.health_scores[sensor] = max(15.0, self.health_scores[sensor] - 10.0)
            else:
                # Gradual recovery on normal ticks
                self.health_scores[sensor] = min(100.0, self.health_scores[sensor] + 0.5)

    def get_tier(self, score: float) -> str:
        if score >= 85.0: return "GOOD"
        if score >= 65.0: return "WARNING"
        if score >= 40.0: return "DEGRADED"
        return "CRITICAL"

    def get_all_tiers(self) -> Dict[str, str]:
        return {s: self.get_tier(self.health_scores[s]) for s in ["temperature", "pressure", "humidity"]}


# Multi-station dictionary that automatically creates a tracker for any station ID
station_health_trackers = defaultdict(SensorHealthTracker)

# Default single instance
health_tracker = SensorHealthTracker()