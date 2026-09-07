from typing import Dict, Any

class XAIEngine:
    def explain_deviation(self, reading: Dict[str, float], baseline: Dict[str, float] = None) -> Dict[str, Any]:
        if baseline is None:
            baseline = {"temperature": 30.0, "pressure": 1010.0, "humidity": 60.0}

        t = reading.get("temperature", 30.0) if reading.get("temperature") is not None else 30.0
        p = reading.get("pressure", 1010.0) if reading.get("pressure") is not None else 1010.0
        rh = reading.get("humidity", 60.0) if reading.get("humidity") is not None else 60.0

        # Standardized deviations based on climatological spread
        dev_t = abs(t - baseline["temperature"]) / 6.5
        dev_p = abs(p - baseline["pressure"]) / 3.0
        dev_rh = abs(rh - baseline["humidity"]) / 8.0

        total_dev = dev_t + dev_p + dev_rh + 1e-6
        contrib_t = round((dev_t / total_dev) * 100, 1)
        contrib_p = round((dev_p / total_dev) * 100, 1)
        contrib_rh = round((dev_rh / total_dev) * 100, 1)

        scores = {"temperature": contrib_t, "pressure": contrib_p, "humidity": contrib_rh}
        top_sensor = max(scores, key=scores.get)

        return {
            "attribution_method": "Standardized Feature-Deviation Decomposition",
            "contributions": {
                "temperature": contrib_t,
                "pressure": contrib_p,
                "humidity": contrib_rh
            },
            "primary_contributor": top_sensor,
            "operator_recommendation": f"Inspect {top_sensor} sensor wiring and run field diagnostic test."
        }

    def explain(self, *args, **kwargs) -> Dict[str, Any]:
        """Alias so both .explain() and .explain_deviation() work seamlessly."""
        if args and isinstance(args[0], dict):
            return self.explain_deviation(args[0])
        return self.explain_deviation(kwargs)

# Export the exact singleton instance that api.py line 28 imports
xai_engine = XAIEngine()

# Export standalone function for backward compatibility
explain_deviation = xai_engine.explain_deviation