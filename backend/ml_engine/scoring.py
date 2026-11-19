import numpy as np

def calculate_calibrated_score(raw_decision: float) -> float:
    score = 100.0 / (1.0 + np.exp(raw_decision * 8.5))
    return float(np.clip(score, 0.0, 100.0))
