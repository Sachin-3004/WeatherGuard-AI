"""
WeatherGuard AI - Anomaly Scoring & Interpretation
SIH Problem Statement: SIH26073

Understanding Isolation Forest Outputs:
- `model.decision_function(X)` computes the anomaly score:
  - Higher positive numbers (e.g. +0.15 to +0.25): Centered deep inside normal clusters.
  - Values near 0.0: Near the decision boundary (set by contamination hyperparameter).
  - Negative numbers (e.g. -0.10 to -0.35): Isolated early in the trees; severe anomalies.

This module provides a calibrated mapping from the raw continuous decision function
into a human-interpretable percentage score $[0.0, 100.0]$:
  - 0%   = Perfectly nominal observation
  - 50%  = Right at the classification boundary
  - 100% = Severe, definitive anomaly
"""

import numpy as np


def compute_calibrated_anomaly_score(raw_decision_score: float) -> float:
    """
    Converts raw Isolation Forest decision score into an intuitive 0-100 anomaly score.

    Mathematical Transformation:
    Standard scikit-learn decision_function formula:
        decision_score = score_func(X) - offset_
    Points with negative values are classified as anomalies (-1).
    Points with positive values are classified as inliers (1).

    We use a calibrated logistic sigmoid transformation inverted on the decision score:
        S(x) = 1 / (1 + exp(k * x))
    where k is a scaling factor tuned to spread the sensitivity across the margin.
    """
    # Sensitivity factor k controls the transition sharpness around the boundary (score = 0)
    k = 18.0
    
    # Inverted sigmoid: as decision_score drops (negative), anomaly probability surges towards 1
    prob = 1.0 / (1.0 + np.exp(k * raw_decision_score))
    
    # Scale to percentage [0.0, 100.0] and round to 2 decimal places
    calibrated_score = float(np.round(np.clip(prob * 100.0, 0.0, 100.0), 2))
    return calibrated_score


def evaluate_isolation_forest_result(raw_pred: int, raw_decision: float) -> tuple[bool, float]:
    """
    Translates raw scikit-learn outputs into the API requirements:
    - raw_pred: 1 (inlier/normal) or -1 (outlier/anomaly)
    - raw_decision: continuous float
    Returns:
    - is_anomaly (bool): True if classified as an anomaly, False otherwise
    - anomaly_score (float): Normalized score [0.0 - 100.0]
    """
    is_anomaly = bool(raw_pred == -1)
    anomaly_score = compute_calibrated_anomaly_score(raw_decision)

    # Edge case alignment: If raw_pred is -1 (anomaly), ensure anomaly_score >= 50.0
    if is_anomaly and anomaly_score < 50.0:
        anomaly_score = 50.0 + (50.0 - anomaly_score)

    return is_anomaly, anomaly_score