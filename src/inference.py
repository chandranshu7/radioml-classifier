"""Run modulation classification and anomaly detection on one I/Q capture."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_loader import load_radioml
from features import extract_signal_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "classifiers.joblib"
DEFAULT_ANOMALY_PATH = PROJECT_ROOT / "models" / "anomaly_detector.joblib"


@lru_cache(maxsize=1)
def load_models(
    classifier_path: str = str(DEFAULT_CLASSIFIER_PATH),
    anomaly_path: str = str(DEFAULT_ANOMALY_PATH),
) -> tuple[dict, dict]:
    """Load trained artifacts once and reuse them across predictions."""
    return joblib.load(classifier_path), joblib.load(anomaly_path)


def predict_signal(iq_signal: np.ndarray, snr: float) -> dict:
    """Classify and anomaly-score one RadioML-style I/Q signal.

    ``snr`` is returned as measurement context; it is not a model input.
    """
    signal = np.asarray(iq_signal)
    if signal.shape != (2, 128):
        raise ValueError(f"Expected iq_signal shape (2, 128), received {signal.shape}.")
    if not np.isfinite(signal).all():
        raise ValueError("iq_signal contains NaN or infinite values.")
    if not np.isfinite(snr):
        raise ValueError("snr must be a finite number.")

    classifier_bundle, anomaly_bundle = load_models()
    feature_values = extract_signal_features(signal)
    feature_names = classifier_bundle["feature_names"]
    if feature_names != anomaly_bundle["feature_names"]:
        raise ValueError("Classifier and anomaly detector expect different features.")
    feature_row = pd.DataFrame(
        [[feature_values[name] for name in feature_names]], columns=feature_names
    )

    classifier = classifier_bundle["random_forest"]
    predicted_modulation = str(classifier.predict(feature_row)[0])
    probabilities = classifier.predict_proba(feature_row)[0]
    confidence = float(np.max(probabilities))

    detector = anomaly_bundle["detector"]
    anomaly_score = float(detector.decision_function(feature_row)[0])
    is_anomaly = bool(detector.predict(feature_row)[0] == -1)

    return {
        "predicted_modulation": predicted_modulation,
        "confidence": confidence,
        "anomaly_score": anomaly_score,
        "is_anomaly": is_anomaly,
        "snr_db": float(snr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one RadioML example.")
    parser.add_argument("dataset", type=Path, help="Path to the trusted RadioML pickle")
    parser.add_argument("--modulation", default="QPSK")
    parser.add_argument("--snr", type=int, default=18)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    dataset = load_radioml(args.dataset)
    key = (args.modulation, args.snr)
    if key not in dataset:
        raise ValueError(f"Dataset does not contain {key}.")
    signal = dataset[key][args.sample_index]
    result = predict_signal(signal, args.snr)
    print(f"True modulation: {args.modulation}")
    for name, value in result.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
