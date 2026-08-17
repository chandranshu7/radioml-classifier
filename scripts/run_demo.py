"""Run the compact demo classifier on one RadioML example."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_radioml  # noqa: E402
from features import extract_signal_features  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to a trusted RadioML pickle")
    parser.add_argument("--modulation", default="QPSK")
    parser.add_argument("--snr", type=int, default=18)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "demo" / "demo_classifier.joblib"
    )
    args = parser.parse_args()

    dataset = load_radioml(args.dataset)
    signal = dataset[(args.modulation, args.snr)][args.sample_index]
    bundle = joblib.load(args.model)
    features = extract_signal_features(signal)
    row = pd.DataFrame(
        [[features[name] for name in bundle["feature_names"]]],
        columns=bundle["feature_names"],
    )
    model = bundle["model"]
    probabilities = model.predict_proba(row)[0]
    prediction = model.classes_[probabilities.argmax()]

    print(f"True modulation: {args.modulation}")
    print(f"Predicted modulation: {prediction}")
    print(f"Confidence: {probabilities.max():.3f}")


if __name__ == "__main__":
    main()
