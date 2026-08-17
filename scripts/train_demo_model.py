"""Train the compact classifier artifact used by the bundled demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_NAMES  # noqa: E402


DEMO_FEATURES = [
    name
    for name in FEATURE_NAMES
    if name not in {"mean_amplitude", "rms_amplitude", "amplitude_variance"}
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/radioml_features.parquet")
    parser.add_argument("--split", type=Path, default=ROOT / "models/split_ids.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "demo")
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    split = np.load(args.split)
    train = table[table.signal_id.isin(split["train_ids"])]
    test = table[table.signal_id.isin(split["test_ids"])]

    model = RandomForestClassifier(
        n_estimators=30,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(train[DEMO_FEATURES], train["modulation"])
    predictions = model.predict(test[DEMO_FEATURES])
    overall = accuracy_score(test["modulation"], predictions)
    high_mask = test.snr.to_numpy() >= 6
    high_snr = accuracy_score(test.loc[test.snr >= 6, "modulation"], predictions[high_mask])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "demo_classifier.joblib"
    joblib.dump(
        {"model": model, "feature_names": DEMO_FEATURES, "classes": list(model.classes_)},
        artifact,
        compress=3,
    )
    metadata = {
        "purpose": "Compact bundled demo; not the primary reported model",
        "feature_count": len(DEMO_FEATURES),
        "n_estimators": 30,
        "max_depth": 12,
        "overall_accuracy": overall,
        "accuracy_snr_ge_6": high_snr,
        "size_mb": artifact.stat().st_size / 1024**2,
        "scikit_learn_version": __import__("sklearn").__version__,
    }
    (args.output_dir / "demo_model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
