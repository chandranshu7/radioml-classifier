"""Compare the original feature set with a reduced non-redundant set."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import FEATURE_NAMES  # noqa: E402


REDUCED_FEATURE_NAMES = [
    name
    for name in FEATURE_NAMES
    if name not in {"mean_amplitude", "rms_amplitude", "amplitude_variance"}
]


def evaluate_feature_set(train, test, feature_names):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    started = time.perf_counter()
    model.fit(train[feature_names], train["modulation"])
    training_seconds = time.perf_counter() - started
    predictions = model.predict(test[feature_names])

    result = {
        "feature_count": len(feature_names),
        "overall_accuracy": accuracy_score(test["modulation"], predictions),
        "accuracy_snr_ge_0": accuracy_score(
            test.loc[test.snr >= 0, "modulation"], predictions[test.snr.to_numpy() >= 0]
        ),
        "accuracy_snr_ge_6": accuracy_score(
            test.loc[test.snr >= 6, "modulation"], predictions[test.snr.to_numpy() >= 6]
        ),
        "training_seconds": training_seconds,
    }
    return model, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/radioml_features.parquet")
    parser.add_argument("--split", type=Path, default=ROOT / "models/split_ids.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "results/reduced_feature_experiment.csv")
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    split = np.load(args.split)
    train = table[table.signal_id.isin(split["train_ids"])]
    test = table[table.signal_id.isin(split["test_ids"])]

    rows = []
    for name, features in [("original_15", FEATURE_NAMES), ("reduced_12", REDUCED_FEATURE_NAMES)]:
        model, metrics = evaluate_feature_set(train, test, features)
        with Path("/private/tmp", f"{name}.joblib").open("wb") as file:
            joblib.dump(model, file)
        metrics["feature_set"] = name
        metrics["serialized_size_mb"] = Path("/private/tmp", f"{name}.joblib").stat().st_size / 1024**2
        rows.append(metrics)

    results = pd.DataFrame(rows)[
        ["feature_set", "feature_count", "overall_accuracy", "accuracy_snr_ge_0",
         "accuracy_snr_ge_6", "training_seconds", "serialized_size_mb"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)

    plot = results.set_index("feature_set")[["overall_accuracy", "accuracy_snr_ge_0", "accuracy_snr_ge_6"]]
    ax = plot.plot(kind="bar", figsize=(8, 4.8), rot=0)
    ax.set(title="Original vs reduced RF feature set", ylabel="Accuracy", ylim=(0, 1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["All SNR", "SNR >= 0 dB", "SNR >= 6 dB"])
    ax.figure.tight_layout()
    ax.figure.savefig(args.output.with_suffix(".png"), dpi=180)
    plt.close(ax.figure)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
