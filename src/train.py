"""Train Logistic Regression and Random Forest modulation classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURE_NAMES


RANDOM_STATE = 42


def make_split(table: pd.DataFrame, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Split IDs while preserving every modulation/SNR combination."""
    signal_ids = table["signal_id"].to_numpy()
    strata = table["modulation"].astype(str) + "__snr_" + table["snr"].astype(str)
    return train_test_split(
        signal_ids,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=strata,
    )


def train_models(table: pd.DataFrame, train_ids: np.ndarray) -> dict:
    """Fit both classifiers on the chosen training rows."""
    train_rows = table[table["signal_id"].isin(train_ids)]
    x_train = train_rows[FEATURE_NAMES]
    y_train = train_rows["modulation"]

    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
            ),
        ]
    )
    random_forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print("Training Logistic Regression...")
    logistic_regression.fit(x_train, y_train)
    print("Training Random Forest...")
    random_forest.fit(x_train, y_train)

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "feature_names": FEATURE_NAMES,
        "classes": sorted(y_train.unique().tolist()),
        "random_state": RANDOM_STATE,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "radioml_features.parquet",
    )
    parser.add_argument("--models-dir", type=Path, default=project_root / "models")
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    train_ids, test_ids = make_split(table)
    bundle = train_models(table, train_ids)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.models_dir / "classifiers.joblib")
    np.savez_compressed(
        args.models_dir / "split_ids.npz", train_ids=train_ids, test_ids=test_ids
    )
    print(f"Training rows: {len(train_ids):,}")
    print(f"Test rows: {len(test_ids):,}")
    print(f"Saved models: {(args.models_dir / 'classifiers.joblib').resolve()}")


if __name__ == "__main__":
    main()
