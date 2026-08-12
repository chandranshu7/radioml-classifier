"""Convert each RadioML I/Q signal into a row of RF features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_radioml  # noqa: E402
from features import FEATURE_NAMES, extract_features_batch  # noqa: E402


def build_feature_table(dataset: dict[tuple[str, int], np.ndarray]) -> pd.DataFrame:
    """Build one table row per signal, processing one dictionary block at a time."""
    frames: list[pd.DataFrame] = []
    next_signal_id = 0

    for modulation, snr_db in sorted(dataset, key=lambda key: (key[0], key[1])):
        block = np.asarray(dataset[(modulation, snr_db)])
        features = extract_features_batch(block)
        frame = pd.DataFrame(features)
        frame.insert(0, "snr", int(snr_db))
        frame.insert(0, "signal_id", np.arange(next_signal_id, next_signal_id + len(block)))
        frame["modulation"] = modulation
        frames.append(frame)
        next_signal_id += len(block)

    columns = ["signal_id", "snr", *FEATURE_NAMES, "modulation"]
    return pd.concat(frames, ignore_index=True)[columns]


def validate_feature_table(table: pd.DataFrame, expected_rows: int) -> None:
    """Fail clearly if extraction produced missing/infinite values or wrong counts."""
    if len(table) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(table)}.")
    numeric = table.drop(columns="modulation").to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Feature table contains NaN or infinite values.")
    if not table["spectral_entropy"].between(0, 1).all():
        raise ValueError("Normalized spectral entropy should be between 0 and 1.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the trusted RadioML .pkl file")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "radioml_features.parquet",
    )
    args = parser.parse_args()

    dataset = load_radioml(args.dataset)
    expected_rows = sum(len(block) for block in dataset.values())
    table = build_feature_table(dataset)
    validate_feature_table(table, expected_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)

    print(f"Rows: {len(table):,}")
    print(f"Columns: {len(table.columns)} ({len(FEATURE_NAMES)} extracted features + metadata)")
    print(f"Missing values: {int(table.isna().sum().sum())}")
    print("\nFirst 5 rows:")
    print(table.head().to_string(index=False))
    print(f"\nSaved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
