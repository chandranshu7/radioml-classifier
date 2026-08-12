"""Loading and summary helpers for RadioML 2016.10a."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def load_radioml(path: str | Path) -> dict[tuple[str, int], np.ndarray]:
    """Load a trusted RadioML pickle file.

    Pickle files can execute code while loading, so only use this function with a
    dataset you trust. ``latin1`` keeps compatibility with Python 2-era pickles.
    """
    path = Path(path).expanduser()
    with path.open("rb") as file:
        dataset = pickle.load(file, encoding="latin1")

    if not isinstance(dataset, dict):
        raise ValueError("Expected a dictionary keyed by (modulation, SNR).")
    return dataset


def summarize_dataset(dataset: dict[tuple[str, int], np.ndarray]) -> dict:
    """Return the main facts needed to understand the dataset."""
    modulations = sorted({modulation for modulation, _ in dataset})
    snrs = sorted({int(snr) for _, snr in dataset})
    shapes = sorted({tuple(np.asarray(block).shape) for block in dataset.values()})

    return {
        "modulations": modulations,
        "snrs_db": snrs,
        "block_shapes": shapes,
        "total_signals": sum(np.asarray(block).shape[0] for block in dataset.values()),
        "signals_per_block": sorted(
            {np.asarray(block).shape[0] for block in dataset.values()}
        ),
        "sample_shape": tuple(np.asarray(next(iter(dataset.values())))[0].shape),
        "dtypes": sorted({str(np.asarray(block).dtype) for block in dataset.values()}),
    }
