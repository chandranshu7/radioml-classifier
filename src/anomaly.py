"""Train and evaluate an Isolation Forest on RF features."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data_loader import load_radioml
from features import FEATURE_NAMES, extract_features_batch


RANDOM_STATE = 42


def train_anomaly_detector(
    feature_table: pd.DataFrame, train_ids: np.ndarray, sample_size: int = 50_000
) -> IsolationForest:
    """Fit on a reproducible sample of ordinary training signals."""
    training_rows = feature_table[feature_table["signal_id"].isin(train_ids)]
    if len(training_rows) > sample_size:
        training_rows = training_rows.sample(sample_size, random_state=RANDOM_STATE)

    detector = IsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.02,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    detector.fit(training_rows[FEATURE_NAMES])
    return detector


def choose_reference_signals(
    dataset: dict[tuple[str, int], np.ndarray], count: int = 300
) -> np.ndarray:
    """Choose a mixed, reproducible set of clear signals for artificial changes."""
    candidates = np.concatenate(
        [dataset[key] for key in sorted(dataset) if int(key[1]) >= 10], axis=0
    )
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(candidates), size=count, replace=False)
    return np.asarray(candidates[indices], dtype=np.float64)


def make_artificial_anomalies(normal_signals: np.ndarray) -> dict[str, np.ndarray]:
    """Create three easy-to-understand deviations from the reference signals."""
    rng = np.random.default_rng(RANDOM_STATE)
    complex_signals = normal_signals[:, 0, :] + 1j * normal_signals[:, 1, :]

    signal_rms = np.sqrt(np.mean(np.abs(complex_signals) ** 2, axis=1, keepdims=True))
    noise = (
        rng.normal(size=complex_signals.shape) + 1j * rng.normal(size=complex_signals.shape)
    ) / np.sqrt(2)
    noisy = complex_signals + 3.0 * signal_rms * noise
    high_amplitude = 3.0 * complex_signals

    sample_index = np.arange(complex_signals.shape[1])
    rotation = np.exp(1j * 2 * np.pi * 0.20 * sample_index)
    frequency_offset = complex_signals * rotation[None, :]

    def to_iq(signals: np.ndarray) -> np.ndarray:
        return np.stack((signals.real, signals.imag), axis=1)

    return {
        "normal": normal_signals,
        "increased_noise": to_iq(noisy),
        "high_amplitude": to_iq(high_amplitude),
        "frequency_offset": to_iq(frequency_offset),
    }


def score_groups(
    detector: IsolationForest, groups: dict[str, np.ndarray]
) -> pd.DataFrame:
    """Extract features and return scores/predictions for every experiment group."""
    frames: list[pd.DataFrame] = []
    for group_name, signals in groups.items():
        features = pd.DataFrame(extract_features_batch(signals))[FEATURE_NAMES]
        frames.append(
            pd.DataFrame(
                {
                    "group": group_name,
                    "example_id": np.arange(len(signals)),
                    "anomaly_score": detector.decision_function(features),
                    "prediction": detector.predict(features),
                }
            )
        )
    result = pd.concat(frames, ignore_index=True)
    result["is_anomaly"] = result["prediction"] == -1
    return result


def save_score_plot(scores: pd.DataFrame, path: Path) -> None:
    order = ["normal", "increased_noise", "high_amplitude", "frequency_offset"]
    values = [scores.loc[scores["group"] == name, "anomaly_score"] for name in order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(values, tick_labels=[name.replace("_", "\n") for name in order])
    ax.axhline(0, color="tab:red", linestyle="--", label="Anomaly threshold")
    ax.set(title="Isolation Forest scores by signal group", ylabel="Decision score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_signal_comparison(groups: dict[str, np.ndarray], path: Path) -> None:
    """Compare amplitude and FFT for one reference and its three modifications."""
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    for row, (name, signals) in enumerate(groups.items()):
        complex_signal = signals[0, 0] + 1j * signals[0, 1]
        amplitude = np.abs(complex_signal)
        spectrum = np.fft.fftshift(np.fft.fft(complex_signal))
        frequency = np.fft.fftshift(np.fft.fftfreq(complex_signal.size))
        magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

        axes[row, 0].plot(amplitude)
        axes[row, 0].set(title=f"{name}: amplitude", xlabel="Sample", ylabel="Magnitude")
        axes[row, 1].plot(frequency, magnitude_db)
        axes[row, 1].set(
            title=f"{name}: FFT", xlabel="Normalized frequency", ylabel="Magnitude (dB)"
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the trusted RadioML pickle")
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "radioml_features.parquet",
    )
    parser.add_argument(
        "--split", type=Path, default=project_root / "models" / "split_ids.npz"
    )
    parser.add_argument("--models-dir", type=Path, default=project_root / "models")
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    split = np.load(args.split)
    detector = train_anomaly_detector(table, split["train_ids"])

    dataset = load_radioml(args.dataset)
    normal_signals = choose_reference_signals(dataset)
    groups = make_artificial_anomalies(normal_signals)
    scores = score_groups(detector, groups)

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"detector": detector, "feature_names": FEATURE_NAMES},
        args.models_dir / "anomaly_detector.joblib",
    )
    scores.to_csv(args.output_dir / "anomaly_scores.csv", index=False)
    save_score_plot(scores, args.output_dir / "anomaly_scores.png")
    save_signal_comparison(groups, args.output_dir / "anomaly_signal_comparison.png")

    summary = scores.groupby("group").agg(
        mean_score=("anomaly_score", "mean"),
        anomaly_rate=("is_anomaly", "mean"),
    )
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved detector: {(args.models_dir / 'anomaly_detector.joblib').resolve()}")
    print(f"Saved experiment: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
