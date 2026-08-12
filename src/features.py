"""Interpretable RF feature extraction for RadioML I/Q signals."""

from __future__ import annotations

import numpy as np
from scipy.stats import kurtosis


FEATURE_NAMES = [
    "mean_amplitude",
    "std_amplitude",
    "rms_amplitude",
    "max_amplitude",
    "mean_power",
    "amplitude_variance",
    "phase_std",
    "i_std",
    "q_std",
    "iq_correlation",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "fft_peak_magnitude",
    "amplitude_kurtosis",
]


def extract_features_batch(iq_signals: np.ndarray) -> dict[str, np.ndarray]:
    """Extract 15 features from an array shaped (n_signals, 2, n_samples)."""
    signals = np.asarray(iq_signals, dtype=np.float64)
    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError("Expected I/Q signals shaped (n_signals, 2, n_samples).")

    i_samples = signals[:, 0, :]
    q_samples = signals[:, 1, :]
    complex_signals = i_samples + 1j * q_samples
    amplitude = np.abs(complex_signals)
    power = amplitude**2
    phase = np.angle(complex_signals)

    # Correlation is covariance divided by the product of standard deviations.
    i_centered = i_samples - i_samples.mean(axis=1, keepdims=True)
    q_centered = q_samples - q_samples.mean(axis=1, keepdims=True)
    correlation_denominator = np.sqrt(
        np.sum(i_centered**2, axis=1) * np.sum(q_centered**2, axis=1)
    )
    iq_correlation = np.divide(
        np.sum(i_centered * q_centered, axis=1),
        correlation_denominator,
        out=np.zeros(signals.shape[0]),
        where=correlation_denominator > 0,
    )

    spectrum = np.fft.fftshift(np.fft.fft(complex_signals, axis=1), axes=1)
    spectral_power = np.abs(spectrum) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(signals.shape[2]))
    total_spectral_power = spectral_power.sum(axis=1)
    safe_total = np.where(total_spectral_power > 0, total_spectral_power, 1.0)
    probabilities = spectral_power / safe_total[:, None]

    spectral_centroid = np.sum(probabilities * frequencies[None, :], axis=1)
    spectral_bandwidth = np.sqrt(
        np.sum(
            probabilities * (frequencies[None, :] - spectral_centroid[:, None]) ** 2,
            axis=1,
        )
    )
    # Compute log2 only for positive probabilities to avoid log2(0) warnings.
    log_probabilities = np.zeros_like(probabilities)
    np.log2(probabilities, out=log_probabilities, where=probabilities > 0)
    spectral_entropy = -np.sum(probabilities * log_probabilities, axis=1) / np.log2(
        signals.shape[2]
    )

    amplitude_kurtosis = kurtosis(amplitude, axis=1, fisher=True, bias=False)
    amplitude_kurtosis = np.nan_to_num(amplitude_kurtosis)

    return {
        "mean_amplitude": amplitude.mean(axis=1),
        "std_amplitude": amplitude.std(axis=1),
        "rms_amplitude": np.sqrt(power.mean(axis=1)),
        "max_amplitude": amplitude.max(axis=1),
        "mean_power": power.mean(axis=1),
        "amplitude_variance": amplitude.var(axis=1),
        "phase_std": phase.std(axis=1),
        "i_std": i_samples.std(axis=1),
        "q_std": q_samples.std(axis=1),
        "iq_correlation": iq_correlation,
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_entropy": spectral_entropy,
        "fft_peak_magnitude": np.abs(spectrum).max(axis=1) / signals.shape[2],
        "amplitude_kurtosis": amplitude_kurtosis,
    }


def extract_signal_features(iq_signal: np.ndarray) -> dict[str, float]:
    """Extract the same features from one signal shaped (2, n_samples)."""
    signal = np.asarray(iq_signal)
    if signal.ndim != 2 or signal.shape[0] != 2:
        raise ValueError("Expected one I/Q signal shaped (2, n_samples).")
    batch_features = extract_features_batch(signal[None, :, :])
    return {name: float(values[0]) for name, values in batch_features.items()}
