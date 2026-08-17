from pathlib import Path

import joblib
import numpy as np
import pytest

from inference import predict_signal, predict_with_bundles


ROOT = Path(__file__).resolve().parents[1]


def test_inference_rejects_transposed_signal_before_loading_models():
    with pytest.raises(ValueError, match="shape"):
        predict_signal(np.zeros((128, 2)), snr=10)


def test_inference_rejects_non_finite_signal_before_loading_models():
    signal = np.zeros((2, 128))
    signal[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        predict_signal(signal, snr=10)


def test_inference_rejects_non_finite_snr_before_loading_models():
    with pytest.raises(ValueError, match="finite number"):
        predict_signal(np.zeros((2, 128)), snr=np.inf)


def test_bundled_demo_runs_without_full_dataset_or_models():
    samples = np.load(ROOT / "demo/sample_signals.npz")
    classifier = joblib.load(ROOT / "demo/demo_classifier.joblib")
    anomaly = joblib.load(ROOT / "demo/demo_anomaly_detector.joblib")

    result = predict_with_bundles(samples["signals"][0], samples["snrs"][0], classifier, anomaly)

    assert result["predicted_modulation"] in classifier["classes"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert np.isfinite(result["anomaly_score"])
