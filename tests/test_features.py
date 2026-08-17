import numpy as np
import pytest

from features import FEATURE_NAMES, extract_features_batch, extract_signal_features


def test_single_and_batch_extraction_match():
    rng = np.random.default_rng(7)
    signal = rng.normal(size=(2, 128))

    single = extract_signal_features(signal)
    batch = extract_features_batch(signal[None, :, :])

    assert list(single) == FEATURE_NAMES
    for name in FEATURE_NAMES:
        assert single[name] == pytest.approx(batch[name][0])


def test_zero_signal_produces_finite_features():
    features = extract_signal_features(np.zeros((2, 128)))
    assert np.isfinite(list(features.values())).all()
    assert 0 <= features["spectral_entropy"] <= 1


@pytest.mark.parametrize("shape", [(128, 2), (1, 2, 128)])
def test_single_signal_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError):
        extract_signal_features(np.zeros(shape))


def test_feature_extractor_supports_other_window_lengths():
    features = extract_signal_features(np.zeros((2, 64)))
    assert list(features) == FEATURE_NAMES


def test_batch_rejects_invalid_shape():
    with pytest.raises(ValueError):
        extract_features_batch(np.zeros((4, 128, 2)))
