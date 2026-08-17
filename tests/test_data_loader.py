import pickle

import numpy as np
import pytest

from data_loader import load_radioml, summarize_dataset


def test_loader_and_summary(tmp_path):
    dataset = {
        ("BPSK", 0): np.zeros((3, 2, 128), dtype=np.float32),
        ("QPSK", 2): np.ones((3, 2, 128), dtype=np.float32),
    }
    path = tmp_path / "small.pkl"
    with path.open("wb") as file:
        pickle.dump(dataset, file)

    loaded = load_radioml(path)
    summary = summarize_dataset(loaded)

    assert summary["modulations"] == ["BPSK", "QPSK"]
    assert summary["snrs_db"] == [0, 2]
    assert summary["total_signals"] == 6
    assert summary["sample_shape"] == (2, 128)
    assert summary["dtypes"] == ["float32"]


def test_loader_rejects_non_dictionary_pickle(tmp_path):
    path = tmp_path / "wrong.pkl"
    with path.open("wb") as file:
        pickle.dump([1, 2, 3], file)

    with pytest.raises(ValueError, match="dictionary"):
        load_radioml(path)
