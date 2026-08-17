"""Streamlit dashboard for exploring RadioML predictions."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_radioml  # noqa: E402
from inference import predict_signal, predict_with_bundles  # noqa: E402


DEFAULT_DATASET = os.getenv(
    "RADIOML_DATASET", str(PROJECT_ROOT / "data" / "RML2016.10a_dict_optimized.pkl")
)


@st.cache_resource(show_spinner="Loading RadioML dataset...")
def cached_dataset(path: str) -> dict:
    """Keep the large read-only dataset in memory between Streamlit reruns."""
    return load_radioml(path)


@st.cache_data
def bundled_samples() -> dict:
    """Load the small signal gallery shipped with the repository."""
    archive = np.load(PROJECT_ROOT / "demo" / "sample_signals.npz")
    dataset = {}
    for signal, modulation, snr in zip(
        archive["signals"], archive["modulations"], archive["snrs"]
    ):
        dataset.setdefault((str(modulation), int(snr)), []).append(signal)
    return {key: np.stack(value) for key, value in dataset.items()}


@st.cache_resource
def bundled_models() -> tuple[dict, dict]:
    classifier = joblib.load(PROJECT_ROOT / "demo" / "demo_classifier.joblib")
    anomaly = joblib.load(PROJECT_ROOT / "demo" / "demo_anomaly_detector.joblib")
    return classifier, anomaly


def signal_figure(iq_signal: np.ndarray) -> plt.Figure:
    """Plot I, Q, amplitude, and FFT for the selected example."""
    i_samples, q_samples = iq_signal
    complex_signal = i_samples + 1j * q_samples
    amplitude = np.abs(complex_signal)
    spectrum = np.fft.fftshift(np.fft.fft(complex_signal))
    frequency = np.fft.fftshift(np.fft.fftfreq(complex_signal.size))
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0, 0].plot(i_samples, label="I")
    axes[0, 0].plot(q_samples, label="Q", alpha=0.8)
    axes[0, 0].set(title="I/Q waveforms", xlabel="Sample", ylabel="Value")
    axes[0, 0].legend()

    axes[0, 1].plot(amplitude, color="tab:green")
    axes[0, 1].set(title="Amplitude", xlabel="Sample", ylabel="Magnitude")

    axes[1, 0].scatter(i_samples, q_samples, s=14, alpha=0.7)
    axes[1, 0].set(title="I/Q scatter", xlabel="I", ylabel="Q", aspect="equal")

    axes[1, 1].plot(frequency, magnitude_db, color="tab:red")
    axes[1, 1].set(
        title="FFT magnitude",
        xlabel="Normalized frequency",
        ylabel="Magnitude (dB)",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="RadioML Explorer", layout="wide")
    st.title("RadioML Explorer")
    st.caption("I/Q features, modulation classification, and anomaly detection")

    source = st.sidebar.radio(
        "Signal source", ["Bundled signal gallery", "Full RadioML dataset"]
    )
    demo_mode = source == "Bundled signal gallery"
    if demo_mode:
        dataset = bundled_samples()
        st.info(
            "Demo mode uses 33 representative signals and the compact 2.4 MB model. "
            "Select the full dataset in the sidebar to use locally rebuilt artifacts."
        )
    else:
        dataset_path = st.sidebar.text_input("RadioML pickle path", value=DEFAULT_DATASET)
        path = Path(dataset_path).expanduser()
        if not path.is_file():
            st.error("Dataset file not found. Update the path in the sidebar.")
            st.stop()
        dataset = cached_dataset(str(path))
    modulations = sorted({modulation for modulation, _ in dataset})
    modulation = st.sidebar.selectbox("True modulation", modulations, index=modulations.index("QPSK"))
    available_snrs = sorted(snr for mod, snr in dataset if mod == modulation)
    snr = st.sidebar.selectbox("SNR (dB)", available_snrs, index=len(available_snrs) - 1)
    sample_count = len(dataset[(modulation, snr)])
    if sample_count == 1:
        sample_index = 0
        st.sidebar.caption("One bundled example is available for this selection.")
    else:
        sample_index = st.sidebar.slider("Sample index", 0, sample_count - 1, 0)

    iq_signal = dataset[(modulation, snr)][sample_index]
    if demo_mode:
        classifier_bundle, anomaly_bundle = bundled_models()
        result = predict_with_bundles(
            iq_signal, snr, classifier_bundle, anomaly_bundle
        )
    else:
        result = predict_signal(iq_signal, snr)

    st.subheader("Selected signal")
    st.pyplot(signal_figure(iq_signal), clear_figure=True)

    st.subheader("Model output")
    first, second, third, fourth = st.columns(4)
    first.metric("True modulation", modulation)
    second.metric("Prediction", result["predicted_modulation"])
    third.metric("Confidence", f"{result['confidence']:.1%}")
    fourth.metric("Anomaly score", f"{result['anomaly_score']:.3f}")
    if result["is_anomaly"]:
        st.warning("Isolation Forest flagged this signal as anomalous.")
    else:
        st.success("Isolation Forest considers this signal normal.")

    st.subheader("Evaluation context")
    chart_column, matrix_column = st.columns(2)
    accuracy_path = PROJECT_ROOT / "results" / "accuracy_by_snr.csv"
    confusion_path = PROJECT_ROOT / "results" / "random_forest_confusion_matrix.png"
    with chart_column:
        st.markdown("**Accuracy by SNR**")
        accuracy = pd.read_csv(accuracy_path).pivot(index="snr", columns="model", values="accuracy")
        accuracy.index.name = "SNR (dB)"
        st.line_chart(accuracy)
    with matrix_column:
        st.markdown("**Random Forest confusion matrix**")
        st.image(str(confusion_path))

    st.caption(
        "Confidence is the Random Forest's maximum class probability. "
        "The anomaly score is positive for more-normal signals and negative for anomalies. "
        "Charts below report the full 100-tree model, not the compact dashboard model."
    )


if __name__ == "__main__":
    main()
