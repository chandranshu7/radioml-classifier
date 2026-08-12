"""Inspect RadioML and plot a representative I/Q signal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_radioml, summarize_dataset  # noqa: E402


def plot_signal(iq_signal: np.ndarray, modulation: str, snr_db: int, output: Path) -> None:
    """Plot time, constellation, and frequency views of one signal."""
    i_samples = iq_signal[0]
    q_samples = iq_signal[1]
    complex_signal = i_samples + 1j * q_samples
    amplitude = np.abs(complex_signal)

    # fftshift places negative frequencies on the left and positive on the right.
    spectrum = np.fft.fftshift(np.fft.fft(complex_signal))
    frequency = np.fft.fftshift(np.fft.fftfreq(complex_signal.size))
    magnitude_db = 20 * np.log10(np.abs(spectrum) + 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    sample_index = np.arange(iq_signal.shape[1])

    axes[0, 0].plot(sample_index, i_samples, color="tab:blue")
    axes[0, 0].set(title="I waveform", xlabel="Sample index", ylabel="I value")

    axes[0, 1].plot(sample_index, q_samples, color="tab:orange")
    axes[0, 1].set(title="Q waveform", xlabel="Sample index", ylabel="Q value")

    axes[0, 2].plot(sample_index, amplitude, color="tab:green")
    axes[0, 2].set(title="Amplitude over time", xlabel="Sample index", ylabel="|I + jQ|")

    axes[1, 0].scatter(i_samples, q_samples, s=18, alpha=0.7)
    axes[1, 0].axhline(0, color="0.7", linewidth=0.8)
    axes[1, 0].axvline(0, color="0.7", linewidth=0.8)
    axes[1, 0].set(title="I/Q scatter", xlabel="I", ylabel="Q", aspect="equal")

    axes[1, 1].plot(frequency, magnitude_db, color="tab:red")
    axes[1, 1].set(
        title="FFT magnitude",
        xlabel="Normalized frequency (cycles/sample)",
        ylabel="Magnitude (dB)",
    )

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.05,
        0.95,
        f"Modulation: {modulation}\nSNR: {snr_db} dB\nShape: {iq_signal.shape}\n"
        f"Samples: {iq_signal.shape[1]}",
        va="top",
        fontsize=13,
    )

    fig.suptitle(f"RadioML example: {modulation} at {snr_db} dB", fontsize=16)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to the trusted RadioML .pkl file")
    parser.add_argument("--modulation", default="QPSK")
    parser.add_argument("--snr", type=int, default=18)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "signal_overview.png",
    )
    args = parser.parse_args()

    dataset = load_radioml(args.dataset)
    summary = summarize_dataset(dataset)
    print("RadioML dataset summary")
    for name, value in summary.items():
        print(f"{name}: {value}")

    key = (args.modulation, args.snr)
    if key not in dataset:
        raise ValueError(f"No block for {key}. Choose from the printed classes/SNRs.")
    if not 0 <= args.sample_index < dataset[key].shape[0]:
        raise IndexError("sample-index is outside this block.")

    plot_signal(dataset[key][args.sample_index], *key, args.output)
    print(f"Saved plot: {args.output.resolve()}")


if __name__ == "__main__":
    main()
