# RadioML modulation classifier

A classical machine-learning baseline for modulation recognition on RadioML
2016.10a. The project converts short I/Q captures into 15 RF features, compares
Logistic Regression with Random Forest, measures accuracy across SNR, and uses
Isolation Forest as a lightweight out-of-distribution check.

No neural networks are used.

## Results

The models were evaluated on a fixed 80/20 split stratified by modulation and SNR.
SNR is used for analysis, not as a model input.

| Model | Test accuracy |
|---|---:|
| Logistic Regression | 44.45% |
| Random Forest | 50.44% |

The aggregate score includes captures from -20 dB to 18 dB. Random Forest accuracy
is close to chance at -20 dB and reaches about 79% at 18 dB. Spectral entropy and
spectral bandwidth are the two most useful features in the fitted forest.

## Repository layout

```text
.
├── app.py                    # Streamlit UI
├── scripts/
│   ├── explore.py            # dataset summary and I/Q plots
│   └── build_features.py     # feature table generation
├── src/
│   ├── data_loader.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── anomaly.py
│   └── inference.py
├── data/                     # local dataset and feature table
├── models/                   # fitted model artifacts
└── results/                  # metrics and plots
```

## Setup

Python 3.9 or newer is recommended.

```bash
git clone <repository-url> radioml-classifier
cd radioml-classifier

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download `RML2016.10a_dict.pkl` or the optimized equivalent and either place it at:

```text
data/RML2016.10a_dict_optimized.pkl
```

or point the commands to its current location:

```bash
export RADIOML_DATASET="/path/to/RML2016.10a_dict_optimized.pkl"
```

Only load a pickle file from a source you trust.

## Run the pipeline

The commands below rebuild every generated artifact in dependency order.

```bash
python scripts/explore.py "$RADIOML_DATASET"
python scripts/build_features.py "$RADIOML_DATASET"
python src/train.py
MPLBACKEND=Agg python src/evaluate.py
MPLBACKEND=Agg python src/anomaly.py "$RADIOML_DATASET"
python src/inference.py "$RADIOML_DATASET" \
  --modulation QPSK --snr 18 --sample-index 0
```

Feature extraction writes `data/radioml_features.parquet`. Training writes the
classifier and split IDs to `models/`; evaluation and anomaly experiments write their
tables and figures to `results/`.

## Dashboard

Once the models and evaluation files exist:

```bash
streamlit run app.py
```

The dashboard lets you select a modulation, SNR, and sample, then displays the I/Q
waveform, spectrum, prediction, model confidence, and anomaly score. If the dataset
is not at the default location, set `RADIOML_DATASET` before launching Streamlit or
edit the path in the sidebar.

## Inference from Python

```python
import sys

sys.path.insert(0, "src")
from inference import predict_signal

result = predict_signal(iq_signal, snr=18)
```

`iq_signal` must have shape `(2, 128)`, with I samples in the first row and Q samples
in the second. The returned confidence is the Random Forest's largest class
probability; it should be treated as a confidence score rather than a calibrated
probability of correctness.

## Notes

- The feature representation is intentionally compact and interpretable rather than
  optimized for maximum accuracy.
- Very low-SNR examples are often indistinguishable using these summary features.
- The anomaly detector reliably catches large amplitude and noise changes, but a pure
  frequency shift is usually better handled with an explicit channel-frequency rule.
