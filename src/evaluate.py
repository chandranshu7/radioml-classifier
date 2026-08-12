"""Evaluate modulation classifiers overall, by class, and by SNR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def accuracy_by_snr(y_true: pd.Series, y_pred: np.ndarray, snr: pd.Series) -> pd.DataFrame:
    """Calculate classification accuracy separately at each SNR level."""
    results = pd.DataFrame({"snr": snr.to_numpy(), "correct": y_true.to_numpy() == y_pred})
    return results.groupby("snr", as_index=False)["correct"].mean().rename(
        columns={"correct": "accuracy"}
    )


def save_confusion_matrix(
    y_true: pd.Series, y_pred: np.ndarray, labels: list[str], title: str, path: Path
) -> None:
    """Save a row-normalized matrix so each row reads as percentages of a true class."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(matrix, display_labels=labels).plot(
        ax=ax, cmap="Blues", values_format=".2f", colorbar=False
    )
    ax.set_title(f"{title}: normalized confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "radioml_features.parquet",
    )
    parser.add_argument(
        "--models", type=Path, default=project_root / "models" / "classifiers.joblib"
    )
    parser.add_argument(
        "--split", type=Path, default=project_root / "models" / "split_ids.npz"
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "results")
    args = parser.parse_args()

    table = pd.read_parquet(args.features)
    bundle = joblib.load(args.models)
    test_ids = np.load(args.split)["test_ids"]
    test_rows = table[table["signal_id"].isin(test_ids)].copy()
    feature_names = bundle["feature_names"]
    labels = bundle["classes"]
    x_test = test_rows[feature_names]
    y_test = test_rows["modulation"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict] = {}
    snr_tables: list[pd.DataFrame] = []
    for model_key in ("logistic_regression", "random_forest"):
        model = bundle[model_key]
        predictions = model.predict(x_test)
        display_name = DISPLAY_NAMES[model_key]
        overall_accuracy = accuracy_score(y_test, predictions)
        report = classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        )
        metrics[model_key] = {"accuracy": overall_accuracy, "classification_report": report}
        pd.DataFrame(report).transpose().to_csv(
            args.output_dir / f"{model_key}_classification_report.csv"
        )
        save_confusion_matrix(
            y_test,
            predictions,
            labels,
            display_name,
            args.output_dir / f"{model_key}_confusion_matrix.png",
        )
        by_snr = accuracy_by_snr(y_test, predictions, test_rows["snr"])
        by_snr["model"] = display_name
        snr_tables.append(by_snr)
        print(f"{display_name} accuracy: {overall_accuracy:.4f}")

    snr_results = pd.concat(snr_tables, ignore_index=True)
    snr_results.to_csv(args.output_dir / "accuracy_by_snr.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, rows in snr_results.groupby("model"):
        ax.plot(rows["snr"], rows["accuracy"], marker="o", label=model_name)
    ax.set(
        title="Modulation classification accuracy by SNR",
        xlabel="SNR (dB)",
        ylabel="Accuracy",
        xticks=sorted(snr_results["snr"].unique()),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "accuracy_by_snr.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    forest = bundle["random_forest"]
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.output_dir / "random_forest_feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_rows = importance.sort_values("importance")
    ax.barh(plot_rows["feature"], plot_rows["importance"])
    ax.set(title="Random Forest feature importance", xlabel="Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "random_forest_feature_importance.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)

    with (args.output_dir / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)
    print("\nRandom Forest feature importance:")
    print(importance.to_string(index=False))
    print(f"\nSaved evaluation: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
