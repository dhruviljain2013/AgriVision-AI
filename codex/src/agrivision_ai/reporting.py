"""Training and evaluation reports for AgriVision AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from .evaluation_utils import (
        calculate_classification_metrics,
        save_confusion_matrix_plot,
        save_json_report,
    )
except ImportError:
    from evaluation_utils import (
        calculate_classification_metrics,
        save_confusion_matrix_plot,
        save_json_report,
    )


def save_training_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save epoch-by-epoch training results as JSON."""

    serializable_history = {
        metric_name: [float(value) for value in metric_values]
        for metric_name, metric_values in history.history.items()
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=2)


def plot_training_curves(history: tf.keras.callbacks.History, accuracy_path: Path, loss_path: Path) -> None:
    """Create accuracy and loss graphs for the project report."""

    metrics = history.history
    save_line_plot(
        title="Training and Validation Accuracy",
        y_label="Accuracy",
        train_values=metrics.get("accuracy", []),
        validation_values=metrics.get("val_accuracy", []),
        output_path=accuracy_path,
    )
    save_line_plot(
        title="Training and Validation Loss",
        y_label="Loss",
        train_values=metrics.get("loss", []),
        validation_values=metrics.get("val_loss", []),
        output_path=loss_path,
    )


def save_line_plot(
    title: str,
    y_label: str,
    train_values: list[float],
    validation_values: list[float],
    output_path: Path,
) -> None:
    """Save one clean line graph."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, marker="o", label="Training")
    if validation_values:
        plt.plot(epochs, validation_values, marker="o", label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    testing_dataset: tf.data.Dataset,
    class_names: list[str],
    report_path: Path,
    confusion_matrix_path: Path,
) -> dict[str, Any]:
    """Evaluate the trained model and save standard classification metrics."""

    true_labels: list[int] = []
    predicted_labels: list[int] = []

    for image_batch, label_batch in testing_dataset:
        probabilities = model.predict(image_batch, verbose=0)
        labels = label_batch.numpy().reshape(-1).astype(int)
        if probabilities.shape[-1] == 1:
            predictions = (probabilities.reshape(-1) >= 0.5).astype(int)
        else:
            predictions = np.argmax(probabilities, axis=1).astype(int)

        true_labels.extend(labels.tolist())
        predicted_labels.extend(predictions.tolist())

    if not true_labels:
        raise ValueError("Testing dataset is empty. Add images before evaluating the model.")

    report = calculate_classification_metrics(true_labels, predicted_labels, class_names)
    report["prediction_type"] = "multiclass" if len(class_names) > 2 else "binary"

    save_json_report(report, report_path)
    save_confusion_matrix_plot(
        matrix=np.array(report["confusion_matrix"]),
        class_names=class_names,
        output_path=confusion_matrix_path,
    )
    return report
