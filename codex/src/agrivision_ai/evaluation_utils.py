"""Reusable evaluation utilities for AgriVision AI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def calculate_classification_metrics(
    true_labels: list[int],
    predicted_labels: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    """Calculate standard metrics for a classification model."""

    if not true_labels:
        raise ValueError("No labels were provided for evaluation.")

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)
    per_class_accuracy = calculate_per_class_accuracy(matrix, class_names)

    return {
        "class_order": class_names,
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "precision": float(
            precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)
        ),
        "recall": float(recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "sample_count": len(true_labels),
    }


def calculate_per_class_accuracy(matrix: np.ndarray, class_names: list[str]) -> dict[str, float]:
    """Calculate accuracy for each class from a confusion matrix."""

    per_class_accuracy: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        class_total = matrix[index].sum()
        correct_count = matrix[index, index]
        per_class_accuracy[class_name] = float(correct_count / class_total) if class_total else 0.0
    return per_class_accuracy


def save_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Write an evaluation report to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


def save_confusion_matrix_plot(matrix: np.ndarray, class_names: list[str], output_path: Path) -> None:
    """Save a visual confusion matrix.

    Rows show the true class. Columns show the predicted class. Correct
    predictions appear on the diagonal.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_size = max(6, min(18, len(class_names) * 0.35))
    plt.figure(figsize=(figure_size, figure_size))
    plt.imshow(matrix, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    display_names = [name.title() for name in class_names]
    tick_font_size = 6 if len(class_names) > 20 else 10
    plt.xticks(tick_marks, display_names, rotation=90, fontsize=tick_font_size)
    plt.yticks(tick_marks, display_names, fontsize=tick_font_size)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    max_value = matrix.max() if matrix.size else 0
    threshold = max_value / 2 if max_value else 0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            color = "white" if value > threshold else "black"
            plt.text(column_index, row_index, value, ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
