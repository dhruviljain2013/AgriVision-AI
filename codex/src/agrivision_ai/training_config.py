"""Configuration helpers for the AgriVision AI training pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """All settings needed to train and evaluate the model."""

    image_size: tuple[int, int]
    classes: list[str]
    random_seed: int
    training_dir: Path
    validation_dir: Path
    testing_dir: Path
    model_architecture: str
    best_model_output_path: Path
    last_model_output_path: Path
    history_output_path: Path
    accuracy_graph_path: Path
    loss_graph_path: Path
    classification_report_path: Path
    confusion_matrix_path: Path
    tensorboard_log_dir: Path
    batch_size: int
    epochs: int
    learning_rate: float
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    dropout_rate: float
    dense_units: int


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load training settings and reuse dataset settings from the config folder.

    The dataset config already knows the image size, class names, and random
    seed. Reusing those values prevents the training code from silently using
    different assumptions than the dataset preparation step.
    """

    with config_path.open("r", encoding="utf-8") as file:
        training_raw: dict[str, Any] = json.load(file)

    dataset_config_path = Path(training_raw["dataset_config_path"])
    with dataset_config_path.open("r", encoding="utf-8") as file:
        dataset_raw: dict[str, Any] = json.load(file)

    image_size = tuple(dataset_raw["image_size"])
    config = TrainingConfig(
        image_size=(int(image_size[0]), int(image_size[1])),
        classes=list(dataset_raw["classes"]),
        random_seed=int(dataset_raw["random_seed"]),
        training_dir=Path(training_raw["training_dir"]),
        validation_dir=Path(training_raw["validation_dir"]),
        testing_dir=Path(training_raw["testing_dir"]),
        model_architecture=str(training_raw.get("model_architecture", "MobileNetV2")),
        best_model_output_path=Path(training_raw["best_model_output_path"]),
        last_model_output_path=Path(training_raw["last_model_output_path"]),
        history_output_path=Path(training_raw["history_output_path"]),
        accuracy_graph_path=Path(training_raw["accuracy_graph_path"]),
        loss_graph_path=Path(training_raw["loss_graph_path"]),
        classification_report_path=Path(training_raw["classification_report_path"]),
        confusion_matrix_path=Path(training_raw["confusion_matrix_path"]),
        tensorboard_log_dir=Path(training_raw["tensorboard_log_dir"]),
        batch_size=int(training_raw["batch_size"]),
        epochs=int(training_raw["epochs"]),
        learning_rate=float(training_raw["learning_rate"]),
        early_stopping_patience=int(training_raw["early_stopping_patience"]),
        reduce_lr_patience=int(training_raw["reduce_lr_patience"]),
        reduce_lr_factor=float(training_raw["reduce_lr_factor"]),
        dropout_rate=float(training_raw["dropout_rate"]),
        dense_units=int(training_raw["dense_units"]),
    )
    validate_training_config(config)
    return config


def validate_training_config(config: TrainingConfig) -> None:
    """Fail early if a training setting would make the run invalid."""

    if len(config.classes) < 2:
        raise ValueError("Training requires at least two classes.")

    if len(config.classes) != len(set(config.classes)):
        raise ValueError("Class names must be unique.")

    supported_architectures = {"MobileNetV2", "EfficientNetB0", "ResNet50"}
    if config.model_architecture not in supported_architectures:
        raise ValueError(
            "model_architecture must be one of: MobileNetV2, EfficientNetB0, ResNet50."
        )

    if config.image_size[0] <= 0 or config.image_size[1] <= 0:
        raise ValueError("Image size must be positive.")

    if config.batch_size <= 0:
        raise ValueError("Batch size must be greater than zero.")

    if config.epochs <= 0:
        raise ValueError("Epoch count must be greater than zero.")

    if not 0.0 <= config.dropout_rate < 1.0:
        raise ValueError("Dropout rate must be between 0.0 and 1.0.")

    if not 0.0 < config.reduce_lr_factor < 1.0:
        raise ValueError("reduce_lr_factor must be greater than 0.0 and less than 1.0.")


def ensure_output_directories(config: TrainingConfig) -> None:
    """Create folders that will hold models, graphs, logs, and reports."""

    output_paths = [
        config.best_model_output_path,
        config.last_model_output_path,
        config.history_output_path,
        config.accuracy_graph_path,
        config.loss_graph_path,
        config.classification_report_path,
        config.confusion_matrix_path,
    ]

    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    config.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
