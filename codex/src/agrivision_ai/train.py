"""Train the AgriVision AI plant health classifier.

Phase 3 trains a model, saves training graphs, and evaluates the result. It
does not include prediction scripts or any desktop GUI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

try:
    from .data_loader import load_training_datasets
    from .model_builder import build_transfer_learning_model
    from .reporting import evaluate_model, plot_training_curves, save_training_history
    from .training_config import ensure_output_directories, load_training_config
except ImportError:
    from data_loader import load_training_datasets
    from model_builder import build_transfer_learning_model
    from reporting import evaluate_model, plot_training_curves, save_training_history
    from training_config import ensure_output_directories, load_training_config


def build_callbacks(config) -> list[tf.keras.callbacks.Callback]:
    """Create callbacks that make training more controlled.

    EarlyStopping prevents wasting time when validation performance stops
    improving. ModelCheckpoint saves the best model. ReduceLROnPlateau lowers
    the learning rate when progress slows. TensorBoard records logs that can
    be viewed visually.
    """

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=config.best_model_output_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_dir),
    ]


def train(config_path: Path) -> dict:
    """Run the complete Phase 3 training pipeline."""

    config = load_training_config(config_path)
    ensure_output_directories(config)

    tf.keras.utils.set_random_seed(config.random_seed)

    model = build_transfer_learning_model(config)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.epochs,
        callbacks=build_callbacks(config),
    )

    # ModelCheckpoint saves the best validation model during training. This
    # separate file keeps the final model from the last completed epoch.
    model.save(config.last_model_output_path)

    save_training_history(history, config.history_output_path)
    plot_training_curves(history, config.accuracy_graph_path, config.loss_graph_path)

    return evaluate_model(
        model=model,
        testing_dataset=testing_dataset,
        class_names=config.classes,
        report_path=config.classification_report_path,
        confusion_matrix_path=config.confusion_matrix_path,
    )


def parse_args() -> argparse.Namespace:
    """Read command-line options for training."""

    parser = argparse.ArgumentParser(description="Train the AgriVision AI model.")
    parser.add_argument(
        "--config",
        default="config/training_config.json",
        help="Path to the training configuration JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    report = train(Path(args.config))
    print("Training complete")
    print("-----------------")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Precision: {report['precision']:.4f}")
    print(f"Recall: {report['recall']:.4f}")
    print(f"F1 Score: {report['f1_score']:.4f}")


if __name__ == "__main__":
    main()
