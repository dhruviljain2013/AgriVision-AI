"""Batch loading utilities for AgriVision AI model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetSplitSummary:
    """Small summary used to verify that a split was detected correctly."""

    name: str
    image_count: int
    class_count: int


def load_image_dataset(
    directory: Path,
    image_size: tuple[int, int],
    batch_size: int,
    class_names: list[str],
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """Load images from nested class folders as efficient TensorFlow batches.

    The extracted dataset is already split as train/validation/test and stores
    images below plant/disease folders, for example `tomato/late blight`.
    This loader reads those files in place and uses each image's parent folder
    path as its class label. It does not copy, duplicate, or rewrite images.
    """

    if not directory.exists():
        raise FileNotFoundError(f"Dataset folder not found: {directory}")

    image_paths, labels = collect_image_records(directory, class_names)
    if not image_paths:
        raise ValueError(f"No supported image files found in dataset folder: {directory}")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda image_path, label: (decode_and_resize_image(image_path, image_size), label),
        num_parallel_calls=AUTOTUNE,
        # Training samples are shuffled, so preserving the completion order of
        # parallel image decoding only creates an input bottleneck. Validation
        # and testing keep their deterministic ordering.
        deterministic=not shuffle,
    )
    dataset = dataset.batch(batch_size, num_parallel_calls=AUTOTUNE)

    # Prefetch lets TensorFlow prepare the next batch while the current batch
    # is training. This usually makes training smoother and faster.
    return dataset.prefetch(AUTOTUNE)


def load_training_datasets(config) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Load training, validation, and testing datasets from extracted folders."""

    summaries = [
        summarize_split("training", config.training_dir, config.classes),
        summarize_split("validation", config.validation_dir, config.classes),
        summarize_split("testing", config.testing_dir, config.classes),
    ]
    print("Direct dataset loader")
    print("---------------------")
    for summary in summaries:
        print(
            f"{summary.name}: {summary.image_count} images "
            f"across {summary.class_count} classes"
        )

    train_dataset = load_image_dataset(
        directory=config.training_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        class_names=config.classes,
        shuffle=True,
        seed=config.random_seed,
    )
    validation_dataset = load_image_dataset(
        directory=config.validation_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        class_names=config.classes,
        shuffle=False,
        seed=config.random_seed,
    )
    testing_dataset = load_image_dataset(
        directory=config.testing_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        class_names=config.classes,
        shuffle=False,
        seed=config.random_seed,
    )
    return train_dataset, validation_dataset, testing_dataset


def collect_image_records(directory: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    """Return image paths and integer labels without copying any files."""

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    image_paths: list[str] = []
    labels: list[int] = []
    unknown_classes: set[str] = set()

    for image_path in sorted(directory.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        class_name = class_name_for_image(directory, image_path)
        if class_name not in class_to_index:
            unknown_classes.add(class_name)
            continue

        image_paths.append(str(image_path))
        labels.append(class_to_index[class_name])

    if unknown_classes:
        unknown_list = ", ".join(sorted(unknown_classes))
        raise ValueError(f"Found class folders missing from config classes: {unknown_list}")

    return image_paths, labels


def class_name_for_image(split_directory: Path, image_path: Path) -> str:
    """Use the image's parent path inside the split as the class name."""

    return image_path.parent.relative_to(split_directory).as_posix()


def decode_and_resize_image(image_path: tf.Tensor, image_size: tuple[int, int]) -> tf.Tensor:
    """Read one image from disk, decode it as RGB, and resize it for the model."""

    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.image.resize(image, image_size)


def summarize_split(name: str, directory: Path, class_names: list[str]) -> DatasetSplitSummary:
    """Count images and detected classes for one split."""

    image_paths, labels = collect_image_records(directory, class_names)
    return DatasetSplitSummary(
        name=name,
        image_count=len(image_paths),
        class_count=len(set(labels)),
    )
