"""Dataset management for AgriVision AI.

This module prepares the leaf image dataset before any model training happens.
It validates images, standardizes names, resizes images, converts them to RGB,
creates train/validation/test splits, and writes a statistics report.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from PIL import Image, UnidentifiedImageError


SPLIT_FOLDER_ALIASES = {
    "training": ("training", "train"),
    "validation": ("validation", "valid", "val"),
    "testing": ("testing", "test"),
}


@dataclass(frozen=True)
class DatasetConfig:
    """Settings that control dataset preparation."""

    image_size: tuple[int, int]
    split_ratio: dict[str, float]
    random_seed: int
    input_dir: Path
    output_dir: Path
    classes: list[str]
    supported_formats: set[str]


@dataclass(frozen=True)
class PreparedImage:
    """Information about one valid image after it has been standardized."""

    class_name: str
    original_path: str
    processed_path: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def load_config(config_path: Path) -> DatasetConfig:
    """Load dataset settings from a JSON file.

    A configuration file lets you change image size and split ratios without
    editing Python code. That is useful when experimenting with machine
    learning projects.
    """

    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = json.load(file)

    image_size = tuple(raw_config["image_size"])
    if len(image_size) != 2:
        raise ValueError("image_size must contain exactly two numbers: width and height.")

    config = DatasetConfig(
        image_size=(int(image_size[0]), int(image_size[1])),
        split_ratio={
            "training": float(raw_config["split_ratio"]["training"]),
            "validation": float(raw_config["split_ratio"]["validation"]),
            "testing": float(raw_config["split_ratio"]["testing"]),
        },
        random_seed=int(raw_config["random_seed"]),
        input_dir=Path(raw_config["input_dir"]),
        output_dir=Path(raw_config["output_dir"]),
        classes=list(raw_config["classes"]),
        supported_formats={item.lower() for item in raw_config["supported_formats"]},
    )
    validate_config(config)
    return config


def validate_config(config: DatasetConfig) -> None:
    """Check configuration values before touching the dataset."""

    if config.image_size[0] <= 0 or config.image_size[1] <= 0:
        raise ValueError("image_size values must be greater than zero.")

    if set(config.split_ratio) != {"training", "validation", "testing"}:
        raise ValueError("split_ratio must contain training, validation, and testing.")

    ratio_total = sum(config.split_ratio.values())
    if abs(ratio_total - 1.0) > 0.0001:
        raise ValueError("training + validation + testing ratios must add up to 1.0.")

    if not config.classes:
        raise ValueError("At least one class name is required.")


def prepare_output_folders(config: DatasetConfig) -> None:
    """Create fresh managed output folders for the prepared dataset."""

    managed_folders = [
        config.output_dir / "standardized",
        config.output_dir / "training",
        config.output_dir / "validation",
        config.output_dir / "testing",
    ]

    for folder in managed_folders:
        if folder.exists():
            shutil.rmtree(folder)

    for class_name in config.classes:
        standardized_class_dir = config.output_dir / "standardized" / class_name
        standardized_class_dir.mkdir(parents=True, exist_ok=True)
        write_gitkeep(standardized_class_dir)
        for split_name in config.split_ratio:
            split_class_dir = config.output_dir / split_name / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
            write_gitkeep(split_class_dir)

    for folder in managed_folders:
        write_gitkeep(folder)


def write_gitkeep(folder: Path) -> None:
    """Keep generated folders visible in Git even when they contain no images."""

    (folder / ".gitkeep").touch()


def prepare_dataset(config: DatasetConfig) -> dict[str, Any]:
    """Run the complete Phase 2 dataset management process."""

    existing_split_dirs = find_existing_split_dirs(config.input_dir)
    if existing_split_dirs is not None:
        return prepare_existing_split_dataset(config, existing_split_dirs)

    prepare_output_folders(config)

    prepared_images: list[PreparedImage] = []
    ignored_unsupported: list[str] = []
    corrupted_images: list[str] = []

    for class_name in config.classes:
        class_input_dir = config.input_dir / class_name
        class_output_dir = config.output_dir / "standardized" / class_name

        if not class_input_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_input_dir}")

        valid_count = 0
        for image_path in sorted(class_input_dir.iterdir()):
            if not image_path.is_file() or image_path.name == ".gitkeep":
                continue

            if image_path.suffix.lower() not in config.supported_formats:
                ignored_unsupported.append(str(image_path))
                continue

            prepared_image = validate_and_standardize_image(
                image_path=image_path,
                class_name=class_name,
                output_dir=class_output_dir,
                image_number=valid_count + 1,
                image_size=config.image_size,
            )

            if prepared_image is None:
                corrupted_images.append(str(image_path))
                continue

            prepared_images.append(prepared_image)
            valid_count += 1

    split_summary = split_dataset(prepared_images, config)
    statistics = build_statistics(
        prepared_images=prepared_images,
        corrupted_images=corrupted_images,
        ignored_unsupported=ignored_unsupported,
        split_summary=split_summary,
        config=config,
    )
    write_statistics(statistics, config.output_dir / "dataset_statistics.json")
    return statistics


def find_existing_split_dirs(input_dir: Path) -> dict[str, Path] | None:
    """Return existing train/validation/test folders if the dataset is pre-split."""

    split_dirs: dict[str, Path] = {}
    for managed_name, aliases in SPLIT_FOLDER_ALIASES.items():
        for alias in aliases:
            candidate = input_dir / alias
            if candidate.is_dir():
                split_dirs[managed_name] = candidate
                break

    if not split_dirs:
        return None

    missing = [split_name for split_name in SPLIT_FOLDER_ALIASES if split_name not in split_dirs]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(
            f"Found a partially split dataset in {input_dir}, but these split folders are missing: {missing_text}"
        )

    return split_dirs


def prepare_existing_split_dataset(config: DatasetConfig, split_dirs: dict[str, Path]) -> dict[str, Any]:
    """Validate and resize a dataset that already has train/validation/test splits."""

    prepare_output_folders(config)

    prepared_images: list[PreparedImage] = []
    ignored_unsupported: list[str] = []
    corrupted_images: list[str] = []
    split_summary: dict[str, dict[str, int]] = {
        "training": {class_name: 0 for class_name in config.classes},
        "validation": {class_name: 0 for class_name in config.classes},
        "testing": {class_name: 0 for class_name in config.classes},
    }

    for split_name, split_input_dir in split_dirs.items():
        class_image_numbers = {class_name: 0 for class_name in config.classes}

        for image_path in sorted(split_input_dir.rglob("*")):
            if not image_path.is_file() or image_path.name == ".gitkeep":
                continue

            if image_path.suffix.lower() not in config.supported_formats:
                ignored_unsupported.append(str(image_path))
                continue

            class_name = infer_binary_class_from_split_path(image_path, split_input_dir, config.classes)
            if class_name is None:
                ignored_unsupported.append(f"unknown class: {image_path}")
                continue

            class_image_numbers[class_name] += 1
            prepared_image = validate_and_standardize_image(
                image_path=image_path,
                class_name=class_name,
                output_dir=config.output_dir / split_name / class_name,
                image_number=class_image_numbers[class_name],
                image_size=config.image_size,
            )

            if prepared_image is None:
                corrupted_images.append(str(image_path))
                class_image_numbers[class_name] -= 1
                continue

            prepared_images.append(prepared_image)
            split_summary[split_name][class_name] += 1

    statistics = build_statistics(
        prepared_images=prepared_images,
        corrupted_images=corrupted_images,
        ignored_unsupported=ignored_unsupported,
        split_summary=split_summary,
        config=config,
    )
    statistics["split_strategy"] = "existing_split_preserved"
    write_statistics(statistics, config.output_dir / "dataset_statistics.json")
    return statistics


def infer_binary_class_from_split_path(
    image_path: Path,
    split_input_dir: Path,
    class_names: list[str],
) -> str | None:
    """Infer healthy/diseased from a pre-split image path."""

    relative_parts = [part.lower().strip().replace("-", "_") for part in image_path.relative_to(split_input_dir).parts[:-1]]
    if "healthy" in relative_parts or "health" in relative_parts:
        return "healthy" if "healthy" in class_names else None

    if relative_parts:
        return "diseased" if "diseased" in class_names else None

    return None


def validate_and_standardize_image(
    image_path: Path,
    class_name: str,
    output_dir: Path,
    image_number: int,
    image_size: tuple[int, int],
) -> PreparedImage | None:
    """Validate one image and save a clean 224x224 RGB JPEG copy.

    Validation matters because a broken image can stop training later. RGB
    conversion matters because machine learning models expect every image to
    have the same number of color channels.
    """

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            original_size = image.size
            standardized = image.convert("RGB").resize(image_size, Image.Resampling.LANCZOS)
            output_path = output_dir / f"{class_name}_{image_number:04d}.jpg"
            standardized.save(output_path, format="JPEG", quality=95)

    except (UnidentifiedImageError, OSError, ValueError):
        return None

    return PreparedImage(
        class_name=class_name,
        original_path=str(image_path),
        processed_path=str(output_path),
        original_size=original_size,
        processed_size=image_size,
    )


def split_dataset(prepared_images: list[PreparedImage], config: DatasetConfig) -> dict[str, Any]:
    """Split standardized images into training, validation, and testing folders.

    The split is done separately for each class so both Healthy and Diseased
    images are represented in every dataset section when enough images exist.
    """

    random_generator = random.Random(config.random_seed)
    split_summary: dict[str, dict[str, int]] = {
        "training": {},
        "validation": {},
        "testing": {},
    }

    for class_name in config.classes:
        class_images = [image for image in prepared_images if image.class_name == class_name]
        random_generator.shuffle(class_images)

        total = len(class_images)
        training_count = int(total * config.split_ratio["training"])
        validation_count = int(total * config.split_ratio["validation"])

        split_groups = {
            "training": class_images[:training_count],
            "validation": class_images[training_count : training_count + validation_count],
            "testing": class_images[training_count + validation_count :],
        }

        for split_name, images in split_groups.items():
            split_summary[split_name][class_name] = len(images)
            split_output_dir = config.output_dir / split_name / class_name
            for image in images:
                shutil.copy2(image.processed_path, split_output_dir / Path(image.processed_path).name)

    return split_summary


def build_statistics(
    prepared_images: list[PreparedImage],
    corrupted_images: list[str],
    ignored_unsupported: list[str],
    split_summary: dict[str, Any],
    config: DatasetConfig,
) -> dict[str, Any]:
    """Create a simple report that explains what is inside the dataset."""

    class_counts = Counter(image.class_name for image in prepared_images)
    original_widths = [image.original_size[0] for image in prepared_images]
    original_heights = [image.original_size[1] for image in prepared_images]
    size_counts = Counter(f"{width}x{height}" for width, height in (image.original_size for image in prepared_images))

    average_resolution = {
        "width": round(mean(original_widths), 2) if original_widths else 0,
        "height": round(mean(original_heights), 2) if original_heights else 0,
    }

    return {
        "total_images": len(prepared_images),
        "healthy_images": class_counts.get("healthy", 0),
        "diseased_images": class_counts.get("diseased", 0),
        "class_counts": dict(class_counts),
        "image_sizes": dict(sorted(size_counts.items())),
        "average_original_resolution": average_resolution,
        "processed_image_size": {
            "width": config.image_size[0],
            "height": config.image_size[1],
        },
        "split_summary": split_summary,
        "corrupted_images": corrupted_images,
        "ignored_unsupported_files": ignored_unsupported,
    }


def write_statistics(statistics: dict[str, Any], output_path: Path) -> None:
    """Save dataset statistics so they can be used in reports later."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(statistics, file, indent=2)


def print_statistics(statistics: dict[str, Any]) -> None:
    """Print the most important dataset statistics for quick checking."""

    print("Dataset statistics")
    print("------------------")
    print(f"Total images: {statistics['total_images']}")
    print(f"Healthy images: {statistics['healthy_images']}")
    print(f"Diseased images: {statistics['diseased_images']}")
    print(f"Average original resolution: {statistics['average_original_resolution']}")
    print(f"Processed image size: {statistics['processed_image_size']}")
    print(f"Corrupted images: {len(statistics['corrupted_images'])}")
    print(f"Unsupported files ignored: {len(statistics['ignored_unsupported_files'])}")
    print("Split summary:")
    for split_name, class_counts in statistics["split_summary"].items():
        print(f"  {split_name}: {class_counts}")


def parse_args() -> argparse.Namespace:
    """Read command-line options for the dataset manager."""

    parser = argparse.ArgumentParser(description="Prepare the AgriVision AI dataset.")
    parser.add_argument(
        "--config",
        default="config/dataset_config.json",
        help="Path to the dataset configuration JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    config = load_config(Path(args.config))
    statistics = prepare_dataset(config)
    print_statistics(statistics)


if __name__ == "__main__":
    main()
