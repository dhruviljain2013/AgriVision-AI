"""Import existing plant leaf AI datasets into AgriVision AI.

Supported inputs:

- A ZIP file containing a dataset.
- An extracted dataset folder.

The importer is designed for PlantVillage, PlantDoc, and similar datasets that
store images inside class folders such as:

    Tomato___Healthy
    Tomato___Early_Blight
    Potato___Late_Blight
    Apple___Scab

It also supports train/validation/test datasets arranged by plant and disease:

    image_data/train/apple/healthy
    image_data/train/apple/apple_scab
    image_data/train/tomato/early_blight
    image_data/validation/tomato/late_blight

Output folders:

    PlantVillage-style inputs:
        data/raw/healthy/
        data/raw/diseased/

    Already-split inputs:
        data/raw/training/healthy/
        data/raw/training/diseased/
        data/raw/validation/healthy/
        data/raw/validation/diseased/
        data/raw/testing/healthy/
        data/raw/testing/diseased/

This module does not train a model and does not scrape the web.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from PIL import Image, ImageStat, UnidentifiedImageError


RAW_DATA_DIR = Path("data/raw")
HEALTHY_DIR = RAW_DATA_DIR / "healthy"
DISEASED_DIR = RAW_DATA_DIR / "diseased"
METADATA_PATH = RAW_DATA_DIR / "metadata.csv"
REPORT_PATH = Path("reports/dataset_import_report.json")
LOG_PATH = Path("reports/dataset_importer.log")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
OUTPUT_EXTENSION = ".jpg"
MIN_IMAGE_WIDTH = 64
MIN_IMAGE_HEIGHT = 64
BLANK_IMAGE_STDDEV_THRESHOLD = 2.0
SPLIT_FOLDER_NAMES = {"train", "training", "valid", "validation", "val", "test", "testing"}

METADATA_FIELDS = [
    "filename",
    "split",
    "plant_name",
    "health_status",
    "disease_name",
    "original_class",
]


@dataclass(frozen=True)
class ParsedClass:
    """Class information inferred from a dataset folder name."""

    plant_name: str
    health_status: str
    disease_name: str
    original_class: str
    split_name: str | None = None


@dataclass
class ImportReport:
    """Counters for the import operation."""

    total_images: int = 0
    healthy_images: int = 0
    diseased_images: int = 0
    duplicates_removed: int = 0
    invalid_images_removed: int = 0
    unsupported_files_removed: int = 0
    plant_names: set[str] | None = None
    disease_names: set[str] | None = None

    def __post_init__(self) -> None:
        """Initialize mutable sets safely."""

        if self.plant_names is None:
            self.plant_names = set()
        if self.disease_names is None:
            self.disease_names = set()

    def as_dict(self) -> dict:
        """Return report data in a JSON-friendly format."""

        return {
            "total_images": self.total_images,
            "healthy_images": self.healthy_images,
            "diseased_images": self.diseased_images,
            "plant_count": len(self.plant_names or set()),
            "disease_count": len(self.disease_names or set()),
            "duplicates_removed": self.duplicates_removed,
            "invalid_images_removed": self.invalid_images_removed,
            "unsupported_files_removed": self.unsupported_files_removed,
            "plants": sorted(self.plant_names or set()),
            "diseases": sorted(self.disease_names or set()),
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
        }


def setup_logging() -> None:
    """Configure file and console logging."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )


def import_dataset(source_path: Path) -> ImportReport:
    """Import a ZIP dataset or extracted dataset folder."""

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset source not found: {source_path}")

    prepare_output_folders()
    existing_hashes = build_existing_hashes()
    metadata_rows: list[dict[str, str]] = []
    report = ImportReport()

    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with TemporaryDirectory() as temporary_directory:
            extraction_dir = Path(temporary_directory) / "dataset"
            extract_zip_safely(source_path, extraction_dir)
            process_dataset_folder(extraction_dir, existing_hashes, metadata_rows, report)
    elif source_path.is_dir():
        process_dataset_folder(source_path, existing_hashes, metadata_rows, report)
    else:
        raise ValueError("Source must be a ZIP file or an extracted dataset folder.")

    write_metadata(metadata_rows)
    write_report(report)
    logging.info("Dataset import complete: %s", report.as_dict())
    return report


def prepare_output_folders() -> None:
    """Create raw dataset folders if they do not already exist."""

    HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
    DISEASED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def extract_zip_safely(zip_path: Path, output_dir: Path) -> None:
    """Extract a ZIP file while preventing path traversal."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            member_path = output_dir / member.filename
            resolved_path = member_path.resolve()
            if not str(resolved_path).startswith(str(output_dir.resolve())):
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
        archive.extractall(output_dir)


def process_dataset_folder(
    dataset_root: Path,
    existing_hashes: set[str],
    metadata_rows: list[dict[str, str]],
    report: ImportReport,
) -> None:
    """Find image files, infer classes, validate, and copy into raw folders."""

    detected_layout = detect_dataset_layout(dataset_root)
    logging.info("Detected dataset layout: %s", detected_layout)

    for image_path in iter_image_like_files(dataset_root):
        parsed_class = parse_class_from_path(image_path, dataset_root, detected_layout)
        if parsed_class is None:
            report.invalid_images_removed += 1
            logging.info("Skipped image with unknown class structure: %s", image_path)
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            report.unsupported_files_removed += 1
            continue

        is_valid, reason = validate_image(image_path)
        if not is_valid:
            report.invalid_images_removed += 1
            logging.info("Skipped invalid image %s: %s", image_path, reason)
            continue

        image_hash = hash_image_pixels(image_path)
        if image_hash in existing_hashes:
            report.duplicates_removed += 1
            continue

        output_dir = output_directory_for_class(parsed_class)
        output_filename = unique_output_filename(parsed_class, output_dir)
        output_path = output_dir / output_filename
        save_clean_image(image_path, output_path)

        existing_hashes.add(image_hash)
        metadata_rows.append(
            {
                "filename": str(output_path).replace("\\", "/"),
                "split": parsed_class.split_name or "",
                "plant_name": parsed_class.plant_name,
                "health_status": parsed_class.health_status,
                "disease_name": parsed_class.disease_name,
                "original_class": parsed_class.original_class,
            }
        )
        update_report(report, parsed_class)


def iter_image_like_files(dataset_root: Path) -> Iterable[Path]:
    """Yield files from a dataset folder, including unsupported image-like files."""

    for path in dataset_root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def detect_dataset_layout(dataset_root: Path) -> str:
    """Detect whether the dataset is PlantVillage-style or split-style.

    Layout A stores plant and disease in one class folder, for example:
    `Tomato___Early_blight`.

    Layout B stores images inside split/plant/class folders, for example:
    `train/tomato/early_blight` or `image_data/test/apple/apple_scab`.
    """

    layout_b_count = 0
    layout_a_count = 0

    for image_path in iter_image_like_files(dataset_root):
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_parts = image_path.relative_to(dataset_root).parts[:-1]
        if parse_split_layout_parts(relative_parts) is not None:
            layout_b_count += 1
        elif any(parse_class_name(folder_name) is not None for folder_name in relative_parts):
            layout_a_count += 1

        if layout_b_count >= 5:
            return "split"
        if layout_a_count >= 5 and layout_b_count == 0:
            return "class_folder"

    if layout_b_count > 0:
        return "split"
    return "class_folder"


def parse_class_from_path(image_path: Path, dataset_root: Path, detected_layout: str) -> ParsedClass | None:
    """Infer plant, health status, and disease from parent folder names."""

    relative_parts = image_path.relative_to(dataset_root).parts[:-1]

    if detected_layout == "split":
        parsed_from_split = parse_split_layout_parts(relative_parts)
        if parsed_from_split is not None:
            return parsed_from_split

    for folder_name in reversed(relative_parts):
        parsed = parse_class_name(folder_name)
        if parsed is not None:
            return parsed
    return None


def parse_split_layout_parts(relative_parts: tuple[str, ...]) -> ParsedClass | None:
    """Parse split/plant/class paths used by train/validation/test datasets."""

    normalized_parts = [part.strip() for part in relative_parts if part.strip()]
    lowered_parts = [part.lower() for part in normalized_parts]

    for index, folder_name in enumerate(lowered_parts):
        if folder_name not in SPLIT_FOLDER_NAMES:
            continue

        plant_index = index + 1
        disease_index = index + 2
        if disease_index >= len(normalized_parts):
            continue

        split_name = normalize_split_name(folder_name)
        plant_part = normalized_parts[plant_index]
        disease_part = normalized_parts[disease_index]
        plant_name = normalize_label(plant_part)
        disease_name = normalize_split_disease_name(plant_part, disease_part)
        health_status = "healthy" if disease_name == "healthy" else "diseased"
        if health_status == "healthy":
            disease_name = "none"

        return ParsedClass(
            plant_name=plant_name,
            health_status=health_status,
            disease_name=disease_name,
            original_class="/".join(normalized_parts[index : disease_index + 1]),
            split_name=split_name,
        )

    return None


def normalize_split_name(folder_name: str) -> str:
    """Convert split folder aliases to the project's managed names."""

    normalized = folder_name.lower().strip()
    if normalized in {"train", "training"}:
        return "training"
    if normalized in {"valid", "validation", "val"}:
        return "validation"
    if normalized in {"test", "testing"}:
        return "testing"
    raise ValueError(f"Unsupported split folder: {folder_name}")


def normalize_split_disease_name(plant_part: str, disease_part: str) -> str:
    """Normalize disease folder names from split-style datasets.

    Some datasets repeat the plant inside the disease folder, such as
    `apple/apple_scab`. The plant prefix is removed so the disease becomes
    simply `scab`.
    """

    plant_slug = slugify(plant_part)
    disease_slug = slugify(disease_part)

    if disease_slug in {"healthy", "health"}:
        return "healthy"

    prefix = f"{plant_slug}_"
    if disease_slug.startswith(prefix):
        disease_slug = disease_slug[len(prefix) :]

    return normalize_label(disease_slug)


def parse_class_name(class_name: str) -> ParsedClass | None:
    """Parse class names used by PlantVillage, PlantDoc, and similar datasets."""

    cleaned = class_name.strip().replace(" ", "_")
    if not cleaned:
        return None

    if "___" in cleaned:
        plant_part, disease_part = cleaned.split("___", 1)
    elif "__" in cleaned:
        plant_part, disease_part = cleaned.split("__", 1)
    elif "_" in cleaned:
        parts = cleaned.split("_")
        plant_part = parts[0]
        disease_part = "_".join(parts[1:])
    else:
        lower_name = cleaned.lower()
        if "healthy" in lower_name:
            plant_part, disease_part = "unknown_plant", "healthy"
        elif any(word in lower_name for word in ["blight", "rust", "scab", "spot", "mildew", "rot"]):
            plant_part, disease_part = "unknown_plant", cleaned
        else:
            return None

    plant_name = normalize_label(plant_part)
    disease_name = normalize_label(disease_part)
    health_status = "healthy" if disease_name.lower() == "healthy" else "diseased"
    if health_status == "healthy":
        disease_name = "none"

    return ParsedClass(
        plant_name=plant_name,
        health_status=health_status,
        disease_name=disease_name,
        original_class=class_name,
    )


def normalize_label(value: str) -> str:
    """Convert folder-name text into a readable label."""

    cleaned = re.sub(r"[_-]+", " ", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower() if cleaned else "unknown"


def validate_image(image_path: Path) -> tuple[bool, str]:
    """Reject corrupted, tiny, unsupported, or visually blank images."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            if image.format not in {"JPEG", "PNG", "BMP", "WEBP"}:
                return False, f"unsupported image format: {image.format}"

            width, height = image.size
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                return False, "image is extremely small"

            rgb_image = image.convert("RGB")
            stat = ImageStat.Stat(rgb_image)
            if max(stat.stddev) < BLANK_IMAGE_STDDEV_THRESHOLD:
                return False, "image appears blank"

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return False, f"corrupted image: {error}"

    return True, "valid"


def hash_image_pixels(image_path: Path) -> str:
    """Hash normalized image pixels to catch duplicates across file formats."""

    with Image.open(image_path) as image:
        normalized = image.convert("RGB").resize((224, 224))
        return hashlib.sha256(normalized.tobytes()).hexdigest()


def build_existing_hashes() -> set[str]:
    """Hash existing raw images so imports do not create duplicates."""

    hashes: set[str] = set()
    for path in RAW_DATA_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                if validate_image(path)[0]:
                    hashes.add(hash_image_pixels(path))
            except OSError:
                logging.warning("Could not hash existing image: %s", path)
    return hashes


def output_directory_for_class(parsed_class: ParsedClass) -> Path:
    """Choose the raw output folder while preserving existing dataset splits."""

    class_folder = "healthy" if parsed_class.health_status == "healthy" else "diseased"
    if parsed_class.split_name is not None:
        return RAW_DATA_DIR / parsed_class.split_name / class_folder
    return RAW_DATA_DIR / class_folder


def unique_output_filename(parsed_class: ParsedClass, output_dir: Path) -> str:
    """Create a stable unique filename for the imported image."""

    plant_slug = slugify(parsed_class.plant_name)
    disease_slug = slugify(parsed_class.disease_name)
    prefix = f"{plant_slug}_{parsed_class.health_status}"
    if parsed_class.health_status == "diseased":
        prefix = f"{plant_slug}_{disease_slug}"

    index = 1
    while True:
        filename = f"{prefix}_{index:05d}{OUTPUT_EXTENSION}"
        if not (output_dir / filename).exists():
            return filename
        index += 1


def slugify(value: str) -> str:
    """Make a filesystem-friendly lowercase label."""

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"


def save_clean_image(source_path: Path, output_path: Path) -> None:
    """Save a validated image as RGB JPG for consistent raw storage."""

    with Image.open(source_path) as image:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.convert("RGB").save(output_path, format="JPEG", quality=95)


def update_report(report: ImportReport, parsed_class: ParsedClass) -> None:
    """Update report counters for one accepted image."""

    report.total_images += 1
    report.plant_names.add(parsed_class.plant_name)
    if parsed_class.health_status == "healthy":
        report.healthy_images += 1
    else:
        report.diseased_images += 1
        report.disease_names.add(parsed_class.disease_name)


def write_metadata(rows: list[dict[str, str]]) -> None:
    """Write metadata for imported images."""

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: ImportReport) -> None:
    """Save the import report as JSON."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report.as_dict(), file, indent=2)


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(description="Import a plant leaf dataset into AgriVision AI.")
    parser.add_argument(
        "source",
        help="Path to a ZIP dataset or extracted dataset folder.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    setup_logging()
    args = parse_args()
    report = import_dataset(Path(args.source))
    print("Dataset import report")
    print("---------------------")
    for key, value in report.as_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
