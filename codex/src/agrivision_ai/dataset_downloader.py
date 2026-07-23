"""Automatic public dataset downloader for AgriVision AI.

This Phase 4 script downloads real plant leaf photographs into:

    data/raw/healthy/
    data/raw/diseased/

It does not train a model, run TensorFlow, or modify the training pipeline.

The downloader prioritizes Wikimedia Commons because it provides a public API,
image metadata, author information, and license information. The code is
structured so more public agricultural repositories can be added later without
changing the rest of the workflow.

Run with:

    py -3.11 src/agrivision_ai/dataset_downloader.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


COMMONS_API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "AgriVisionAI-CBSE-Project/1.0 (educational dataset downloader)"

TARGET_COUNTS = {
    "healthy": 150,
    "diseased": 150,
}

RAW_DATA_DIR = Path("data/raw")
HEALTHY_DIR = RAW_DATA_DIR / "healthy"
DISEASED_DIR = RAW_DATA_DIR / "diseased"
METADATA_PATH = RAW_DATA_DIR / "metadata.csv"
STATE_PATH = RAW_DATA_DIR / "download_state.json"
REPORT_PATH = Path("reports/dataset_download_report.json")
LOG_PATH = Path("reports/dataset_downloader.log")

SUPPORTED_MIME_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

METADATA_FIELDS = [
    "filename",
    "plant_name",
    "health_status",
    "disease_name",
    "source_url",
    "license",
    "author",
    "download_date",
]

EXCLUDED_TITLE_WORDS = {
    "diagram",
    "drawing",
    "illustration",
    "icon",
    "logo",
    "map",
    "painting",
    "plate",
    "poster",
    "scan",
    "svg",
}

LEAF_KEYWORDS = {
    "leaf",
    "leaves",
    "foliage",
    "plant",
}


@dataclass(frozen=True)
class ImageCandidate:
    """One possible image returned by a public source."""

    source_name: str
    title: str
    download_url: str
    source_url: str
    mime_type: str
    author: str
    license_name: str
    plant_name: str
    health_status: str
    disease_name: str


@dataclass
class DownloadReport:
    """Counters that explain what happened during the download run."""

    healthy_images_downloaded: int = 0
    diseased_images_downloaded: int = 0
    failed_downloads: int = 0
    duplicate_images_skipped: int = 0
    invalid_images_removed: int = 0
    unsupported_files_skipped: int = 0
    candidates_checked: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return report data in a JSON-friendly shape."""

        return {
            "healthy_images_downloaded": self.healthy_images_downloaded,
            "diseased_images_downloaded": self.diseased_images_downloaded,
            "failed_downloads": self.failed_downloads,
            "duplicate_images_skipped": self.duplicate_images_skipped,
            "invalid_images_removed": self.invalid_images_removed,
            "unsupported_files_skipped": self.unsupported_files_skipped,
            "candidates_checked": self.candidates_checked,
        }


class WikimediaCommonsProvider:
    """Search Wikimedia Commons for real plant leaf photographs."""

    healthy_queries = [
        "healthy leaf plant photograph",
        "green leaf plant photograph",
        "healthy tomato leaf photograph",
        "healthy potato leaf photograph",
        "healthy grape leaf photograph",
        "healthy apple leaf photograph",
        "healthy rice leaf photograph",
        "healthy maize leaf photograph",
        "healthy bean leaf photograph",
        "healthy cucumber leaf photograph",
    ]

    diseased_queries = [
        "plant disease leaf photograph",
        "leaf spot disease photograph",
        "powdery mildew leaf photograph",
        "downy mildew leaf photograph",
        "leaf rust disease photograph",
        "late blight leaf photograph",
        "early blight leaf photograph",
        "bacterial leaf spot photograph",
        "mosaic virus leaf photograph",
        "anthracnose leaf photograph",
        "septoria leaf spot photograph",
        "scab disease leaf photograph",
    ]

    disease_terms = {
        "anthracnose",
        "bacterial",
        "blight",
        "disease",
        "mildew",
        "mosaic",
        "rust",
        "scab",
        "septoria",
        "spot",
        "virus",
    }

    def search(self, health_status: str, limit: int) -> Iterable[ImageCandidate]:
        """Yield image candidates from Commons search results."""

        queries = self.healthy_queries if health_status == "healthy" else self.diseased_queries
        seen_titles: set[str] = set()

        for query in queries:
            try:
                for page in self._search_pages(query=query, limit=limit):
                    title = str(page.get("title", ""))
                    if title in seen_titles:
                        continue
                    seen_titles.add(title)

                    candidate = self._page_to_candidate(page, health_status)
                    if candidate is not None:
                        yield candidate
            except Exception as error:
                logging.warning("Commons search failed for query %r: %s", query, error)
                continue

    def _search_pages(self, query: str, limit: int) -> Iterable[dict]:
        """Read paginated Commons API search results."""

        continue_token: dict[str, str] = {}
        collected = 0

        while collected < limit:
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": "50",
                "prop": "imageinfo",
                "iiprop": "url|mime|extmetadata",
                "iiurlwidth": "1200",
            }
            params.update(continue_token)
            data = http_get_json(COMMONS_API_URL, params=params)
            pages = data.get("query", {}).get("pages", {})

            for page in pages.values():
                collected += 1
                yield page
                if collected >= limit:
                    break

            if "continue" not in data:
                break
            continue_token = data["continue"]

    def _page_to_candidate(self, page: dict, health_status: str) -> ImageCandidate | None:
        """Convert one Commons API page into an ImageCandidate."""

        title = str(page.get("title", ""))
        image_info = (page.get("imageinfo") or [{}])[0]
        mime_type = str(image_info.get("mime", "")).lower()

        if not is_likely_leaf_photo(title):
            return None

        if mime_type not in SUPPORTED_MIME_TYPES:
            return None

        metadata = image_info.get("extmetadata", {})
        download_url = str(image_info.get("thumburl") or image_info.get("url") or "")
        source_url = str(image_info.get("descriptionurl") or "")

        if not download_url or not source_url:
            return None

        disease_name = "none" if health_status == "healthy" else infer_disease_name(title, self.disease_terms)
        return ImageCandidate(
            source_name="Wikimedia Commons",
            title=title,
            download_url=download_url,
            source_url=source_url,
            mime_type=mime_type,
            author=clean_metadata_value(metadata.get("Artist", {}).get("value", "Unknown")),
            license_name=clean_metadata_value(metadata.get("LicenseShortName", {}).get("value", "Unknown")),
            plant_name=infer_plant_name(title),
            health_status=health_status,
            disease_name=disease_name,
        )


def setup_logging() -> None:
    """Send logs to both the screen and a report log file."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )


def http_get_json(url: str, params: dict[str, str], retries: int = 3) -> dict:
    """GET JSON with retries so temporary network failures do not stop the run."""

    request_url = f"{url}?{urlencode(params)}"
    raw = http_get_bytes(request_url, retries=retries)
    return json.loads(raw.decode("utf-8"))


def http_get_bytes(url: str, retries: int = 3, timeout: int = 30) -> bytes:
    """Download bytes from a URL with retry and backoff."""

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            wait_seconds = 2 ** (attempt - 1)
            logging.warning("Download attempt %s failed for %s: %s", attempt, url, error)
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed after {retries} attempts: {url}") from last_error


def clean_metadata_value(value: str) -> str:
    """Remove simple HTML fragments commonly found in Commons metadata."""

    text = str(value)
    replacements = {
        "<bdi>": "",
        "</bdi>": "",
        "<span>": "",
        "</span>": "",
        "&amp;": "&",
        "&quot;": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split()) or "Unknown"


def is_likely_leaf_photo(title: str) -> bool:
    """Apply conservative text filters before downloading.

    A script cannot guarantee human-level visual clarity without a separate
    vision model, so this downloader combines reliable source metadata with
    practical filters: leaf-related words, supported photo formats, and later
    image validation for size and aspect ratio.
    """

    normalized = title.lower()
    if any(word in normalized for word in EXCLUDED_TITLE_WORDS):
        return False
    return any(word in normalized for word in LEAF_KEYWORDS)


def infer_plant_name(title: str) -> str:
    """Create a readable plant name guess from the source title."""

    cleaned = title.replace("File:", "")
    cleaned = cleaned.rsplit(".", 1)[0]
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    words = [word for word in cleaned.split() if word.lower() not in {"leaf", "leaves", "plant", "photo", "photograph"}]
    return " ".join(words[:5]) if words else "unknown plant"


def infer_disease_name(title: str, disease_terms: set[str]) -> str:
    """Infer a simple disease label from the source title."""

    normalized = title.lower().replace("_", " ").replace("-", " ")
    found_terms = [term for term in sorted(disease_terms) if term in normalized]
    return " ".join(found_terms) if found_terms else "plant disease"


def prepare_folders() -> None:
    """Create folders used by the downloader."""

    HEALTHY_DIR.mkdir(parents=True, exist_ok=True)
    DISEASED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_existing_metadata() -> list[dict[str, str]]:
    """Read metadata.csv if a previous run already created it."""

    if not METADATA_PATH.exists():
        return []

    with METADATA_PATH.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_metadata(rows: list[dict[str, str]]) -> None:
    """Write metadata atomically so interruption is less likely to corrupt it."""

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = METADATA_PATH.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(METADATA_PATH)


def load_state() -> dict:
    """Load resumable downloader state."""

    if not STATE_PATH.exists():
        return {
            "downloaded_urls": [],
            "image_hashes": [],
        }

    with STATE_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_state(state: dict) -> None:
    """Save resumable downloader state after each accepted image."""

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = STATE_PATH.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2)
    temporary_path.replace(STATE_PATH)


def existing_status_counts(metadata_rows: list[dict[str, str]]) -> dict[str, int]:
    """Count already downloaded healthy and diseased images."""

    counts = {"healthy": 0, "diseased": 0}
    for row in metadata_rows:
        status = row.get("health_status", "").lower()
        filename = row.get("filename", "")
        if status in counts and filename and Path(filename).exists():
            counts[status] += 1
    return counts


def build_existing_hashes() -> set[str]:
    """Hash existing accepted images so duplicate files can be skipped."""

    hashes: set[str] = set()
    for folder in (HEALTHY_DIR, DISEASED_DIR):
        for image_path in folder.glob("*"):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    hashes.add(hash_file(image_path))
                except OSError:
                    logging.warning("Could not hash existing image: %s", image_path)
    return hashes


def hash_file(path: Path) -> str:
    """Create a SHA-256 hash for duplicate detection."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_filename(health_status: str, extension: str, folder: Path) -> str:
    """Generate names such as healthy_public_0001.jpg."""

    existing_numbers = []
    for path in folder.glob(f"{health_status}_public_*{extension}"):
        number_text = path.stem.rsplit("_", 1)[-1]
        if number_text.isdigit():
            existing_numbers.append(int(number_text))

    next_number = max(existing_numbers, default=0) + 1
    while True:
        filename = f"{health_status}_public_{next_number:04d}{extension}"
        if not (folder / filename).exists():
            return filename
        next_number += 1


def validate_image(path: Path) -> tuple[bool, str]:
    """Validate that a downloaded file is a usable, leaf-friendly image."""

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size
            if width < 500 or height < 500:
                return False, "image is smaller than 500x500"

            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 3.0:
                return False, "image aspect ratio is too extreme"

            if image.format not in {"JPEG", "PNG"}:
                return False, f"unsupported image format: {image.format}"

    except (UnidentifiedImageError, OSError, ValueError) as error:
        return False, f"image validation failed: {error}"

    return True, "valid"


def save_candidate_image(candidate: ImageCandidate, output_path: Path) -> None:
    """Download and save one candidate image."""

    image_bytes = http_get_bytes(candidate.download_url)
    output_path.write_bytes(image_bytes)


def accepted_metadata_row(candidate: ImageCandidate, filename: Path) -> dict[str, str]:
    """Create the metadata.csv row for one accepted image."""

    return {
        "filename": str(filename).replace("\\", "/"),
        "plant_name": candidate.plant_name,
        "health_status": candidate.health_status,
        "disease_name": candidate.disease_name,
        "source_url": candidate.source_url,
        "license": candidate.license_name,
        "author": candidate.author,
        "download_date": datetime.now(timezone.utc).date().isoformat(),
    }


def download_dataset() -> DownloadReport:
    """Download enough healthy and diseased images to satisfy TARGET_COUNTS."""

    prepare_folders()
    provider = WikimediaCommonsProvider()
    metadata_rows = load_existing_metadata()
    state = load_state()
    downloaded_urls = set(state.get("downloaded_urls", []))
    image_hashes = set(state.get("image_hashes", [])) | build_existing_hashes()
    status_counts = existing_status_counts(metadata_rows)
    report = DownloadReport()

    logging.info("Starting dataset download. Current counts: %s", status_counts)

    for health_status, target_count in TARGET_COUNTS.items():
        output_dir = HEALTHY_DIR if health_status == "healthy" else DISEASED_DIR

        if status_counts[health_status] >= target_count:
            logging.info("%s already has %s images. Skipping.", health_status, status_counts[health_status])
            continue

        for candidate in provider.search(health_status=health_status, limit=800):
            if status_counts[health_status] >= target_count:
                break

            temporary_path: Path | None = None
            report.candidates_checked += 1
            if candidate.download_url in downloaded_urls or candidate.source_url in downloaded_urls:
                report.duplicate_images_skipped += 1
                continue

            extension = SUPPORTED_MIME_TYPES.get(candidate.mime_type)
            if extension is None:
                report.unsupported_files_skipped += 1
                continue

            filename = unique_filename(health_status, extension, output_dir)
            final_path = output_dir / filename

            try:
                with NamedTemporaryFile(delete=False, suffix=extension) as temporary_file:
                    temporary_path = Path(temporary_file.name)

                save_candidate_image(candidate, temporary_path)
                is_valid, reason = validate_image(temporary_path)
                if not is_valid:
                    temporary_path.unlink(missing_ok=True)
                    report.invalid_images_removed += 1
                    logging.info("Rejected image from %s: %s", candidate.source_url, reason)
                    continue

                file_hash = hash_file(temporary_path)
                if file_hash in image_hashes:
                    temporary_path.unlink(missing_ok=True)
                    report.duplicate_images_skipped += 1
                    continue

                shutil.move(str(temporary_path), final_path)
                image_hashes.add(file_hash)
                downloaded_urls.add(candidate.download_url)
                downloaded_urls.add(candidate.source_url)
                metadata_rows.append(accepted_metadata_row(candidate, final_path))
                status_counts[health_status] += 1

                state["downloaded_urls"] = sorted(downloaded_urls)
                state["image_hashes"] = sorted(image_hashes)
                save_state(state)
                write_metadata(metadata_rows)

                if health_status == "healthy":
                    report.healthy_images_downloaded += 1
                else:
                    report.diseased_images_downloaded += 1

                logging.info(
                    "Accepted %s image %s/%s: %s",
                    health_status,
                    status_counts[health_status],
                    target_count,
                    final_path,
                )

            except Exception as error:
                report.failed_downloads += 1
                logging.warning("Failed candidate %s: %s", candidate.source_url, error)
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

    write_download_report(report, status_counts)
    logging.info("Download complete. Final counts: %s", status_counts)
    return report


def write_download_report(report: DownloadReport, final_counts: dict[str, int]) -> None:
    """Save the final download report as JSON."""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    payload["final_healthy_count"] = final_counts.get("healthy", 0)
    payload["final_diseased_count"] = final_counts.get("diseased", 0)
    payload["metadata_path"] = str(METADATA_PATH).replace("\\", "/")
    payload["state_path"] = str(STATE_PATH).replace("\\", "/")
    payload["report_generated_at"] = datetime.now(timezone.utc).isoformat()

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    """Command-line entry point."""

    setup_logging()
    report = download_dataset()
    print("Dataset download report")
    print("-----------------------")
    for key, value in report.as_dict().items():
        print(f"{key}: {value}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
