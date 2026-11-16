import csv
import hashlib
import json
import logging
import platform

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from elpv_dataset.utils import load_dataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "elpv"
IMAGE_DIR = RAW_DATA_DIR / "images"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "01_eda"
    / "source_verification"
)

LOG_DIR = PROJECT_ROOT / "logs"

EXPECTED_IMAGE_COUNT = 2624
EXPECTED_IMAGE_SIZE = (300, 300)
EXPECTED_IMAGE_DTYPE = np.uint8

DATASET_NAME = "ELPV"
PACKAGE_NAME = "elpv-dataset"
PACKAGE_VERSION = version(PACKAGE_NAME)

SOURCE_URL = "https://github.com/zae-bayern/elpv-dataset"

ARCHIVE_URL = (
    "https://data.fz-juelich.de/dataset.xhtml"
    "?persistentId=doi:10.26165/JUELICH-DATA/GCBNMA"
)

DATASET_DOI = "10.26165/JUELICH-DATA/GCBNMA"

LICENSE = "CC BY-NC-SA 4.0"

PROBABILITY_LEVELS = [0.0, 1 / 3, 2 / 3, 1.0]


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

for directory in [
    IMAGE_DIR,
    REPORT_DIR,
    LOG_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

LOG_FILE = LOG_DIR / "download_verification.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def calculate_sha256(file_path):
    """Calculate SHA-256 checksum of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def normalize_image_dtype(image):
    """
    Convert image to uint8 without changing pixel intensity
    when the original values are already within 0-255.
    """

    image = np.asarray(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected grayscale image, received shape {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or infinite values.")

    minimum = float(image.min())
    maximum = float(image.max())

    if minimum < 0 or maximum > 255:
        raise ValueError(
            f"Invalid pixel range: {minimum} - {maximum}"
        )

    if np.issubdtype(image.dtype, np.floating):
        if not np.allclose(image, np.rint(image), atol=1e-6):
            raise ValueError(
                "Non-integer pixel values detected. "
                "Automatic conversion is not allowed."
            )

    return np.rint(image).astype(np.uint8)


def write_json(file_path, data):
    """Write metadata in JSON format."""

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# DATASET LOADING
# ============================================================

def load_elpv_data():
    """Load dataset using the official Python package."""

    logger.info("Loading ELPV dataset from installed package.")

    images, probabilities, cell_types = load_dataset()

    images = np.asarray(images)
    probabilities = np.asarray(probabilities)
    cell_types = np.asarray(cell_types)

    logger.info("Dataset loaded successfully.")
    logger.info("Images shape: %s", images.shape)
    logger.info("Probabilities shape: %s", probabilities.shape)
    logger.info("Cell types shape: %s", cell_types.shape)

    return images, probabilities, cell_types


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(images, probabilities, cell_types):
    """Validate dataset structure and basic properties."""

    logger.info("Starting dataset validation.")

    errors = []

    if images.ndim != 3:
        errors.append(
            f"Expected 3-dimensional image array, got {images.shape}"
        )

    if images.ndim == 3 and images.shape[1:] != EXPECTED_IMAGE_SIZE:
        errors.append(
            f"Unexpected image dimensions: {images.shape[1:]}"
        )

    if len(images) != EXPECTED_IMAGE_COUNT:
        errors.append(
            f"Expected {EXPECTED_IMAGE_COUNT} images, got {len(images)}"
        )

    if len(images) != len(probabilities):
        errors.append("Image and probability counts do not match.")

    if len(images) != len(cell_types):
        errors.append("Image and cell type counts do not match.")

    if not np.isfinite(probabilities).all():
        errors.append("Probability array contains invalid values.")

    valid_probabilities = np.isclose(
        probabilities[:, None],
        np.asarray(PROBABILITY_LEVELS)[None, :],
        atol=1e-6
    ).any(axis=1)

    if not valid_probabilities.all():
        errors.append("Unexpected defect probability values detected.")

    if errors:
        for error in errors:
            logger.error(error)

        raise ValueError(
            "Dataset validation failed. Check the log file."
        )

    logger.info("Dataset structural validation passed.")

    return True


# ============================================================
# IMAGE EXPORT AND HASH GENERATION
# ============================================================

def export_images(images, probabilities, cell_types):
    """Export original images and generate metadata."""

    logger.info("Starting image export.")

    label_records = []
    hash_records = []

    image_hashes = Counter()

    for index in range(len(images)):
        filename = f"ELPV_{index:05d}.png"
        image_path = IMAGE_DIR / filename

        image = normalize_image_dtype(images[index])

        if image.shape != EXPECTED_IMAGE_SIZE:
            raise ValueError(
                f"Invalid image shape at index {index}"
            )

        # Save original image without resizing or enhancement.
        Image.fromarray(image, mode="L").save(
            image_path,
            format="PNG"
        )

        # Calculate SHA-256 from the exported PNG file.
        file_hash = calculate_sha256(image_path)

        image_hashes[file_hash] += 1

        label_records.append({
            "image_id": filename,
            "image_index": index,
            "defect_probability": float(probabilities[index]),
            "cell_type": str(cell_types[index])
        })

        hash_records.append({
            "image_id": filename,
            "sha256": file_hash
        })

        if (index + 1) % 200 == 0:
            logger.info(
                "Exported %d / %d images.",
                index + 1,
                len(images)
            )

    # Save label metadata.
    labels_path = RAW_DATA_DIR / "labels.csv"

    with open(
        labels_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "image_index",
                "defect_probability",
                "cell_type"
            ]
        )

        writer.writeheader()
        writer.writerows(label_records)

    # Save SHA-256 checksums.
    hashes_path = RAW_DATA_DIR / "image_hashes.csv"

    with open(
        hashes_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "sha256"]
        )

        writer.writeheader()
        writer.writerows(hash_records)

    duplicate_count = sum(
        count - 1
        for count in image_hashes.values()
        if count > 1
    )

    logger.info("Image export completed.")
    logger.info("Exported images: %d", len(images))
    logger.info("Exact duplicate images: %d", duplicate_count)

    return {
        "label_records": label_records,
        "hash_records": hash_records,
        "unique_hash_count": len(image_hashes),
        "duplicate_count": duplicate_count
    }


# ============================================================
# EXPORTED DATA VERIFICATION
# ============================================================

def verify_export(export_results):
    """Verify integrity of exported images."""

    logger.info("Starting exported data verification.")

    hash_records = export_results["hash_records"]

    verified_count = 0
    verification_errors = []

    for record in hash_records:
        image_path = IMAGE_DIR / record["image_id"]

        if not image_path.exists():
            verification_errors.append(
                f"Missing file: {record['image_id']}"
            )
            continue

        actual_hash = calculate_sha256(image_path)

        if actual_hash != record["sha256"]:
            verification_errors.append(
                f"Hash mismatch: {record['image_id']}"
            )
            continue

        with Image.open(image_path) as image:
            if image.mode != "L":
                verification_errors.append(
                    f"Invalid image mode: {record['image_id']}"
                )
                continue

            if image.size != EXPECTED_IMAGE_SIZE:
                verification_errors.append(
                    f"Invalid image size: {record['image_id']}"
                )
                continue

        verified_count += 1

    if verification_errors:
        for error in verification_errors:
            logger.error(error)

        raise RuntimeError(
            "Export verification failed."
        )

    logger.info(
        "Verified %d / %d images successfully.",
        verified_count,
        len(hash_records)
    )

    return verified_count


# ============================================================
# DATASET METADATA
# ============================================================

def create_metadata(images, probabilities, cell_types, export_results):
    """Create dataset summary metadata."""

    probability_counts = {
        str(float(level)): int(
            np.isclose(probabilities, level, atol=1e-6).sum()
        )
        for level in PROBABILITY_LEVELS
    }

    cell_type_counts = {
        str(key): int(value)
        for key, value in Counter(cell_types.tolist()).items()
    }

    metadata = {
        "dataset_name": DATASET_NAME,
        "description": (
            "Electroluminescence images of functional "
            "and defective photovoltaic solar cells."
        ),
        "source": {
            "repository": SOURCE_URL,
            "archive": ARCHIVE_URL,
            "doi": DATASET_DOI,
            "package": PACKAGE_NAME,
            "package_version": PACKAGE_VERSION
        },
        "license": LICENSE,
        "retrieved_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__
        },
        "dataset_properties": {
            "total_images": int(len(images)),
            "image_height": int(images.shape[1]),
            "image_width": int(images.shape[2]),
            "image_channels": 1,
            "original_dtype": str(images.dtype),
            "export_dtype": str(EXPECTED_IMAGE_DTYPE),
            "pixel_minimum": int(images.min()),
            "pixel_maximum": int(images.max()),
            "probability_distribution": probability_counts,
            "cell_type_distribution": cell_type_counts
        },
        "integrity": {
            "exported_images": len(export_results["hash_records"]),
            "verified_images": len(export_results["hash_records"]),
            "unique_image_hashes": export_results["unique_hash_count"],
            "duplicate_images": export_results["duplicate_count"],
            "verification_method": "SHA-256"
        }
    }

    metadata_path = RAW_DATA_DIR / "dataset_metadata.json"

    write_json(metadata_path, metadata)

    return metadata


# ============================================================
# SOURCE VERIFICATION REPORT
# ============================================================

def generate_report(metadata):
    """Generate Markdown report about dataset source and verification."""

    properties = metadata["dataset_properties"]
    integrity = metadata["integrity"]

    report = f"""# ELPV Dataset Acquisition and Source Verification

## 1. Dataset Information

- Dataset: {DATASET_NAME}
- Description: Electroluminescence photovoltaic solar cell images.
- Total images: {properties['total_images']}
- Image resolution: {properties['image_width']} x {properties['image_height']}
- Image channels: {properties['image_channels']}
- Original dtype: {properties['original_dtype']}
- Export dtype: {properties['export_dtype']}

## 2. Source Information

- Official repository: {SOURCE_URL}
- Research data archive: {ARCHIVE_URL}
- Dataset DOI: {DATASET_DOI}
- Python package: {PACKAGE_NAME}
- Package version: {PACKAGE_VERSION}
- License: {LICENSE}

## 3. Label Distribution

| Defect Probability | Number of Images |
|---|---:|
"""

    for probability, count in properties[
        "probability_distribution"
    ].items():

        report += f"| {probability} | {count} |\n"

    report += """
## 4. Cell Type Distribution

| Cell Type | Number of Images |
|---|---:|
"""

    for cell_type, count in properties[
        "cell_type_distribution"
    ].items():

        report += f"| {cell_type} | {count} |\n"

    report += f"""
## 5. Integrity Verification

- Exported images: {integrity['exported_images']}
- Verified images: {integrity['verified_images']}
- Unique image hashes: {integrity['unique_image_hashes']}
- Exact duplicate images: {integrity['duplicate_images']}
- Verification method: SHA-256

All exported files were checked for file existence, image dimensions,
grayscale mode and SHA-256 consistency.

## 6. License and Attribution

The dataset images are distributed under CC BY-NC-SA 4.0.
The original dataset authors and source must be acknowledged.

Commercial use requires separate permission from the dataset owners.

## 7. Scientific Reference

Buerhop-Lutz, C. et al. (2018).
A Benchmark for Visual Identification of Defective Solar Cells
in Electroluminescence Imagery.

European Photovoltaic Solar Energy Conference and Exhibition.

DOI: 10.4229/35thEUPVSEC20182018-5CV.3.15

## 8. Acquisition Timestamp

{metadata['retrieved_at_utc']}
"""

    report_path = REPORT_DIR / "report.md"

    report_path.write_text(
        report,
        encoding="utf-8"
    )

    verification_summary = {
        "status": "PASSED",
        "dataset": DATASET_NAME,
        "source": SOURCE_URL,
        "doi": DATASET_DOI,
        "total_images": properties["total_images"],
        "verified_images": integrity["verified_images"],
        "duplicate_images": integrity["duplicate_images"],
        "timestamp": metadata["retrieved_at_utc"]
    }

    write_json(
        REPORT_DIR / "verification.json",
        verification_summary
    )

    logger.info("Verification report generated.")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("ELPV DATASET ACQUISITION")
    logger.info("=" * 60)

    try:
        images, probabilities, cell_types = load_elpv_data()

        validate_dataset(
            images,
            probabilities,
            cell_types
        )

        export_results = export_images(
            images,
            probabilities,
            cell_types
        )

        verified_count = verify_export(export_results)

        metadata = create_metadata(
            images,
            probabilities,
            cell_types,
            export_results
        )

        metadata["integrity"]["verified_images"] = verified_count

        write_json(
            RAW_DATA_DIR / "dataset_metadata.json",
            metadata
        )

        generate_report(metadata)

        logger.info("=" * 60)
        logger.info("DATASET ACQUISITION COMPLETED")
        logger.info("Total images: %d", len(images))
        logger.info("Verified images: %d", verified_count)
        logger.info(
            "Duplicate images: %d",
            export_results["duplicate_count"]
        )
        logger.info("Report: %s", REPORT_DIR / "report.md")
        logger.info("=" * 60)

    except Exception:
        logger.exception(
            "Dataset acquisition or verification failed."
        )
        raise


if __name__ == "__main__":
    main()
