import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "elpv"
IMAGE_DIR = DATA_DIR / "images"
LABELS_FILE = DATA_DIR / "labels.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "02_annotation_review"

CONTACT_SHEET_DIR = OUTPUT_DIR / "contact_sheets"
FEATURE_DIR = OUTPUT_DIR / "features"

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

SHEET_COLUMNS = 4
SHEET_ROWS = 3
IMAGES_PER_SHEET = SHEET_COLUMNS * SHEET_ROWS

THUMBNAIL_SIZE = (300, 300)

# Fixed gradient threshold for exploratory edge-density measurement.
# This is not a defect threshold.
EDGE_GRADIENT_THRESHOLD = 30.0

# Histogram bins used for intensity distribution.
INTENSITY_HISTOGRAM_BINS = 32

LOG_LEVEL = logging.INFO


# ============================================================
# Logging
# ============================================================

def setup_logging():
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


logger = logging.getLogger(__name__)


# ============================================================
# General utilities
# ============================================================

def ensure_directories():
    directories = [
        OUTPUT_DIR,
        CONTACT_SHEET_DIR,
        FEATURE_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def normalize_identifier(value):
    """
    Normalize identifiers for matching CSV records with image filenames.
    """

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    value = Path(value).stem

    return value.lower()


def natural_sort_key(value):
    """
    Sort filenames naturally, for example:
    image_2.png before image_10.png.
    """

    value = str(value)

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    ]


def find_column(dataframe, candidates, required=True):
    """
    Find a column using case-insensitive candidate names.
    """

    normalized_columns = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        candidate = candidate.strip().lower()

        if candidate in normalized_columns:
            return normalized_columns[candidate]

    if required:
        raise ValueError(
            f"Cannot find a column matching {candidates}. "
            f"Available columns: {list(dataframe.columns)}"
        )

    return None


# ============================================================
# Image indexing and matching
# ============================================================

def build_image_lookup():
    """
    Build lookup dictionaries for image filenames.

    The lookup supports:
    - Exact filename
    - Filename without extension
    - Case-insensitive matching
    """

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {IMAGE_DIR}"
        )

    image_paths = [
        path
        for path in IMAGE_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in: {IMAGE_DIR}"
        )

    exact_lookup = {}
    stem_lookup = {}

    for image_path in image_paths:
        filename = image_path.name.lower()
        stem = image_path.stem.lower()

        exact_lookup[filename] = image_path

        if stem not in stem_lookup:
            stem_lookup[stem] = []

        stem_lookup[stem].append(image_path)

    logger.info("Indexed %d images.", len(image_paths))

    return image_paths, exact_lookup, stem_lookup


def resolve_image_path(
    image_id,
    image_index,
    exact_lookup,
    stem_lookup,
):
    """
    Resolve an image path using image_id first, then image_index.

    If multiple images share the same stem, the record is considered
    ambiguous and is not matched automatically.
    """

    candidates = []

    if not pd.isna(image_id):
        candidates.append(image_id)

    if not pd.isna(image_index):
        candidates.append(image_index)

    for candidate in candidates:
        candidate = str(candidate).strip()

        if not candidate:
            continue

        candidate_filename = Path(candidate).name.lower()

        # Try exact filename.
        if candidate_filename in exact_lookup:
            return exact_lookup[candidate_filename]

        # Try filename without extension.
        candidate_stem = normalize_identifier(candidate)

        if candidate_stem is None:
            continue

        matches = stem_lookup.get(candidate_stem, [])

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            logger.warning(
                "Ambiguous image identifier '%s': %s",
                candidate,
                [path.name for path in matches],
            )

    return None


# ============================================================
# Dataset loading
# ============================================================

def load_dataset():
    """
    Load ELPV metadata and match each record to its image.

    Expected CSV columns:
        image_id
        image_index
        defect_probability
        cell_type
    """

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Labels file does not exist: {LABELS_FILE}"
        )

    logger.info("Loading labels from %s", LABELS_FILE)

    dataframe = pd.read_csv(LABELS_FILE)

    logger.info(
        "Loaded metadata with %d rows and %d columns.",
        len(dataframe),
        len(dataframe.columns),
    )

    logger.info(
        "Available columns: %s",
        list(dataframe.columns),
    )

    image_id_column = find_column(
        dataframe,
        ["image_id", "filename", "image", "image_name"],
        required=False,
    )

    image_index_column = find_column(
        dataframe,
        ["image_index", "index"],
        required=False,
    )

    probability_column = find_column(
        dataframe,
        [
            "defect_probability",
            "probability",
            "prob",
            "p",
            "label",
        ],
    )

    cell_type_column = find_column(
        dataframe,
        [
            "cell_type",
            "type",
            "celltype",
            "material",
        ],
    )

    if image_id_column is None and image_index_column is None:
        raise ValueError(
            "The labels CSV must contain image_id or image_index."
        )

    if image_id_column is None:
        dataframe["_image_id"] = None
        image_id_column = "_image_id"

    if image_index_column is None:
        dataframe["_image_index"] = None
        image_index_column = "_image_index"

    dataframe = dataframe.rename(
        columns={
            image_id_column: "image_id",
            image_index_column: "image_index",
            probability_column: "defect_probability",
            cell_type_column: "cell_type",
        }
    )

    dataframe["defect_probability"] = pd.to_numeric(
        dataframe["defect_probability"],
        errors="coerce",
    )

    dataframe["cell_type"] = (
        dataframe["cell_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if dataframe["defect_probability"].isna().any():
        invalid_count = dataframe["defect_probability"].isna().sum()

        raise ValueError(
            f"{invalid_count} records have invalid defect probabilities."
        )

    image_paths, exact_lookup, stem_lookup = build_image_lookup()

    resolved_paths = []

    for _, row in dataframe.iterrows():
        image_path = resolve_image_path(
            row["image_id"],
            row["image_index"],
            exact_lookup,
            stem_lookup,
        )

        resolved_paths.append(
            str(image_path) if image_path is not None else None
        )

    dataframe["image_path"] = resolved_paths

    missing_records = dataframe[
        dataframe["image_path"].isna()
    ]

    if not missing_records.empty:
        missing_report = OUTPUT_DIR / "unmatched_images.csv"

        missing_records.to_csv(
            missing_report,
            index=False,
        )

        logger.error(
            "%d metadata records could not be matched to images.",
            len(missing_records),
        )

        logger.error(
            "Unmatched records saved to %s",
            missing_report,
        )

        raise RuntimeError(
            "Image matching is incomplete. "
            "Check unmatched_images.csv and verify the identifier format."
        )

    dataframe["image_path"] = dataframe["image_path"].map(Path)

    dataframe["filename"] = dataframe["image_path"].map(
        lambda path: path.name
    )

    duplicate_filenames = dataframe[
        dataframe["filename"].duplicated(keep=False)
    ]

    if not duplicate_filenames.empty:
        raise ValueError(
            "Multiple metadata records point to the same image. "
            "Check image_id and image_index mapping."
        )

    logger.info(
        "Successfully matched %d metadata records to images.",
        len(dataframe),
    )

    return dataframe


# ============================================================
# Image loading
# ============================================================

def load_grayscale_image(image_path):
    """
    Load an image as grayscale without modifying the original file.
    """

    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L")

        return np.asarray(image, dtype=np.uint8)


# ============================================================
# Feature extraction
# ============================================================

def calculate_gradient(image):
    """
    Calculate image gradient magnitude using central differences.
    """

    image_float = image.astype(np.float32)

    gradient_x = np.gradient(image_float, axis=1)
    gradient_y = np.gradient(image_float, axis=0)

    gradient_magnitude = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    return gradient_magnitude


def extract_image_features(image_path):
    """
    Extract exploratory image statistics.

    These features describe image appearance.
    They are not ground-truth defect labels.
    """

    image = load_grayscale_image(image_path)

    image_float = image.astype(np.float32)

    gradient = calculate_gradient(image)

    height, width = image.shape

    histogram, _ = np.histogram(
        image,
        bins=INTENSITY_HISTOGRAM_BINS,
        range=(0, 256),
        density=True,
    )

    entropy = -np.sum(
        histogram[histogram > 0]
        * np.log2(histogram[histogram > 0])
    )

    features = {
        "width": width,
        "height": height,
        "mean_intensity": float(np.mean(image_float)),
        "std_intensity": float(np.std(image_float)),
        "min_intensity": int(np.min(image)),
        "max_intensity": int(np.max(image)),
        "p05_intensity": float(np.percentile(image_float, 5)),
        "p25_intensity": float(np.percentile(image_float, 25)),
        "p50_intensity": float(np.percentile(image_float, 50)),
        "p75_intensity": float(np.percentile(image_float, 75)),
        "p95_intensity": float(np.percentile(image_float, 95)),
        "intensity_entropy": float(entropy),
        "gradient_mean": float(np.mean(gradient)),
        "gradient_std": float(np.std(gradient)),
        "gradient_p90": float(np.percentile(gradient, 90)),
        "gradient_p95": float(np.percentile(gradient, 95)),
        "gradient_p99": float(np.percentile(gradient, 99)),
        "edge_density": float(
            np.mean(gradient > EDGE_GRADIENT_THRESHOLD)
        ),
        "dark_pixel_ratio_30": float(
            np.mean(image < 30)
        ),
        "dark_pixel_ratio_50": float(
            np.mean(image < 50)
        ),
        "dark_pixel_ratio_80": float(
            np.mean(image < 80)
        ),
        "bright_pixel_ratio_200": float(
            np.mean(image > 200)
        ),
    }

    return features


def extract_all_features(dataframe):
    """
    Extract features for all images.
    """

    records = []

    total = len(dataframe)

    for index, row in dataframe.iterrows():
        image_path = row["image_path"]

        try:
            features = extract_image_features(image_path)

            record = {
                "image_id": row["image_id"],
                "image_index": row["image_index"],
                "filename": row["filename"],
                "defect_probability": row["defect_probability"],
                "cell_type": row["cell_type"],
                **features,
            }

            records.append(record)

            if len(records) % 100 == 0:
                logger.info(
                    "Extracted features for %d/%d images.",
                    len(records),
                    total,
                )

        except Exception:
            logger.exception(
                "Failed to process image: %s",
                image_path,
            )

    feature_dataframe = pd.DataFrame(records)

    if len(feature_dataframe) != total:
        logger.warning(
            "Feature extraction completed for %d/%d images.",
            len(feature_dataframe),
            total,
        )

    return feature_dataframe


# ============================================================
# Contact sheet generation
# ============================================================

def create_contact_sheet(dataframe, output_path, title):
    """
    Create a contact sheet from a group of images.

    Each image displays:
    - Filename
    - Defect probability
    - Cell type
    """

    if dataframe.empty:
        logger.warning(
            "Skipping empty contact sheet: %s",
            title,
        )
        return

    dataframe = dataframe.sort_values(
        by="filename",
        key=lambda column: column.map(natural_sort_key),
    )

    for sheet_index in range(
        0,
        len(dataframe),
        IMAGES_PER_SHEET,
    ):
        subset = dataframe.iloc[
            sheet_index:sheet_index + IMAGES_PER_SHEET
        ]

        figure = Figure(
            figsize=(16, 12),
            dpi=120,
        )

        canvas = FigureCanvasAgg(figure)

        axes = figure.subplots(
            SHEET_ROWS,
            SHEET_COLUMNS,
        )

        axes = np.asarray(axes).reshape(-1)

        for axis in axes:
            axis.axis("off")

        for axis, (_, row) in zip(axes, subset.iterrows()):
            image = load_grayscale_image(row["image_path"])

            axis.imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=255,
            )

            axis.set_title(
                f"{row['filename']}\n"
                f"P={row['defect_probability']:.3f} | "
                f"{row['cell_type']}",
                fontsize=8,
            )

            axis.axis("off")

        figure.suptitle(
            f"{title} | Sheet {sheet_index // IMAGES_PER_SHEET + 1}",
            fontsize=14,
        )

        figure.tight_layout(
            rect=[0, 0, 1, 0.96]
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sheet_number = (
            sheet_index // IMAGES_PER_SHEET + 1
        )

        sheet_path = output_path.parent / (
            f"{output_path.stem}_{sheet_number:03d}"
            f"{output_path.suffix}"
        )

        canvas.print_figure(
            sheet_path,
            bbox_inches="tight",
        )

        logger.info(
            "Saved contact sheet: %s",
            sheet_path,
        )


def generate_contact_sheets(dataframe):
    """
    Generate contact sheets for every cell type and probability group.
    """

    probability_values = sorted(
        dataframe["defect_probability"].unique()
    )

    cell_types = sorted(
        dataframe["cell_type"].unique()
    )

    for cell_type in cell_types:
        for probability in probability_values:
            subset = dataframe[
                (dataframe["cell_type"] == cell_type)
                & (
                    np.isclose(
                        dataframe["defect_probability"],
                        probability,
                    )
                )
            ]

            if subset.empty:
                continue

            probability_name = (
                f"{probability:.3f}".replace(".", "_")
            )

            output_path = (
                CONTACT_SHEET_DIR
                / f"{cell_type}_p_{probability_name}.png"
            )

            title = (
                f"Cell type: {cell_type} | "
                f"Defect probability: {probability:.3f}"
            )

            create_contact_sheet(
                subset,
                output_path,
                title,
            )


# ============================================================
# Manual review preparation
# ============================================================

def create_manual_review_file(dataframe):
    """
    Create a CSV for human annotation feasibility review.

    Review labels are intentionally left blank.
    """

    review_dataframe = dataframe[
        [
            "image_id",
            "image_index",
            "filename",
            "image_path",
            "cell_type",
            "defect_probability",
        ]
    ].copy()

    review_dataframe["visual_review_status"] = ""
    review_dataframe["suspected_defect_type"] = ""
    review_dataframe["annotation_difficulty"] = ""
    review_dataframe["review_notes"] = ""

    review_dataframe["visual_review_status"] = (
        review_dataframe["visual_review_status"].astype(str)
    )

    output_path = OUTPUT_DIR / "manual_review.csv"

    review_dataframe.to_csv(
        output_path,
        index=False,
    )

    logger.info(
        "Saved manual review template: %s",
        output_path,
    )


# ============================================================
# Statistical analysis
# ============================================================

def generate_feature_statistics(feature_dataframe):
    """
    Generate descriptive statistics grouped by cell type
    and defect probability.
    """

    numeric_columns = feature_dataframe.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    excluded_columns = {
        "image_index",
        "defect_probability",
    }

    numeric_columns = [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]

    grouped = feature_dataframe.groupby(
        [
            "cell_type",
            "defect_probability",
        ],
        observed=False,
    )

    statistics = grouped[numeric_columns].agg(
        [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
    )

    output_path = FEATURE_DIR / "feature_statistics.csv"

    statistics.to_csv(output_path)

    logger.info(
        "Saved feature statistics: %s",
        output_path,
    )


def generate_dataset_summary(dataframe, feature_dataframe):
    """
    Generate a JSON summary of dataset composition.
    """

    summary = {
        "total_records": int(len(dataframe)),
        "matched_images": int(dataframe["image_path"].notna().sum()),
        "cell_types": {},
        "probability_distribution": {},
        "cell_type_probability_distribution": {},
        "image_dimensions": {},
        "feature_extraction_success": int(len(feature_dataframe)),
    }

    for cell_type, count in dataframe["cell_type"].value_counts().items():
        summary["cell_types"][str(cell_type)] = int(count)

    for probability, count in dataframe[
        "defect_probability"
    ].value_counts().sort_index().items():
        summary["probability_distribution"][
            str(probability)
        ] = int(count)

    grouped = dataframe.groupby(
        ["cell_type", "defect_probability"],
        observed=False,
    ).size()

    for (cell_type, probability), count in grouped.items():
        key = f"{cell_type}_p_{probability:.3f}"

        summary["cell_type_probability_distribution"][
            key
        ] = int(count)

    if not feature_dataframe.empty:
        dimensions = (
            feature_dataframe.groupby(
                ["width", "height"],
                observed=False,
            )
            .size()
        )

        for (width, height), count in dimensions.items():
            key = f"{width}x{height}"

            summary["image_dimensions"][key] = int(count)

    output_path = OUTPUT_DIR / "dataset_summary.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    logger.info(
        "Saved dataset summary: %s",
        output_path,
    )


# ============================================================
# Feature visualization
# ============================================================

def plot_feature_distributions(feature_dataframe):
    """
    Plot selected feature distributions grouped by probability.
    """

    selected_features = [
        "mean_intensity",
        "std_intensity",
        "gradient_mean",
        "gradient_std",
        "edge_density",
        "dark_pixel_ratio_50",
        "intensity_entropy",
    ]

    probability_values = sorted(
        feature_dataframe["defect_probability"].unique()
    )

    for feature in selected_features:
        if feature not in feature_dataframe.columns:
            continue

        figure, axis = plt.subplots(
            figsize=(10, 6)
        )

        grouped_values = [
            feature_dataframe.loc[
                np.isclose(
                    feature_dataframe["defect_probability"],
                    probability,
                ),
                feature,
            ].dropna().values
            for probability in probability_values
        ]

        axis.boxplot(
            grouped_values,
            tick_labels=[
                f"{probability:.3f}"
                for probability in probability_values
            ],
        )

        axis.set_title(
            f"{feature} by defect probability"
        )

        axis.set_xlabel(
            "Image-level defect probability"
        )

        axis.set_ylabel(feature)

        axis.grid(
            axis="y",
            alpha=0.3,
        )

        figure.tight_layout()

        output_path = (
            FEATURE_DIR
            / f"{feature}_by_probability.png"
        )

        figure.savefig(
            output_path,
            dpi=150,
        )

        plt.close(figure)

        logger.info(
            "Saved feature plot: %s",
            output_path,
        )


def plot_cell_type_comparison(feature_dataframe):
    """
    Compare selected features between mono and poly images.
    """

    selected_features = [
        "mean_intensity",
        "std_intensity",
        "gradient_mean",
        "gradient_std",
        "edge_density",
        "intensity_entropy",
    ]

    for feature in selected_features:
        if feature not in feature_dataframe.columns:
            continue

        figure, axis = plt.subplots(
            figsize=(9, 6)
        )

        cell_types = sorted(
            feature_dataframe["cell_type"].unique()
        )

        values = [
            feature_dataframe.loc[
                feature_dataframe["cell_type"] == cell_type,
                feature,
            ].dropna().values
            for cell_type in cell_types
        ]

        axis.boxplot(
            values,
            tick_labels=cell_types,
        )

        axis.set_title(
            f"{feature} by cell type"
        )

        axis.set_xlabel("Cell type")
        axis.set_ylabel(feature)

        axis.grid(
            axis="y",
            alpha=0.3,
        )

        figure.tight_layout()

        output_path = (
            FEATURE_DIR
            / f"{feature}_by_cell_type.png"
        )

        figure.savefig(
            output_path,
            dpi=150,
        )

        plt.close(figure)

        logger.info(
            "Saved cell type comparison: %s",
            output_path,
        )


# ============================================================
# Main
# ============================================================

def main():
    setup_logging()
    ensure_directories()

    logger.info(
        "Starting ELPV annotation feasibility review."
    )

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Image directory: %s",
        IMAGE_DIR,
    )

    logger.info(
        "Labels file: %s",
        LABELS_FILE,
    )

    dataframe = load_dataset()

    logger.info(
        "Dataset contains %d images.",
        len(dataframe),
    )

    logger.info(
        "Cell type distribution:\n%s",
        dataframe["cell_type"].value_counts().to_string(),
    )

    logger.info(
        "Probability distribution:\n%s",
        dataframe["defect_probability"]
        .value_counts()
        .sort_index()
        .to_string(),
    )

    create_manual_review_file(dataframe)

    logger.info(
        "Generating contact sheets."
    )

    generate_contact_sheets(dataframe)

    logger.info(
        "Extracting image features."
    )

    feature_dataframe = extract_all_features(dataframe)

    feature_output_path = (
        FEATURE_DIR / "image_features.csv"
    )

    feature_dataframe.to_csv(
        feature_output_path,
        index=False,
    )

    logger.info(
        "Saved image features: %s",
        feature_output_path,
    )

    if not feature_dataframe.empty:
        generate_feature_statistics(
            feature_dataframe
        )

        plot_feature_distributions(
            feature_dataframe
        )

        plot_cell_type_comparison(
            feature_dataframe
        )

    generate_dataset_summary(
        dataframe,
        feature_dataframe,
    )

    logger.info(
        "Annotation feasibility review completed."
    )

    logger.info(
        "Output directory: %s",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
