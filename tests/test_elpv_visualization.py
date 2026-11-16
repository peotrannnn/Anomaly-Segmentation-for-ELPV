
import csv
import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "elpv"
IMAGE_DIR = RAW_DATA_DIR / "images"
LABELS_PATH = RAW_DATA_DIR / "labels.csv"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "01_eda"
    / "visualization"
)

EXPECTED_IMAGE_COUNT = 2624
EXPECTED_IMAGE_SIZE = (300, 300)

PROBABILITY_LEVELS = [
    0.0,
    1 / 3,
    2 / 3,
    1.0
]

PROBABILITY_NAMES = {
    0.0: "No defect probability",
    1 / 3: "Low defect probability",
    2 / 3: "Moderate defect probability",
    1.0: "High defect probability"
}

RANDOM_SEED = 42

RANDOM_GRID_SIZE = 25
SAMPLES_PER_GROUP = 12

FIGURE_DPI = 300

# Set True to display figures interactively.
SHOW_FIGURES = True


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY INITIALIZATION
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATA LOADING
# ============================================================

def load_labels():
    """Load image metadata from labels.csv."""

    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}"
        )

    records = []

    with open(
        LABELS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "image_id",
            "image_index",
            "defect_probability",
            "cell_type"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "labels.csv is missing required columns."
            )

        for row in reader:
            records.append({
                "image_id": row["image_id"],
                "image_index": int(row["image_index"]),
                "defect_probability": float(
                    row["defect_probability"]
                ),
                "cell_type": row["cell_type"]
            })

    logger.info("Loaded %d label records.", len(records))

    return records


def load_image(image_id):
    """Load original grayscale image without preprocessing."""

    image_path = IMAGE_DIR / image_id

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        if image.mode != "L":
            raise ValueError(
                f"Expected grayscale image: {image_id}"
            )

        image_array = np.asarray(image)

    if image_array.shape != EXPECTED_IMAGE_SIZE:
        raise ValueError(
            f"Unexpected image dimensions: {image_id}"
        )

    if image_array.dtype != np.uint8:
        raise ValueError(
            f"Unexpected image dtype: {image_id}"
        )

    return image_array


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_labels(records):
    """Validate metadata before visualization."""

    if len(records) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} records, "
            f"got {len(records)}"
        )

    image_ids = [
        record["image_id"]
        for record in records
    ]

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Duplicate image IDs detected.")

    probabilities = np.asarray([
        record["defect_probability"]
        for record in records
    ])

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Invalid defect probabilities detected."
        )

    valid_probabilities = np.isclose(
        probabilities[:, None],
        np.asarray(PROBABILITY_LEVELS)[None, :],
        atol=1e-6
    ).any(axis=1)

    if not valid_probabilities.all():
        raise ValueError(
            "Unexpected defect probability values detected."
        )

    for record in records:
        if record["cell_type"] not in ["mono", "poly"]:
            raise ValueError(
                f"Unexpected cell type: {record['cell_type']}"
            )

        if not (IMAGE_DIR / record["image_id"]).exists():
            raise FileNotFoundError(
                f"Missing image: {record['image_id']}"
            )

    logger.info("Dataset metadata validation passed.")


# ============================================================
# FIGURE UTILITIES
# ============================================================

def save_figure(fig, filename):
    """Save figure to the EDA output directory."""

    output_path = OUTPUT_DIR / filename

    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight"
    )

    logger.info("Saved figure: %s", output_path)


def finish_figure(fig):
    """Display or close figure according to configuration."""

    if SHOW_FIGURES:
        plt.show()

    plt.close(fig)


# ============================================================
# EDA-01: RANDOM IMAGE VISUALIZATION
# ============================================================

def visualize_random_images(records):
    """Display a reproducible random 5x5 image grid."""

    logger.info("Generating random image visualization.")

    rng = np.random.default_rng(RANDOM_SEED)

    sample_count = min(
        RANDOM_GRID_SIZE,
        len(records)
    )

    selected_indices = rng.choice(
        len(records),
        size=sample_count,
        replace=False
    )

    selected_records = [
        records[index]
        for index in selected_indices
    ]

    fig, axes = plt.subplots(
        5,
        5,
        figsize=(15, 15)
    )

    axes = axes.flatten()

    for index, record in enumerate(selected_records):

        image = load_image(record["image_id"])

        ax = axes[index]

        ax.imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(
            f"P={record['defect_probability']:.3f}\n"
            f"{record['cell_type']} | "
            f"{record['image_id']}",
            fontsize=8
        )

        ax.axis("off")

    for index in range(sample_count, len(axes)):
        axes[index].axis("off")

    fig.suptitle(
        "ELPV Dataset - Random Image Samples",
        fontsize=16
    )

    fig.tight_layout()

    save_figure(
        fig,
        "01_random_image_grid.png"
    )

    finish_figure(fig)


# ============================================================
# EDA-02: DEFECT PROBABILITY DISTRIBUTION
# ============================================================

def visualize_probability_distribution(records):
    """Analyze the distribution of defect probabilities."""

    logger.info("Generating probability distribution.")

    probabilities = np.asarray([
        record["defect_probability"]
        for record in records
    ])

    counts = [
        int(
            np.isclose(
                probabilities,
                probability,
                atol=1e-6
            ).sum()
        )
        for probability in PROBABILITY_LEVELS
    ]

    percentages = [
        count / len(records) * 100
        for count in counts
    ]

    labels = [
        "0",
        "1/3",
        "2/3",
        "1"
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bars = ax.bar(
        labels,
        counts,
        width=0.6
    )

    for bar, count, percentage in zip(
        bars,
        counts,
        percentages
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_title(
        "ELPV Defect Probability Distribution"
    )

    ax.set_xlabel("Defect Probability")
    ax.set_ylabel("Number of Images")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "02_probability_distribution.png"
    )

    finish_figure(fig)

    logger.info("Probability distribution:")

    for label, count, percentage in zip(
        labels,
        counts,
        percentages
    ):
        logger.info(
            "P=%s | Count=%d | Percentage=%.2f%%",
            label,
            count,
            percentage
        )


# ============================================================
# EDA-03: CELL TYPE DISTRIBUTION
# ============================================================

def visualize_cell_type_distribution(records):
    """Analyze mono and poly cell distributions."""

    logger.info("Generating cell type distribution.")

    cell_types = [
        record["cell_type"]
        for record in records
    ]

    unique_types = sorted(set(cell_types))

    counts = [
        cell_types.count(cell_type)
        for cell_type in unique_types
    ]

    percentages = [
        count / len(records) * 100
        for count in counts
    ]

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    bars = ax.bar(
        unique_types,
        counts,
        width=0.5
    )

    for bar, count, percentage in zip(
        bars,
        counts,
        percentages
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({percentage:.2f}%)",
            ha="center",
            va="bottom"
        )

    ax.set_title(
        "ELPV Cell Type Distribution"
    )

    ax.set_xlabel("Cell Type")
    ax.set_ylabel("Number of Images")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    fig.tight_layout()

    save_figure(
        fig,
        "03_cell_type_distribution.png"
    )

    finish_figure(fig)


# ============================================================
# EDA-04: PROBABILITY-BASED IMAGE VISUALIZATION
# ============================================================

def visualize_probability_groups(records):
    """
    Visualize representative images from each probability group.

    Each group is sampled independently using a fixed random seed.
    """

    logger.info("Generating probability-based image grids.")

    rng = np.random.default_rng(RANDOM_SEED)

    for probability in PROBABILITY_LEVELS:

        group_records = [
            record
            for record in records
            if np.isclose(
                record["defect_probability"],
                probability,
                atol=1e-6
            )
        ]

        if not group_records:
            logger.warning(
                "No images found for probability %s",
                probability
            )
            continue

        sample_count = min(
            SAMPLES_PER_GROUP,
            len(group_records)
        )

        selected_indices = rng.choice(
            len(group_records),
            size=sample_count,
            replace=False
        )

        selected_records = [
            group_records[index]
            for index in selected_indices
        ]

        fig, axes = plt.subplots(
            3,
            4,
            figsize=(14, 11)
        )

        axes = axes.flatten()

        for index, record in enumerate(selected_records):

            image = load_image(record["image_id"])

            axes[index].imshow(
                image,
                cmap="gray",
                vmin=0,
                vmax=255
            )

            axes[index].set_title(
                f"{record['image_id']}\n"
                f"Cell: {record['cell_type']}",
                fontsize=9
            )

            axes[index].axis("off")

        for index in range(sample_count, len(axes)):
            axes[index].axis("off")

        group_name = PROBABILITY_NAMES[probability]

        fig.suptitle(
            f"{group_name} (P={probability:.3f})\n"
            f"Total images: {len(group_records)}",
            fontsize=15
        )

        fig.tight_layout()

        probability_label = (
            f"{probability:.3f}".replace(".", "_")
        )

        save_figure(
            fig,
            f"04_probability_group_{probability_label}.png"
        )

        finish_figure(fig)


# ============================================================
# EDA-05: IMAGE INTENSITY STATISTICS
# ============================================================

def analyze_image_intensity(records):
    """
    Analyze original grayscale pixel intensity statistics.

    No preprocessing or modification is performed.
    """

    logger.info("Analyzing image intensity statistics.")

    image_minima = []
    image_maxima = []
    image_means = []
    image_stds = []

    for index, record in enumerate(records):

        image = load_image(record["image_id"])

        image_minima.append(
            int(image.min())
        )

        image_maxima.append(
            int(image.max())
        )

        image_means.append(
            float(image.mean())
        )

        image_stds.append(
            float(image.std())
        )

        if (index + 1) % 200 == 0:
            logger.info(
                "Analyzed %d / %d images.",
                index + 1,
                len(records)
            )

    statistics = {
        "minimum": image_minima,
        "maximum": image_maxima,
        "mean": image_means,
        "std": image_stds
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 9)
    )

    plots = [
        ("Minimum Pixel Intensity", statistics["minimum"]),
        ("Maximum Pixel Intensity", statistics["maximum"]),
        ("Mean Pixel Intensity", statistics["mean"]),
        ("Pixel Intensity Standard Deviation", statistics["std"])
    ]

    for ax, (title, values) in zip(
        axes.flatten(),
        plots
    ):
        ax.hist(
            values,
            bins=30
        )

        ax.set_title(title)
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Number of Images")
        ax.grid(
            axis="y",
            alpha=0.3
        )

    fig.suptitle(
        "ELPV Original Grayscale Image Statistics",
        fontsize=15
    )

    fig.tight_layout()

    save_figure(
        fig,
        "05_image_intensity_statistics.png"
    )

    finish_figure(fig)

    logger.info(
        "Overall pixel intensity range: %d - %d",
        min(image_minima),
        max(image_maxima)
    )

    logger.info(
        "Mean of image means: %.4f",
        np.mean(image_means)
    )

    logger.info(
        "Mean of image standard deviations: %.4f",
        np.mean(image_stds)
    )


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():

    logger.info("=" * 65)
    logger.info("ELPV DATASET EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 65)

    records = load_labels()

    validate_labels(records)

    visualize_random_images(records)

    visualize_probability_distribution(records)

    visualize_cell_type_distribution(records)

    visualize_probability_groups(records)

    analyze_image_intensity(records)

    logger.info("=" * 65)
    logger.info("ELPV VISUALIZATION COMPLETED")
    logger.info("Total images: %d", len(records))
    logger.info("Output directory: %s", OUTPUT_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
