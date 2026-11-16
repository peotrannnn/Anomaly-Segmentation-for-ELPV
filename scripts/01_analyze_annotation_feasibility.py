from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


# Configuration

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "elpv" / "images"
LABELS_FILE = PROJECT_ROOT / "data" / "raw" / "elpv" / "labels.csv"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "01_eda" / "integrity_taxonomy"

EXPECTED_IMAGE_COUNT = 2624
EXPECTED_IMAGE_SIZE = (300, 300)
EXPECTED_IMAGE_MODE = "L"

EXPECTED_PROBABILITIES = [0.0, 1 / 3, 2 / 3, 1.0]
PROBABILITY_TOLERANCE = 0.005

SAMPLES_PER_GROUP = 12
RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

LOG_LEVEL = logging.INFO


# Logging

def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                OUTPUT_DIR / "analysis.log",
                encoding="utf-8",
            ),
        ],
    )


# Utility functions

def calculate_sha256(image_path: Path) -> str:
    sha256 = hashlib.sha256()

    with image_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def normalize_probability(value: float) -> float | None:
    for expected in EXPECTED_PROBABILITIES:
        if abs(float(value) - expected) <= PROBABILITY_TOLERANCE:
            return float(expected)

    return None


def format_probability(value: float) -> str:
    if np.isclose(value, 0):
        return "0.000"

    if np.isclose(value, 1 / 3):
        return "0.333"

    if np.isclose(value, 2 / 3):
        return "0.667"

    if np.isclose(value, 1):
        return "1.000"

    return f"{value:.3f}"


def resolve_image_path(image_id: str) -> Path | None:
    image_path = IMAGE_DIR / str(image_id)

    if image_path.is_file():
        return image_path

    image_name = Path(str(image_id)).name

    matches = list(IMAGE_DIR.rglob(image_name))

    if matches:
        return matches[0]

    return None


# Dataset loading

def load_metadata() -> pd.DataFrame:
    logging.info("Loading dataset metadata.")

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_FILE}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Image directory not found: {IMAGE_DIR}"
        )

    df = pd.read_csv(LABELS_FILE)

    required_columns = {
        "image_id",
        "image_index",
        "defect_probability",
        "cell_type",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {missing_columns}"
        )

    df["image_id"] = df["image_id"].astype(str)
    df["cell_type"] = df["cell_type"].astype(str).str.lower()

    df["defect_probability"] = pd.to_numeric(
        df["defect_probability"],
        errors="coerce",
    )

    df["normalized_probability"] = (
        df["defect_probability"].apply(
            lambda x: normalize_probability(x)
            if pd.notna(x)
            else None
        )
    )

    logging.info("Loaded %d metadata records.", len(df))

    return df


# Dataset integrity analysis

def analyze_integrity(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Starting dataset integrity analysis.")

    records = []
    hash_groups = defaultdict(list)

    for index, row in df.iterrows():
        image_id = row["image_id"]
        image_path = resolve_image_path(image_id)

        record = {
            "image_id": image_id,
            "image_index": row["image_index"],
            "defect_probability": row["defect_probability"],
            "cell_type": row["cell_type"],
            "image_exists": False,
            "image_readable": False,
            "image_mode": None,
            "image_width": None,
            "image_height": None,
            "expected_size": False,
            "expected_mode": False,
            "min_intensity": None,
            "max_intensity": None,
            "mean_intensity": None,
            "std_intensity": None,
            "sha256": None,
            "error": None,
        }

        if image_path is None:
            record["error"] = "Image file not found"
            records.append(record)
            continue

        record["image_exists"] = True

        try:
            with Image.open(image_path) as image:
                image.load()

                record["image_readable"] = True
                record["image_mode"] = image.mode
                record["image_width"] = image.width
                record["image_height"] = image.height

                record["expected_size"] = (
                    image.size == EXPECTED_IMAGE_SIZE
                )

                record["expected_mode"] = (
                    image.mode == EXPECTED_IMAGE_MODE
                )

                grayscale = np.asarray(
                    image.convert("L"),
                    dtype=np.uint8,
                )

                record["min_intensity"] = int(grayscale.min())
                record["max_intensity"] = int(grayscale.max())
                record["mean_intensity"] = float(grayscale.mean())
                record["std_intensity"] = float(grayscale.std())

            image_hash = calculate_sha256(image_path)

            record["sha256"] = image_hash

            hash_groups[image_hash].append(image_id)

        except (
            OSError,
            ValueError,
            UnidentifiedImageError,
        ) as error:
            record["error"] = str(error)

        records.append(record)

        if (index + 1) % 200 == 0:
            logging.info(
                "Integrity checked: %d / %d",
                index + 1,
                len(df),
            )

    integrity_df = pd.DataFrame(records)

    duplicate_records = []

    for image_hash, image_ids in hash_groups.items():
        if len(image_ids) > 1:
            duplicate_records.append({
                "sha256": image_hash,
                "duplicate_count": len(image_ids),
                "image_ids": ";".join(image_ids),
            })

    duplicate_df = pd.DataFrame(
        duplicate_records,
        columns=[
            "sha256",
            "duplicate_count",
            "image_ids",
        ],
    )

    integrity_df.to_csv(
        OUTPUT_DIR / "01_image_integrity.csv",
        index=False,
    )

    duplicate_df.to_csv(
        OUTPUT_DIR / "02_exact_duplicates.csv",
        index=False,
    )

    logging.info(
        "Integrity analysis completed. Exact duplicate groups: %d",
        len(duplicate_df),
    )

    return integrity_df, duplicate_df


# Label and cell type analysis

def analyze_label_distribution(
    df: pd.DataFrame,
) -> pd.DataFrame:
    logging.info("Analyzing probability and cell type distribution.")

    valid_df = df.dropna(
        subset=["normalized_probability"]
    ).copy()

    summary = (
        valid_df.groupby(
            ["normalized_probability", "cell_type"],
            observed=True,
        )
        .size()
        .reset_index(name="count")
    )

    total = len(valid_df)

    summary["percentage"] = (
        summary["count"] / total * 100
    )

    summary["probability_label"] = (
        summary["normalized_probability"].apply(
            format_probability
        )
    )

    summary.to_csv(
        OUTPUT_DIR / "03_probability_cell_distribution.csv",
        index=False,
    )

    cross_table = pd.crosstab(
        valid_df["normalized_probability"],
        valid_df["cell_type"],
    )

    cross_table.to_csv(
        OUTPUT_DIR / "04_probability_cell_crosstab.csv"
    )

    logging.info("Probability and cell type summary:")
    logging.info("\n%s", cross_table.to_string())

    return summary


def plot_probability_cell_distribution(
    summary: pd.DataFrame,
) -> None:
    probabilities = EXPECTED_PROBABILITIES
    cell_types = sorted(summary["cell_type"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(probabilities))
    width = 0.8 / max(len(cell_types), 1)

    for index, cell_type in enumerate(cell_types):
        counts = []

        for probability in probabilities:
            selected = summary[
                (summary["normalized_probability"] == probability)
                & (summary["cell_type"] == cell_type)
            ]

            count = (
                int(selected["count"].sum())
                if not selected.empty
                else 0
            )

            counts.append(count)

        offset = (
            index - (len(cell_types) - 1) / 2
        ) * width

        bars = ax.bar(
            x + offset,
            counts,
            width,
            label=cell_type,
        )

        ax.bar_label(
            bars,
            padding=2,
            fontsize=8,
        )

    ax.set_title("ELPV Defect Probability by Cell Type")
    ax.set_xlabel("Defect Probability")
    ax.set_ylabel("Number of Images")

    ax.set_xticks(x)
    ax.set_xticklabels([
        format_probability(p)
        for p in probabilities
    ])

    ax.legend(title="Cell Type")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "05_probability_by_cell_type.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# Image sampling

def create_sampling_manifest(
    df: pd.DataFrame,
) -> pd.DataFrame:
    logging.info("Creating reproducible image sampling manifest.")

    rng = random.Random(RANDOM_SEED)

    valid_df = df.dropna(
        subset=["normalized_probability"]
    ).copy()

    samples = []

    for probability in EXPECTED_PROBABILITIES:
        for cell_type in sorted(valid_df["cell_type"].unique()):

            group = valid_df[
                (valid_df["normalized_probability"] == probability)
                & (valid_df["cell_type"] == cell_type)
            ]

            if group.empty:
                logging.warning(
                    "No images for P=%s, cell=%s",
                    format_probability(probability),
                    cell_type,
                )
                continue

            sample_size = min(
                SAMPLES_PER_GROUP,
                len(group),
            )

            selected_indices = rng.sample(
                list(group.index),
                sample_size,
            )

            for selected_index in selected_indices:
                row = valid_df.loc[selected_index]

                samples.append({
                    "image_id": row["image_id"],
                    "image_index": row["image_index"],
                    "defect_probability": probability,
                    "cell_type": cell_type,
                    "sampling_group": (
                        f"P={format_probability(probability)}"
                        f"_{cell_type}"
                    ),
                })

    sampling_df = pd.DataFrame(samples)

    sampling_df.to_csv(
        OUTPUT_DIR / "06_taxonomy_sampling_manifest.csv",
        index=False,
    )

    logging.info(
        "Selected %d images for visual inspection.",
        len(sampling_df),
    )

    return sampling_df


# Contact sheet generation

def create_contact_sheet(
    group_df: pd.DataFrame,
    group_name: str,
) -> None:
    if group_df.empty:
        return

    sample_count = len(group_df)

    columns = 4
    rows = int(np.ceil(sample_count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(16, 4.8 * rows),
    )

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for index, (_, row) in enumerate(group_df.iterrows()):
        image_path = resolve_image_path(row["image_id"])

        if image_path is None:
            continue

        try:
            with Image.open(image_path) as image:
                grayscale = np.asarray(
                    image.convert("L")
                )

            ax = axes[index]

            ax.imshow(
                grayscale,
                cmap="gray",
                vmin=0,
                vmax=255,
            )

            ax.set_title(
                f"{row['image_id']}\n"
                f"P={format_probability(row['defect_probability'])}",
                fontsize=9,
            )

            ax.axis("off")

        except (OSError, ValueError):
            continue

    fig.suptitle(
        f"ELPV Visual Inspection - {group_name}\n"
        f"Total images: {sample_count}",
        fontsize=15,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    safe_name = (
        group_name.lower()
        .replace("=", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )

    output_path = (
        OUTPUT_DIR / "contact_sheets" / f"{safe_name}.png"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    logging.info("Saved contact sheet: %s", output_path)


def generate_contact_sheets(
    sampling_df: pd.DataFrame,
) -> None:
    logging.info("Generating taxonomy contact sheets.")

    for group_name, group_df in sampling_df.groupby(
        "sampling_group",
        sort=True,
    ):
        create_contact_sheet(
            group_df,
            group_name,
        )


# Intensity analysis

def analyze_intensity(
    integrity_df: pd.DataFrame,
) -> pd.DataFrame:
    valid = integrity_df[
        integrity_df["image_readable"]
    ].copy()

    columns = [
        "min_intensity",
        "max_intensity",
        "mean_intensity",
        "std_intensity",
    ]

    summary = (
        valid.groupby("cell_type")[columns]
        .agg(["mean", "std", "min", "max"])
    )

    summary.to_csv(
        OUTPUT_DIR / "07_intensity_by_cell_type.csv"
    )

    return summary


# Report generation

def generate_report(
    df: pd.DataFrame,
    integrity_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    sampling_df: pd.DataFrame,
) -> None:
    total_images = len(df)

    readable = int(
        integrity_df["image_readable"].sum()
    )

    missing = int(
        (~integrity_df["image_exists"]).sum()
    )

    unreadable = int(
        (
            integrity_df["image_exists"]
            & ~integrity_df["image_readable"]
        ).sum()
    )

    incorrect_size = int(
        (
            integrity_df["image_readable"]
            & ~integrity_df["expected_size"]
        ).sum()
    )

    incorrect_mode = int(
        (
            integrity_df["image_readable"]
            & ~integrity_df["expected_mode"]
        ).sum()
    )

    duplicate_images = (
        int(duplicate_df["duplicate_count"].sum())
        if not duplicate_df.empty
        else 0
    )

    report = {
        "dataset": "ELPV",
        "total_metadata_records": total_images,
        "expected_image_count": EXPECTED_IMAGE_COUNT,
        "image_count_matches_expected": (
            total_images == EXPECTED_IMAGE_COUNT
        ),
        "readable_images": readable,
        "missing_images": missing,
        "unreadable_images": unreadable,
        "incorrect_image_size": incorrect_size,
        "incorrect_image_mode": incorrect_mode,
        "exact_duplicate_groups": len(duplicate_df),
        "images_in_duplicate_groups": duplicate_images,
        "taxonomy_sample_count": len(sampling_df),
        "random_seed": RANDOM_SEED,
        "expected_image_size": list(EXPECTED_IMAGE_SIZE),
        "expected_image_mode": EXPECTED_IMAGE_MODE,
        "probability_values": EXPECTED_PROBABILITIES,
    }

    report_path = OUTPUT_DIR / "08_integrity_report.json"

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    markdown = [
        "# ELPV Dataset Integrity and Taxonomy Analysis",
        "",
        "## Dataset overview",
        "",
        f"- Metadata records: {total_images}",
        f"- Expected images: {EXPECTED_IMAGE_COUNT}",
        f"- Readable images: {readable}",
        "",
        "## Integrity results",
        "",
        f"- Missing images: {missing}",
        f"- Unreadable images: {unreadable}",
        f"- Incorrect image size: {incorrect_size}",
        f"- Incorrect image mode: {incorrect_mode}",
        f"- Exact duplicate groups: {len(duplicate_df)}",
        f"- Images in duplicate groups: {duplicate_images}",
        "",
        "## Taxonomy sampling",
        "",
        f"- Sampled images: {len(sampling_df)}",
        f"- Samples per group: {SAMPLES_PER_GROUP}",
        f"- Random seed: {RANDOM_SEED}",
        "",
        "## Important notes",
        "",
        "- Image-level defect probability is not a pixel-level mask.",
        "- Exact duplicate detection uses SHA-256.",
        "- Images are not modified during this analysis.",
        "- Visual taxonomy classification still requires human inspection.",
        "- Similarity-based duplicates are not checked by this script.",
        "",
    ]

    (OUTPUT_DIR / "09_analysis_report.md").write_text(
        "\n".join(markdown),
        encoding="utf-8",
    )

    logging.info("Saved integrity report.")


# Main

def main() -> None:
    setup_logging()

    logging.info("Starting ELPV integrity and taxonomy analysis.")

    df = load_metadata()

    integrity_df, duplicate_df = analyze_integrity(df)

    summary = analyze_label_distribution(df)

    plot_probability_cell_distribution(summary)

    sampling_df = create_sampling_manifest(df)

    generate_contact_sheets(sampling_df)

    analyze_intensity(integrity_df)

    generate_report(
        df,
        integrity_df,
        duplicate_df,
        sampling_df,
    )

    logging.info("Analysis completed.")
    logging.info("Total metadata records: %d", len(df))
    logging.info("Readable images: %d", integrity_df["image_readable"].sum())
    logging.info("Duplicate groups: %d", len(duplicate_df))
    logging.info("Taxonomy samples: %d", len(sampling_df))
    logging.info("Output directory: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
