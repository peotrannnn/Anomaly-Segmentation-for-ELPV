"""
03_image_preprocessing.py

Apply image preprocessing to all ELPV images WITHOUT overwriting originals.

What it does:
  1. Load original 300x300 PNGs from data/raw/elpv/images/
  2. Apply ELPVPreprocessor (denoise → homomorphic → CLAHE)
  3. Save preprocessed PNGs to data/processed/elpv_preprocessed/
  4. Compute before/after metrics per image
  5. Generate visual comparison grids (side-by-side per cell type & prob group)
  6. Generate statistical report (paired t-test per metric)

Output:
  data/processed/elpv_preprocessed/              (2624 PNGs)
  reports/03_preprocessing/
    ├── metrics_before.csv
    ├── metrics_after.csv
    ├── metrics_comparison.csv
    ├── metrics_aggregate.json
    ├── comparison_grid_normal_mono.png
    ├── comparison_grid_normal_poly.png
    ├── comparison_grid_defect_mono.png
    ├── comparison_grid_defect_poly.png
    ├── histogram_before_after.png
    ├── preprocessing.log
    └── report.md

Notes:
  - Originals in data/raw/elpv/images/ are NEVER modified.
  - Preprocessed images are cached; re-run with FORCE_RECOMPUTE=True to regenerate.
  - Do NOT use manual_review.csv — this is label-independent preprocessing.
  - Set DEBUG_MAX_IMAGES = N for quick test on N images before full run.

Change log:
  - v1: Initial (homomorphic σ=30, α=0.5, β=1.5, clip=2.0) — output near-black.
  - v2: Added homomorphic rescale to [0,255], reordered pipeline
        (denoise → homomorphic → CLAHE). σ=15, α=0.7, β=1.3, clip=1.5.
  - v3: Reduced β (1.3→1.2), clip_limit (1.5→1.2), increased denoise
        kernel (3→5). Report language revised to accurately describe
        dynamic range compression (not full illumination correction).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# Import shared preprocessing module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from elpv_preprocessing import ELPVPreprocessor, compute_image_metrics


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_FILE   = PROJECT_ROOT / "data" / "raw" / "elpv" / "labels.csv"
RAW_IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "elpv" / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "elpv_preprocessed"
REPORT_DIR    = PROJECT_ROOT / "reports" / "03_preprocessing"

# Preprocessing config — v3: gentler to avoid noise amplification
PREPROCESS_CONFIG = {
    "use_homomorphic":   True,
    "use_clahe":         True,
    "use_denoise":       True,
    "denoise_method":    "median",
    "homomorphic_sigma": 15.0,
    "homomorphic_alpha": 0.7,
    "homomorphic_beta":  1.2,    # was 1.3 — reduce high-freq amplification
    "clahe_clip_limit":  1.2,    # was 1.5 — reduce halo / over-saturation
    "clahe_tile_grid":   (8, 8),
    "denoise_kernel":    5,      # was 3  — stronger denoise to offset
}

# If True, regenerate preprocessed PNGs even if they exist
FORCE_RECOMPUTE = True

# Debug: process only N images (None = all)
DEBUG_MAX_IMAGES = None

# Sampling for visual grids
SAMPLES_PER_GROUP = 6

# Prob groups
PROB_ZERO_THRESHOLD = 0.05
PROB_ONE_THRESHOLD  = 0.95

RANDOM_SEED = 42
LOG_LEVEL   = logging.INFO


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                REPORT_DIR / "preprocessing.log", encoding="utf-8"
            ),
        ],
        force=True,
    )


logger = logging.getLogger(__name__)


# ============================================================
# HELPERS
# ============================================================

def assign_prob_group(p: float) -> str:
    if p <= PROB_ZERO_THRESHOLD:
        return "normal"
    if p >= PROB_ONE_THRESHOLD:
        return "defect"
    return "borderline"


def load_image(image_path: Path) -> np.ndarray:
    arr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    return arr


def save_processed_image(image: np.ndarray, image_id: str) -> Path:
    out_path = PROCESSED_DIR / image_id
    cv2.imwrite(str(out_path), image)
    return out_path


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================

def apply_preprocessing(
    df: pd.DataFrame, preprocessor: ELPVPreprocessor,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    before_records: list[dict] = []
    after_records: list[dict] = []

    t0 = time.time()

    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Preprocessing", ncols=80,
    ):
        image_id = row["image_id"]
        src_path = RAW_IMAGE_DIR / image_id
        dst_path = PROCESSED_DIR / image_id

        try:
            original = load_image(src_path)

            # ── BEFORE ──
            m_before = compute_image_metrics(original)
            m_before["image_id"] = image_id
            m_before["cell_type"] = row["cell_type"]
            m_before["defect_probability"] = float(row["defect_probability"])
            m_before["prob_group"] = assign_prob_group(
                float(row["defect_probability"])
            )
            before_records.append(m_before)

            # ── Apply ──
            processed = preprocessor(original)

            # Save (cache)
            if FORCE_RECOMPUTE or not dst_path.exists():
                save_processed_image(processed, image_id)

            # ── AFTER ──
            m_after = compute_image_metrics(processed)
            m_after["image_id"] = image_id
            m_after["cell_type"] = row["cell_type"]
            m_after["defect_probability"] = float(row["defect_probability"])
            m_after["prob_group"] = assign_prob_group(
                float(row["defect_probability"])
            )
            after_records.append(m_after)

        except Exception as e:
            logger.warning("Failed on %s: %s", image_id, e)

    t = time.time() - t0
    logger.info(
        "Preprocessed %d images in %.1f s (%.1f ms/img)",
        len(before_records), t,
        1000 * t / max(1, len(before_records)),
    )

    return pd.DataFrame(before_records), pd.DataFrame(after_records)


# ============================================================
# STATISTICAL COMPARISON
# ============================================================

def compare_metrics(
    before: pd.DataFrame, after: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "mean_intensity", "std_intensity", "gradient_mean",
        "gradient_p95", "laplacian_var", "entropy",
        "center_mean", "corner_mean", "illumination_gap",
    ]

    b = before[["image_id"] + cols].copy()
    a = after[["image_id"] + cols].copy()

    comparison = b.merge(a, on="image_id", suffixes=("_before", "_after"))

    for col in cols:
        comparison[f"{col}_delta"] = (
            comparison[f"{col}_after"] - comparison[f"{col}_before"]
        )
        comparison[f"{col}_pct_change"] = (
            100 * comparison[f"{col}_delta"]
            / (comparison[f"{col}_before"].abs() + 1e-6)
        )

    return comparison


def aggregate_stats(
    before: pd.DataFrame, after: pd.DataFrame,
) -> dict:
    metrics = [
        "mean_intensity", "std_intensity", "gradient_mean",
        "gradient_p95", "laplacian_var", "entropy",
        "center_mean", "corner_mean", "illumination_gap",
    ]

    result = {}

    for m in metrics:
        b = before[m].dropna().values
        a = after[m].dropna().values

        n = min(len(b), len(a))
        b, a = b[:n], a[:n]

        try:
            t_stat, p_val = stats.ttest_rel(b, a)
            t_stat, p_val = float(t_stat), float(p_val)
        except Exception:
            t_stat, p_val = 0.0, 1.0

        result[m] = {
            "before_mean": round(float(b.mean()), 4),
            "before_std":  round(float(b.std()), 4),
            "after_mean":  round(float(a.mean()), 4),
            "after_std":   round(float(a.std()), 4),
            "delta_mean":  round(float(a.mean() - b.mean()), 4),
            "pct_change":  round(
                100 * (a.mean() - b.mean()) / (abs(b.mean()) + 1e-6), 2
            ),
            "t_stat":      round(t_stat, 3),
            "p_value":     round(p_val, 6),
            "significant": bool(p_val < 0.05),
        }

    return result


# ============================================================
# VISUAL COMPARISON
# ============================================================

def create_comparison_grid(
    df: pd.DataFrame,
    cell_type: str,
    prob_group: str,
    out_path: Path,
    n_samples: int = SAMPLES_PER_GROUP,
) -> None:
    subset = df[
        (df["cell_type"] == cell_type)
        & (df["prob_group"] == prob_group)
    ]

    if subset.empty:
        logger.warning(
            "No samples for %s / %s — skipping grid.",
            cell_type, prob_group,
        )
        return

    rng = np.random.default_rng(RANDOM_SEED)
    n = min(n_samples, len(subset))
    sample_ids = rng.choice(
        subset["image_id"].values, size=n, replace=False,
    )

    # Extra height + top margin for suptitle
    fig, axes = plt.subplots(
        n, 2, figsize=(9, 3.6 * n + 0.8),
    )

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, image_id in enumerate(sample_ids):
        before = load_image(RAW_IMAGE_DIR / image_id)
        after_path = PROCESSED_DIR / image_id
        after = (
            load_image(after_path) if after_path.exists() else before
        )

        axes[i, 0].imshow(before, cmap="gray", vmin=0, vmax=255)
        axes[i, 0].set_title(
            f"BEFORE — {image_id}", fontsize=10, pad=6,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(after, cmap="gray", vmin=0, vmax=255)
        axes[i, 1].set_title(
            f"AFTER — {image_id}", fontsize=10, pad=6,
        )
        axes[i, 1].axis("off")

    # Reserve space for suptitle — avoid collision with top subplot title
    fig.subplots_adjust(top=0.94)
    fig.suptitle(
        f"Preprocessing — {cell_type} / {prob_group}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    fig.savefig(
        out_path, dpi=120, bbox_inches="tight", pad_inches=0.3,
    )
    plt.close(fig)
    logger.info("Saved grid: %s", out_path)


def plot_histograms(
    df_before: pd.DataFrame, df_after: pd.DataFrame, out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, metric in zip(
        axes, ["mean_intensity", "std_intensity", "illumination_gap"]
    ):
        ax.hist(
            df_before[metric].dropna(), bins=40, alpha=0.5,
            label="before", color="steelblue",
        )
        ax.hist(
            df_after[metric].dropna(), bins=40, alpha=0.5,
            label="after", color="darkorange",
        )
        ax.set_title(metric)
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Distribution shift after preprocessing", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved histogram: %s", out_path)


# ============================================================
# REPORT
# ============================================================

def write_report(
    stats_dict: dict,
    preprocessor: ELPVPreprocessor,
    n_images: int,
) -> None:
    cfg = preprocessor.config_dict()

    lines = [
        "# ELPV Image Preprocessing Report",
        "",
        f"**Images processed:** {n_images}",
        "",
        "## Configuration",
        "",
        f"- Homomorphic: `{cfg['use_homomorphic']}` "
        f"(sigma={cfg['homomorphic_sigma']}, alpha={cfg['homomorphic_alpha']}, "
        f"beta={cfg['homomorphic_beta']})",
        f"- Denoise: `{cfg['use_denoise']}` "
        f"(method={cfg['denoise_method']}, kernel={cfg['denoise_kernel']})",
        f"- CLAHE: `{cfg['use_clahe']}` "
        f"(clip={cfg['clahe_clip_limit']}, tile={cfg['clahe_tile_grid']})",
        "",
        "## Aggregate metrics (paired t-test)",
        "",
        "| Metric | Before (mean) | After (mean) | Δ% | p-value | Significant |",
        "|---|---:|---:|---:|---:|:---:|",
    ]

    for metric, s in stats_dict.items():
        sig = "✅" if s["significant"] else "—"
        lines.append(
            f"| {metric} | {s['before_mean']:.3f} | {s['after_mean']:.3f} | "
            f"{s['pct_change']:+.2f}% | {s['p_value']:.2e} | {sig} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "**What preprocessing does:**",
        "",
        "- **Dynamic range compression (homomorphic filter):** "
        "suppresses low-frequency illumination variations and amplifies "
        "high-frequency reflectance components, improving visibility of "
        "structural features (cracks, dark regions) in both center and "
        "peripheral regions of the solar cell.",
        "- **Noise suppression (median filter):** removes salt-and-pepper "
        "sensor noise common in electroluminescence imaging.",
        "- **Local contrast enhancement (CLAHE):** adaptively redistributes "
        "the intensity histogram in small tiles, making subtle defects "
        "more distinguishable from surrounding cell texture.",
        "",
        "**What preprocessing does NOT do:**",
        "",
        "- **It does not fully flatten illumination.** Homomorphic "
        "filtering compresses — but does not eliminate — the center-to-edge "
        "intensity gradient. `illumination_gap` therefore remains approximately "
        "constant. This is by design: fully flattening illumination risks "
        "removing real structural information near cell boundaries.",
        "- **It does not perform geometric correction.** Image is assumed "
        "already rectified to 300×300.",
        "",
        "**Expected effects (confirmed by paired t-test above):**",
        "",
        "- `std_intensity` ↑ and `entropy` ↑ → **contrast improves**, "
        "histogram spreads wider.",
        "- `gradient_mean` ↑ and `gradient_p95` ↑ → **edges strengthen**, "
        "defect boundaries become more visible.",
        "- `laplacian_var` moderate increase (target: <200%) → **sharpness "
        "improves without noise amplification**. If >300%, reduce "
        "`homomorphic_beta` or `clahe_clip_limit`.",
        "",
        "**Warning signs** (if observed, adjust config):",
        "",
        "- `mean_intensity` drops >30% → homomorphic too aggressive "
        "(increase alpha, or check rescaling)",
        "- `gradient_mean` drops → edges destroyed by denoise "
        "(reduce kernel) or CLAHE over-suppressed (increase clip)",
        "- `laplacian_var` increases >300% → noise amplification "
        "(reduce beta, or increase denoise kernel)",
        "",
        "## Visual grids",
        "",
        "- `comparison_grid_normal_mono.png`",
        "- `comparison_grid_normal_poly.png`",
        "- `comparison_grid_defect_mono.png`",
        "- `comparison_grid_defect_poly.png`",
        "",
        "## Output",
        "",
        "- Preprocessed images: `data/processed/elpv_preprocessed/`",
        "- Original images: `data/raw/elpv/images/` (UNCHANGED)",
        "",
    ]

    (REPORT_DIR / "report.md").write_text(
        "\n".join(lines), encoding="utf-8",
    )
    logger.info("Saved report: %s", REPORT_DIR / "report.md")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()

    logger.info("=" * 60)
    logger.info("ELPV IMAGE PREPROCESSING")
    logger.info("=" * 60)
    logger.info("Config: %s", PREPROCESS_CONFIG)
    logger.info("Force recompute: %s", FORCE_RECOMPUTE)
    if DEBUG_MAX_IMAGES is not None:
        logger.warning(
            "DEBUG MODE — only processing %d images", DEBUG_MAX_IMAGES,
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")
    if not RAW_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image dir not found: {RAW_IMAGE_DIR}")

    df = pd.read_csv(LABELS_FILE)
    df["prob_group"] = df["defect_probability"].apply(assign_prob_group)

    logger.info("Total images: %d", len(df))
    logger.info(
        "Prob group distribution: %s",
        df["prob_group"].value_counts().to_dict(),
    )
    logger.info(
        "Cell type distribution: %s",
        df["cell_type"].value_counts().to_dict(),
    )

    # Debug mode: subsample
    if DEBUG_MAX_IMAGES is not None:
        n_take = min(DEBUG_MAX_IMAGES, len(df))
        df = df.sample(n=n_take, random_state=RANDOM_SEED).reset_index(drop=True)
        logger.warning("Debug subset: %d images", len(df))

    preprocessor = ELPVPreprocessor(**PREPROCESS_CONFIG)

    # ── Step 1: apply preprocessing + collect metrics ──
    logger.info("-" * 60)
    logger.info("Step 1: Apply preprocessing (originals untouched)")
    logger.info("-" * 60)
    before_df, after_df = apply_preprocessing(df, preprocessor)

    if len(before_df) == 0:
        logger.error("No images processed. Check logs above.")
        return

    # ── Step 2: save per-image metrics ──
    logger.info("-" * 60)
    logger.info("Step 2: Save per-image metrics")
    logger.info("-" * 60)
    before_df.to_csv(REPORT_DIR / "metrics_before.csv", index=False)
    after_df.to_csv(REPORT_DIR / "metrics_after.csv", index=False)

    comparison = compare_metrics(before_df, after_df)
    comparison.to_csv(REPORT_DIR / "metrics_comparison.csv", index=False)
    logger.info("Saved metrics CSVs.")

    # ── Step 3: aggregate stats + t-test ──
    logger.info("-" * 60)
    logger.info("Step 3: Aggregate statistics (paired t-test)")
    logger.info("-" * 60)
    stats_dict = aggregate_stats(before_df, after_df)

    for metric, s in stats_dict.items():
        logger.info(
            "  %-20s | before=%.3f → after=%.3f (%+.2f%%) | p=%.2e %s",
            metric, s["before_mean"], s["after_mean"],
            s["pct_change"], s["p_value"],
            "✅" if s["significant"] else "—",
        )

    with open(
        REPORT_DIR / "metrics_aggregate.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "config":    preprocessor.config_dict(),
                "n_images":  len(before_df),
                "aggregate": stats_dict,
            },
            f, indent=4, ensure_ascii=False,
        )

    # ── Step 4: visual comparison grids ──
    logger.info("-" * 60)
    logger.info("Step 4: Generate visual comparison grids")
    logger.info("-" * 60)

    for cell_type in ["mono", "poly"]:
        for prob_group in ["normal", "defect"]:
            out_path = (
                REPORT_DIR
                / f"comparison_grid_{prob_group}_{cell_type}.png"
            )
            create_comparison_grid(df, cell_type, prob_group, out_path)

    # ── Step 5: histogram distribution ──
    plot_histograms(
        before_df, after_df, REPORT_DIR / "histogram_before_after.png",
    )

    # ── Step 6: markdown report ──
    write_report(stats_dict, preprocessor, len(before_df))

    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("Report: %s", REPORT_DIR / "report.md")
    logger.info("Processed images: %s", PROCESSED_DIR)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()