"""
elpv_preprocessing.py

Shared image processing module for ELPV pipeline.
Không ghi đè ảnh gốc — chỉ cung cấp transform functions.

Covers:
  - Homomorphic filtering (illumination correction) — with proper rescaling
  - CLAHE (contrast enhancement)
  - Denoise (median, Gaussian, bilateral)
  - Sobel edge detection (cho localization)
  - Morphological mask cleanup
  - Connected component filtering
  - Image quality metrics (before/after comparison)

Used by:
  - 03_image_preprocessing.py   (apply + report)
  - 04_train_unsupervised.py    (load preprocessed)
  - 05_generate_pseudo_gt.py    (load preprocessed)
  - 06_consensus_pseudo_gt.py
  - 10_localize.py              (Sobel + morphology)
  - 11_evaluate.py

Change log:
  - v2: Fixed homomorphic filter — added rescale to [0,255] to prevent
        image collapsing to near-black. Adjusted default params
        (sigma=15, alpha=0.7, beta=1.3, clip=1.5).
        Reordered pipeline: denoise → homomorphic → CLAHE.
"""

from __future__ import annotations

import cv2
import numpy as np


# ============================================================
# Preprocessing pipeline
# ============================================================

class ELPVPreprocessor:
    """
    Configurable preprocessing pipeline for ELPV images.

    Order (if enabled):
        1. Denoise             — median / gaussian / bilateral
        2. Homomorphic filter  — illumination correction (with rescale)
        3. CLAHE               — local contrast enhancement

    Use the SAME instance for train and inference to avoid
    train-test distribution mismatch.
    """

    def __init__(
        self,
        use_homomorphic: bool = True,
        use_clahe: bool = True,
        use_denoise: bool = True,
        denoise_method: str = "median",
        # Homomorphic params — conservative defaults
        homomorphic_sigma: float = 15.0,
        homomorphic_alpha: float = 0.7,
        homomorphic_beta: float = 1.3,
        # CLAHE params
        clahe_clip_limit: float = 1.5,
        clahe_tile_grid: tuple = (8, 8),
        # Denoise params
        denoise_kernel: int = 3,
    ):
        self.use_homomorphic = use_homomorphic
        self.use_clahe = use_clahe
        self.use_denoise = use_denoise
        self.denoise_method = denoise_method
        self.homomorphic_sigma = homomorphic_sigma
        self.homomorphic_alpha = homomorphic_alpha
        self.homomorphic_beta = homomorphic_beta
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.denoise_kernel = denoise_kernel

    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Order: denoise → homomorphic → CLAHE
        # Denoise first to remove sensor noise before amplification
        if self.use_denoise:
            image = self._denoise(image)
        if self.use_homomorphic:
            image = self._homomorphic(image)
        if self.use_clahe:
            image = self._clahe(image)
        return image

    def _homomorphic(self, image):
        """
        Homomorphic filter with dynamic range rescaling.

        Without the final rescale, output collapses to near-black
        because exp(alpha*low + beta*high) - 1 can produce very
        small values when low is suppressed.
        """
        img = image.astype(np.float32) + 1.0
        log_img = np.log(img)

        low = cv2.GaussianBlur(log_img, (0, 0), self.homomorphic_sigma)
        high = log_img - low

        out = np.exp(
            self.homomorphic_alpha * low + self.homomorphic_beta * high
        ) - 1.0

        # Rescale to preserve dynamic range — CRITICAL
        out = np.clip(out, 0, None)
        out_max = out.max()
        if out_max > 0:
            out = out / out_max * 255.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def _clahe(self, image):
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_grid,
        )
        return clahe.apply(image)

    def _denoise(self, image):
        if self.denoise_method == "median":
            return cv2.medianBlur(image, self.denoise_kernel)
        if self.denoise_method == "gaussian":
            k = self.denoise_kernel
            return cv2.GaussianBlur(image, (k, k), 0)
        if self.denoise_method == "bilateral":
            k = self.denoise_kernel
            return cv2.bilateralFilter(image, k * 2, 75, 75)
        return image

    def config_dict(self) -> dict:
        return {
            "use_homomorphic":    self.use_homomorphic,
            "use_clahe":          self.use_clahe,
            "use_denoise":        self.use_denoise,
            "denoise_method":     self.denoise_method,
            "homomorphic_sigma":  self.homomorphic_sigma,
            "homomorphic_alpha":  self.homomorphic_alpha,
            "homomorphic_beta":   self.homomorphic_beta,
            "clahe_clip_limit":   self.clahe_clip_limit,
            "clahe_tile_grid":    list(self.clahe_tile_grid),
            "denoise_kernel":     self.denoise_kernel,
        }


# ============================================================
# Image quality metrics (for before/after comparison)
# ============================================================

def compute_image_metrics(image: np.ndarray) -> dict:
    """Compute quantitative metrics used to compare pre vs post."""
    img_f = image.astype(np.float32)

    # Gradient magnitude (edge strength)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Sharpness (Laplacian variance)
    lap_var = float(cv2.Laplacian(img_f, cv2.CV_32F).var())

    # Histogram entropy
    hist, _ = np.histogram(image, bins=32, range=(0, 256), density=True)
    hist_pos = hist[hist > 0]
    entropy = float(-np.sum(hist_pos * np.log2(hist_pos)))

    # Illumination uniformity (center vs corner)
    h, w = image.shape
    cy, cx = h // 2, w // 2
    center_mean = float(
        img_f[cy - 30:cy + 30, cx - 30:cx + 30].mean()
    )
    corners = np.concatenate([
        img_f[:30, :30].ravel(),
        img_f[:30, -30:].ravel(),
        img_f[-30:, :30].ravel(),
        img_f[-30:, -30:].ravel(),
    ])
    corner_mean = float(corners.mean())

    return {
        "mean_intensity":   float(img_f.mean()),
        "std_intensity":    float(img_f.std()),
        "min_intensity":    int(image.min()),
        "max_intensity":    int(image.max()),
        "gradient_mean":    float(grad_mag.mean()),
        "gradient_p95":     float(np.percentile(grad_mag, 95)),
        "laplacian_var":    lap_var,
        "entropy":          entropy,
        "center_mean":      center_mean,
        "corner_mean":      corner_mean,
        "illumination_gap": abs(center_mean - corner_mean),
    }


# ============================================================
# Post-processing (localization + mask cleanup)
# ============================================================

def sobel_edge_map(image: np.ndarray) -> np.ndarray:
    """Returns 0-1 edge strength map (for localization refinement)."""
    gx = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    max_v = mag.max()
    return (mag / max_v) if max_v > 0 else mag


def clean_mask(
    mask_binary: np.ndarray,
    kernel_size: int = 3,
    opening_iter: int = 1,
    closing_iter: int = 2,
    keep_largest: bool = True,
) -> np.ndarray:
    """Morphological cleanup. Input/output: uint8 {0, 255}."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )

    m = cv2.morphologyEx(
        mask_binary, cv2.MORPH_OPEN, kernel, iterations=opening_iter
    )
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, kernel, iterations=closing_iter
    )

    if keep_largest:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            m = ((labels == largest) * 255).astype(np.uint8)

    return m