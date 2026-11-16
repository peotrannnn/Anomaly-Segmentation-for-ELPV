"""
04_train_unsupervised.py

Train 3 unsupervised anomaly detection models on ELPV normal images
(prob < 0.05). Models use PREPROCESSED images from 03_image_preprocessing.py.

Models (all custom PyTorch, CPU-only):
  1. PatchCore: WideResNet50 features + kNN memory bank
  2. ConvAutoEncoder: reconstruction error
  3. ConvVAE: reconstruction + KL divergence

Requirements:
  pip install torch torchvision tqdm opencv-python

Input:
  data/processed/elpv_preprocessed/       ← preprocessed images from 03
  data/raw/elpv/labels.csv

Output:
  reports/04_unsupervised_models/
    ├── patchcore/memory_bank.npz
    ├── autoencoder/model.pt
    ├── vae/model.pt
    ├── normal_train.csv
    ├── normal_val.csv
    ├── training_report.json
    └── training.log

Note:
  - Assumes 03_image_preprocessing.py has already been run.
  - To use raw (unpreprocessed) images instead, set USE_PREPROCESSED=False.
  - Does NOT read manual_review.csv — pipeline independent of manual annotation.

Change log:
  - v2: Fixed decoder output range mismatch.
        Input normalized to [-1, 1] but decoder ended with Sigmoid() → [0, 1].
        Any pixel with original intensity < 127 (dark regions: cell border,
        busbar, crack, dark defect) could never be reconstructed.
        Changed final activation from Sigmoid() to Tanh() in BOTH
        ConvAutoEncoder and ConvVAE so output range matches input range.
        PatchCore unaffected (no reconstruction).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELS_FILE = PROJECT_ROOT / "data" / "raw" / "elpv" / "labels.csv"
RAW_IMAGE_DIR       = PROJECT_ROOT / "data" / "raw" / "elpv" / "images"
PREPROCESSED_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "elpv_preprocessed"
)
OUTPUT_DIR = PROJECT_ROOT / "reports" / "04_unsupervised_models"

# Use preprocessed images (from 03) or raw?
USE_PREPROCESSED = True

# Normal filtering
NORMAL_PROB_THRESHOLD = 0.05
VAL_FRACTION          = 0.10
RANDOM_SEED           = 42

# Image preprocessing
IMAGE_SIZE     = 224
NORMALIZE_MEAN = 0.5
NORMALIZE_STD  = 0.5
# Note: with mean=0.5, std=0.5, pixel range [0,1] → [-1, 1]

# PatchCore
PATCHCORE_BACKBONE      = "wide_resnet50_2"
PATCHCORE_LAYERS        = ("layer2", "layer3")
PATCHCORE_CORESET_RATIO = 0.01
PATCHCORE_BATCH_SIZE    = 8

# AutoEncoder
AE_LATENT_DIM = 256
AE_EPOCHS     = 50
AE_BATCH_SIZE = 16
AE_LR         = 1e-3
AE_EARLY_STOP = 10

# VAE
VAE_BETA       = 1.0
VAE_EPOCHS     = 50
VAE_BATCH_SIZE = 16
VAE_LR         = 1e-3
VAE_EARLY_STOP = 10

# Run control
FORCE_RETRAIN = False

DEVICE    = torch.device("cpu")
LOG_LEVEL = logging.INFO


# ============================================================
# LOGGING
# ============================================================

def setup_logging() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                OUTPUT_DIR / "training.log", encoding="utf-8"
            ),
        ],
        force=True,
    )


logger = logging.getLogger(__name__)


# ============================================================
# DATASET
# ============================================================

class ELPVDataset(Dataset):
    """Grayscale PNG → (1,H,W) tensor, normalized to [-1, 1]."""

    def __init__(self, image_paths: list[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("L")
        arr = np.asarray(img, dtype=np.uint8)

        if self.transform is not None:
            return self.transform(arr)

        # Fallback: normalize [0,255] → [-1, 1]
        tensor = torch.from_numpy(arr).unsqueeze(0).float() / 255.0
        return (tensor - 0.5) / 0.5


def build_transform() -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),                       # [0, 1]
        T.Normalize(
            mean=[NORMALIZE_MEAN], std=[NORMALIZE_STD]
        ),                                   # → [-1, 1]
    ])


def get_image_dir() -> Path:
    if USE_PREPROCESSED:
        if not PREPROCESSED_IMAGE_DIR.exists():
            raise FileNotFoundError(
                f"Preprocessed dir not found: {PREPROCESSED_IMAGE_DIR}. "
                f"Run 03_image_preprocessing.py first, "
                f"or set USE_PREPROCESSED=False."
            )
        return PREPROCESSED_IMAGE_DIR
    return RAW_IMAGE_DIR


def make_loader(
    df: pd.DataFrame, batch_size: int, shuffle: bool,
) -> DataLoader:
    image_dir = get_image_dir()
    paths = [image_dir / fid for fid in df["image_id"]]
    ds = ELPVDataset(paths, transform=build_transform())
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=False,
    )


def prepare_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LABELS_FILE.exists():
        raise FileNotFoundError(f"Labels file not found: {LABELS_FILE}")

    df = pd.read_csv(LABELS_FILE)
    logger.info("Total images: %d", len(df))

    normal = df[
        df["defect_probability"] < NORMAL_PROB_THRESHOLD
    ].copy()
    logger.info(
        "Normal images (prob < %.2f): %d",
        NORMAL_PROB_THRESHOLD, len(normal),
    )

    if len(normal) < 10:
        raise ValueError(
            f"Too few normal images ({len(normal)}). "
            f"Check NORMAL_PROB_THRESHOLD."
        )

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(normal))
    n_val = max(1, int(len(normal) * VAL_FRACTION))

    train = normal.iloc[perm[n_val:]].reset_index(drop=True)
    val   = normal.iloc[perm[:n_val]].reset_index(drop=True)

    logger.info("Train: %d | Val: %d", len(train), len(val))

    train.to_csv(OUTPUT_DIR / "normal_train.csv", index=False)
    val.to_csv(OUTPUT_DIR / "normal_val.csv", index=False)

    return train, val


# ============================================================
# MODEL 1 — PatchCore (custom)
# ============================================================

class PatchCoreExtractor(nn.Module):
    """Pretrained WideResNet50 feature extractor (multi-scale)."""

    def __init__(self, layers: tuple[str, ...] = ("layer2", "layer3")):
        super().__init__()

        backbone = tvm.wide_resnet50_2(
            weights=tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V2
        )

        self.layer_names = layers
        self._features: dict[str, torch.Tensor] = {}

        for name in layers:
            getattr(backbone, name).register_forward_hook(
                self._hook(name)
            )

        self.backbone = backbone
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _hook(self, name: str):
        def fn(_module, _inputs, output):
            self._features[name] = output
        return fn

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x in [-1, 1] → [0, 1]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = (x + 1.0) / 2.0

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406], device=x.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], device=x.device
        ).view(1, 3, 1, 1)
        x = (x - mean) / std

        self._features = {}
        self.backbone(x)

        f2 = self._features[self.layer_names[0]]
        f3 = self._features[self.layer_names[1]]

        f3_up = F.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        return f2, f3_up


def build_patchcore_memory_bank(
    df: pd.DataFrame,
    extractor: PatchCoreExtractor,
    coreset_ratio: float,
) -> np.ndarray:
    logger.info(
        "Building PatchCore memory bank "
        "(batch-wise subsampling, ratio=%.3f)…",
        coreset_ratio,
    )

    loader = make_loader(df, PATCHCORE_BATCH_SIZE, shuffle=False)
    rng = np.random.default_rng(RANDOM_SEED)
    all_batches: list[np.ndarray] = []

    for batch in tqdm(
        loader, desc="Extracting PatchCore features", ncols=80,
    ):
        batch = batch.to(DEVICE)
        f2, f3 = extractor(batch)

        f2 = F.avg_pool2d(f2, kernel_size=3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, kernel_size=3, stride=1, padding=1)

        feats = torch.cat([f2, f3], dim=1)
        B, C, H, W = feats.shape

        flat = (
            feats.permute(0, 2, 3, 1)
            .reshape(-1, C)
            .cpu().numpy()
        )

        n_keep = max(1, int(len(flat) * coreset_ratio))
        idx = rng.choice(len(flat), size=n_keep, replace=False)
        all_batches.append(flat[idx])

    memory_bank = np.concatenate(all_batches, axis=0).astype(np.float32)

    logger.info(
        "Memory bank: %d patches × %d dims (%.1f MB)",
        len(memory_bank),
        memory_bank.shape[1],
        memory_bank.nbytes / 1e6,
    )
    return memory_bank


# ============================================================
# MODEL 2 — ConvAutoEncoder
# ============================================================

class ConvAutoEncoder(nn.Module):
    """
    4-down / 4-up conv AE. 224 → 14 latent spatial.
    Input [-1, 1], output [-1, 1] (Tanh at end).
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),    nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )

        self.flatten_dim = 256 * 14 * 14
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),    # ← FIX: [-1, 1] to match input range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        z = self.fc_enc(h)
        h2 = self.fc_dec(z).view(x.size(0), 256, 14, 14)
        return self.decoder(h2)


# ============================================================
# MODEL 3 — ConvVAE
# ============================================================

class ConvVAE(nn.Module):
    """
    VAE with same conv backbone as AE.
    Input [-1, 1], output [-1, 1] (Tanh at end).
    """

    def __init__(self, latent_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),    nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(inplace=True),
        )

        self.flatten_dim = 256 * 14 * 14
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec    = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),    # ← FIX: [-1, 1] to match input range
        )

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        h2 = self.fc_dec(z).view(x.size(0), 256, 14, 14)
        recon = self.decoder(h2)
        return recon, mu, logvar


# ============================================================
# TRAINING LOOP (AE + VAE)
# ============================================================

def _train_epochs(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn,
    epochs: int,
    lr: float,
    early_stop: int,
    ckpt_path: Path,
    tag: str,
) -> float:
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs,
    )

    best_val = float("inf")
    patience = 0
    saved_any = False

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            batch = batch.to(DEVICE)
            loss = loss_fn(model, batch)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # validate
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                val_losses.append(loss_fn(model, batch).item())

        train_m = (
            float(np.mean(train_losses)) if train_losses else float("nan")
        )
        val_m = (
            float(np.mean(val_losses)) if val_losses else float("nan")
        )

        logger.info(
            "%s Epoch %d/%d | train=%.5f | val=%.5f",
            tag, epoch, epochs, train_m, val_m,
        )
        sched.step()

        improved = np.isfinite(val_m) and val_m < best_val - 1e-5
        if improved:
            best_val = val_m
            patience = 0
            torch.save(model.state_dict(), ckpt_path)
            saved_any = True
        else:
            patience += 1
            if patience >= early_stop:
                logger.info(
                    "%s early stop at epoch %d (best val=%.5f)",
                    tag, epoch, best_val,
                )
                break

    if not saved_any:
        logger.warning(
            "%s: no checkpoint was saved (val never improved).", tag,
        )
    else:
        try:
            model.load_state_dict(
                torch.load(ckpt_path, map_location=DEVICE)
            )
            logger.info("%s restored best weights", tag)
        except Exception as e:
            logger.warning("%s failed to restore best weights: %s", tag, e)

    return best_val


def ae_loss(model: ConvAutoEncoder, batch: torch.Tensor) -> torch.Tensor:
    recon = model(batch)
    return F.mse_loss(recon, batch)


def vae_loss(model: ConvVAE, batch: torch.Tensor) -> torch.Tensor:
    recon, mu, logvar = model(batch)
    rec = F.mse_loss(recon, batch, reduction="sum") / batch.size(0)
    kl = (
        -0.5
        * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        / batch.size(0)
    )
    return rec + VAE_BETA * kl


# ============================================================
# SKIP-IF-EXISTS
# ============================================================

def _exists_patchcore() -> bool:
    return (OUTPUT_DIR / "patchcore" / "memory_bank.npz").exists()


def _exists_ae() -> bool:
    return (OUTPUT_DIR / "autoencoder" / "model.pt").exists()


def _exists_vae() -> bool:
    return (OUTPUT_DIR / "vae" / "model.pt").exists()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    setup_logging()

    for sub in ("patchcore", "autoencoder", "vae"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    image_dir = get_image_dir()

    logger.info("=" * 62)
    logger.info("TRAIN UNSUPERVISED MODELS ON ELPV NORMAL IMAGES")
    logger.info("=" * 62)
    logger.info("Device:           %s", DEVICE)
    logger.info("Image size:       %d", IMAGE_SIZE)
    logger.info("Input range:      [-1, 1]")
    logger.info("Image source:     %s", image_dir)
    logger.info("Use preprocessed: %s", USE_PREPROCESSED)
    logger.info("Output dir:       %s", OUTPUT_DIR)
    logger.info("Force retrain:    %s", FORCE_RETRAIN)

    train_df, val_df = prepare_splits()

    report: dict = {
        "device":           str(DEVICE),
        "image_size":       IMAGE_SIZE,
        "input_range":      [-1.0, 1.0],
        "use_preprocessed": USE_PREPROCESSED,
        "image_source":     str(image_dir),
        "normal_threshold": NORMAL_PROB_THRESHOLD,
        "random_seed":      RANDOM_SEED,
        "train_images":     int(len(train_df)),
        "val_images":       int(len(val_df)),
        "force_retrain":    FORCE_RETRAIN,
        "models":           {},
    }

    # ─────────────────────────────────────────────
    # Model 1/3 — PatchCore
    # ─────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info("Model 1/3: PatchCore")
    logger.info("-" * 62)

    if _exists_patchcore() and not FORCE_RETRAIN:
        logger.info("PatchCore already trained → skipping.")
        try:
            with np.load(
                OUTPUT_DIR / "patchcore" / "memory_bank.npz"
            ) as z:
                mb_size = int(z["features"].shape[0])
                mb_dim = int(z["features"].shape[1])
            report["models"]["patchcore"] = {
                "status": "skipped_existing",
                "memory_bank_size": mb_size,
                "feature_dim": mb_dim,
            }
        except Exception:
            report["models"]["patchcore"] = {"status": "skipped_existing"}
    else:
        t0 = time.time()
        extractor = PatchCoreExtractor(
            layers=PATCHCORE_LAYERS
        ).to(DEVICE)
        mb = build_patchcore_memory_bank(
            train_df, extractor, PATCHCORE_CORESET_RATIO,
        )

        np.savez_compressed(
            OUTPUT_DIR / "patchcore" / "memory_bank.npz",
            features=mb,
            layers=np.array(PATCHCORE_LAYERS),
            backbone=PATCHCORE_BACKBONE,
            image_size=IMAGE_SIZE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
            use_preprocessed=USE_PREPROCESSED,
        )
        t_pc = time.time() - t0
        logger.info("PatchCore done in %.1f s.", t_pc)

        report["models"]["patchcore"] = {
            "status": "trained",
            "backbone": PATCHCORE_BACKBONE,
            "layers": list(PATCHCORE_LAYERS),
            "coreset_ratio": PATCHCORE_CORESET_RATIO,
            "memory_bank_size": int(mb.shape[0]),
            "feature_dim": int(mb.shape[1]),
            "training_time_seconds": round(t_pc, 1),
        }

    # ─────────────────────────────────────────────
    # Model 2/3 — AutoEncoder
    # ─────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info("Model 2/3: AutoEncoder")
    logger.info("-" * 62)

    if _exists_ae() and not FORCE_RETRAIN:
        logger.info("AutoEncoder already trained → skipping.")
        report["models"]["autoencoder"] = {"status": "skipped_existing"}
    else:
        t0 = time.time()
        ae = ConvAutoEncoder(latent_dim=AE_LATENT_DIM)
        ae_train_loader = make_loader(
            train_df, AE_BATCH_SIZE, shuffle=True,
        )
        ae_val_loader = make_loader(
            val_df, AE_BATCH_SIZE, shuffle=False,
        )

        ae_best = _train_epochs(
            ae, ae_train_loader, ae_val_loader, ae_loss,
            AE_EPOCHS, AE_LR, AE_EARLY_STOP,
            OUTPUT_DIR / "autoencoder" / "model.pt",
            tag="AE",
        )
        t_ae = time.time() - t0
        logger.info(
            "AutoEncoder done in %.1f s. Best val=%.6f",
            t_ae, ae_best,
        )

        report["models"]["autoencoder"] = {
            "status": "trained",
            "latent_dim": AE_LATENT_DIM,
            "epochs_configured": AE_EPOCHS,
            "best_val_loss": round(ae_best, 6),
            "training_time_seconds": round(t_ae, 1),
            "decoder_activation": "Tanh",
        }

    # ─────────────────────────────────────────────
    # Model 3/3 — VAE
    # ─────────────────────────────────────────────
    logger.info("-" * 62)
    logger.info("Model 3/3: VAE")
    logger.info("-" * 62)

    if _exists_vae() and not FORCE_RETRAIN:
        logger.info("VAE already trained → skipping.")
        report["models"]["vae"] = {"status": "skipped_existing"}
    else:
        t0 = time.time()
        vae = ConvVAE(latent_dim=AE_LATENT_DIM)
        vae_train_loader = make_loader(
            train_df, VAE_BATCH_SIZE, shuffle=True,
        )
        vae_val_loader = make_loader(
            val_df, VAE_BATCH_SIZE, shuffle=False,
        )

        vae_best = _train_epochs(
            vae, vae_train_loader, vae_val_loader, vae_loss,
            VAE_EPOCHS, VAE_LR, VAE_EARLY_STOP,
            OUTPUT_DIR / "vae" / "model.pt",
            tag="VAE",
        )
        t_vae = time.time() - t0
        logger.info(
            "VAE done in %.1f s. Best val=%.5f", t_vae, vae_best,
        )

        report["models"]["vae"] = {
            "status": "trained",
            "latent_dim": AE_LATENT_DIM,
            "beta": VAE_BETA,
            "epochs_configured": VAE_EPOCHS,
            "best_val_loss": round(vae_best, 5),
            "training_time_seconds": round(t_vae, 1),
            "decoder_activation": "Tanh",
        }

    # ─────────────────────────────────────────────
    # Save report
    # ─────────────────────────────────────────────
    with open(
        OUTPUT_DIR / "training_report.json", "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    logger.info("=" * 62)
    logger.info("ALL MODELS TRAINED. Output: %s", OUTPUT_DIR)
    logger.info("=" * 62)


if __name__ == "__main__":
    main()