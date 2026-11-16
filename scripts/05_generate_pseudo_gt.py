"""
05_generate_pseudo_gt.py (v5.2)

Fix v5.1 bug: calibrate_neural returned only seed/grow, but normalize() needs lo/hi.
Now returns {lo, hi, seed_threshold, grow_threshold} → KeyError fixed.

Frangi v5.1 fix already working: global_pct=99.9 → seed ≈ 0.33 (not ~1.0).

Change log v5.2:
  - FIX: calibrate_neural returns lo, hi
"""

from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
import torch.nn.functional as F, torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from skimage.filters import frangi
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    frangi = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_FILE = PROJECT_ROOT / "data/raw/elpv/labels.csv"
PREPROCESSED_DIR = PROJECT_ROOT / "data/processed/elpv_preprocessed"
MODELS_DIR = PROJECT_ROOT / "reports/04_unsupervised_models"
OUTPUT_DIR = PROJECT_ROOT / "reports/05_pseudo_gt"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
MASK_DIR = OUTPUT_DIR / "masks"
VIS_DIR = OUTPUT_DIR / "visualization"
CALIB_DIR = OUTPUT_DIR / "calibration"
VAL_CSV = MODELS_DIR / "normal_val.csv"

USE_FRANGI = True
FRANGI_SIGMAS = (1.0, 1.5, 2.0)
FRANGI_BLACK_RIDGES = True
FRANGI_BASELINE_PCT = 90.0
FRANGI_GLOBAL_PCT = 99.9

IMAGE_SIZE = 224
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5
BATCH_SIZE = 8
RANDOM_SEED = 42

CALIB_LO_PCT = 1.0
CALIB_HI_PCT = 99.9
CALIB_SEED_PCT = 99.0
CALIB_GROW_PCT = 97.0

PATCHCORE_LAYERS = ("layer2", "layer3")
PATCHCORE_KNN_K = 5
PATCHCORE_BATCH_CHUNK = 1000
PATCHCORE_GAUSSIAN_SIGMA = 4.0

PROB_ZERO_THRESHOLD = 0.05
PROB_ONE_THRESHOLD = 0.95
N_VIS_PER_GROUP = 6
DEVICE = torch.device("cpu")
LOG_LEVEL = logging.INFO


def setup_logging():
    for d in (OUTPUT_DIR, HEATMAP_DIR, MASK_DIR, VIS_DIR, CALIB_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "inference.log", encoding="utf-8"),
        ], force=True)


logger = logging.getLogger(__name__)


class ConvAutoEncoder(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, 4, 2, 1), torch.nn.ReLU(inplace=True))
        self.flatten_dim = 256 * 14 * 14
        self.fc_enc = torch.nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = torch.nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 1, 4, 2, 1), torch.nn.Tanh())

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        z = self.fc_enc(h)
        return self.decoder(self.fc_dec(z).view(x.size(0), 256, 14, 14))


class ConvVAE(torch.nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(128, 256, 4, 2, 1), torch.nn.ReLU(inplace=True))
        self.flatten_dim = 256 * 14 * 14
        self.fc_mu = torch.nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = torch.nn.Linear(self.flatten_dim, latent_dim)
        self.fc_dec = torch.nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 32, 4, 2, 1), torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(32, 1, 4, 2, 1), torch.nn.Tanh())

    def forward(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h); logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decoder(self.fc_dec(z).view(x.size(0), 256, 14, 14))

    def recon_from_mu(self, x):
        h = self.encoder(x).flatten(1)
        mu = self.fc_mu(h)
        return self.decoder(self.fc_dec(mu).view(x.size(0), 256, 14, 14))


class PatchCoreExtractor(torch.nn.Module):
    def __init__(self, layers=("layer2", "layer3")):
        super().__init__()
        bb = tvm.wide_resnet50_2(weights=tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.layer_names = layers; self._features = {}
        for name in layers:
            getattr(bb, name).register_forward_hook(self._hook(name))
        self.backbone = bb; self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _hook(self, name):
        def fn(_m, _i, out): self._features[name] = out
        return fn

    @torch.no_grad()
    def forward(self, x):
        if x.shape[1] == 1: x = x.repeat(1, 3, 1, 1)
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        self._features = {}; self.backbone(x)
        f2 = self._features[self.layer_names[0]]
        f3 = self._features[self.layer_names[1]]
        return f2, F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)


def load_image_uint8(path):
    return np.asarray(
        Image.open(path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
        dtype=np.uint8)


def compute_frangi_raw(image_uint8):
    img = image_uint8.astype(np.float32) / 255.0
    return frangi(img, sigmas=FRANGI_SIGMAS, black_ridges=FRANGI_BLACK_RIDGES,
                  mode="reflect").astype(np.float32)


class ELPVDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(np.asarray(Image.open(self.paths[idx]).convert("L"), dtype=np.uint8))


def build_transform():
    return T.Compose([T.ToPILImage(), T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                      T.ToTensor(), T.Normalize(mean=[NORMALIZE_MEAN], std=[NORMALIZE_STD])])


def make_loader(df, bs):
    return DataLoader(ELPVDataset([PREPROCESSED_DIR / fid for fid in df["image_id"]], build_transform()),
                      batch_size=bs, shuffle=False, num_workers=0)


@torch.no_grad()
def raw_patchcore(batch, ext, mb):
    batch = batch.to(DEVICE); f2, f3 = ext(batch)
    f2 = F.avg_pool2d(f2, 3, 1, 1); f3 = F.avg_pool2d(f3, 3, 1, 1)
    feats = torch.cat([f2, f3], 1); B, C, H, W = feats.shape
    flat = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
    mb_t = torch.from_numpy(mb).to(DEVICE); all_d = []
    for s in range(0, len(flat), PATCHCORE_BATCH_CHUNK):
        e = min(s + PATCHCORE_BATCH_CHUNK, len(flat))
        chunk = torch.from_numpy(flat[s:e]).to(DEVICE)
        c_sq = (chunk ** 2).sum(1, keepdim=True)
        m_sq = (mb_t ** 2).sum(1).unsqueeze(0)
        d = (c_sq + m_sq - 2 * chunk @ mb_t.T).clamp(min=0)
        kth = min(PATCHCORE_KNN_K, d.shape[1])
        knn, _ = d.topk(kth, dim=1, largest=False); all_d.append(knn[:, -1].cpu())
    kth = torch.cat(all_d).numpy().reshape(B, H, W)
    out = np.empty((B, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i in range(B):
        up = cv2.resize(kth[i], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        out[i] = cv2.GaussianBlur(up, (0, 0), PATCHCORE_GAUSSIAN_SIGMA)
    return out


@torch.no_grad()
def raw_ae(batch, ae):
    batch = batch.to(DEVICE)
    return ((ae(batch) - batch) ** 2).mean(1).cpu().numpy()


@torch.no_grad()
def raw_vae(batch, vae):
    batch = batch.to(DEVICE)
    return ((vae.recon_from_mu(batch) - batch) ** 2).mean(1).cpu().numpy()


def load_neural_models():
    mb = np.load(MODELS_DIR / "patchcore/memory_bank.npz")["features"].astype(np.float32)
    ext = PatchCoreExtractor(PATCHCORE_LAYERS).to(DEVICE); ext.eval()
    ae = ConvAutoEncoder(256)
    ae.load_state_dict(torch.load(MODELS_DIR / "autoencoder/model.pt", map_location=DEVICE)); ae.eval()
    vae = ConvVAE(256)
    vae.load_state_dict(torch.load(MODELS_DIR / "vae/model.pt", map_location=DEVICE)); vae.eval()
    return mb, ext, ae, vae


def calibrate_neural(name, val_df, infer_fn, save_path):
    """v5.2 FIX: return includes lo, hi (normalize needs them)."""
    loader = make_loader(val_df, BATCH_SIZE)
    all_raw, per_max = [], []
    for batch in tqdm(loader, desc=f"Calib {name}", ncols=80):
        raw = infer_fn(batch)
        all_raw.append(raw.reshape(-1))
        per_max.append(raw.reshape(raw.shape[0], -1).max(1))
    all_raw = np.concatenate(all_raw); per_max = np.concatenate(per_max)
    lo = float(np.percentile(all_raw, CALIB_LO_PCT))
    hi = float(np.percentile(all_raw, CALIB_HI_PCT))
    norm = np.clip((all_raw - lo) / (hi - lo + 1e-8), 0, 1)
    seed_t = float(np.percentile(norm, CALIB_SEED_PCT))
    grow_t = float(np.percentile(norm, CALIB_GROW_PCT))
    with open(save_path, "w") as f:
        json.dump({"model": name, "n_val": int(len(val_df)), "lo": lo, "hi": hi,
                   "seed_threshold": seed_t, "grow_threshold": grow_t,
                   "raw_max_stats": {"mean": float(per_max.mean()),
                                     "p50": float(np.percentile(per_max, 50)),
                                     "p95": float(np.percentile(per_max, 95))}}, f, indent=4)
    logger.info("  %s: lo=%.4f hi=%.4f seed=%.4f grow=%.4f", name, lo, hi, seed_t, grow_t)
    # v5.2 FIX: include lo, hi in return
    return {"lo": lo, "hi": hi, "seed_threshold": seed_t, "grow_threshold": grow_t}


def calibrate_frangi(val_df, save_path, baseline_path):
    logger.info("Calibrating frangi on %d val images…", len(val_df))
    responses = []
    for fid in tqdm(val_df["image_id"], desc="Calib frangi", ncols=80):
        try:
            responses.append(compute_frangi_raw(load_image_uint8(PREPROCESSED_DIR / fid)))
        except FileNotFoundError:
            continue
    stack = np.stack(responses, 0)

    baseline = np.percentile(stack, FRANGI_BASELINE_PCT, axis=0).astype(np.float32)
    np.save(baseline_path, baseline)

    baseline_sub = np.clip(stack - baseline[None], 0, None)
    global_scale = float(np.percentile(baseline_sub.ravel(), FRANGI_GLOBAL_PCT))
    if global_scale < 1e-3:
        global_scale = 1e-3
    hm_norm = np.clip(baseline_sub / global_scale, 0, 1)
    seed_t = float(np.percentile(hm_norm.ravel(), CALIB_SEED_PCT))
    grow_t = float(np.percentile(hm_norm.ravel(), CALIB_GROW_PCT))

    per_max = baseline_sub.reshape(len(baseline_sub), -1).max(1)
    with open(save_path, "w") as f:
        json.dump({"model": "frangi", "n_val": len(responses),
                   "baseline_pct": FRANGI_BASELINE_PCT,
                   "global_pct": FRANGI_GLOBAL_PCT,
                   "global_scale": global_scale,
                   "sigmas": list(FRANGI_SIGMAS),
                   "black_ridges": FRANGI_BLACK_RIDGES,
                   "seed_threshold": seed_t, "grow_threshold": grow_t,
                   "raw_max_stats": {"mean": float(per_max.mean()),
                                     "p50": float(np.percentile(per_max, 50)),
                                     "p95": float(np.percentile(per_max, 95))}}, f, indent=4)

    logger.info("  frangi baseline: max=%.4f mean=%.4f", baseline.max(), baseline.mean())
    logger.info("  frangi baseline_sub: p50=%.4f p95=%.4f p99=%.4f p99.9=%.4f",
                float(np.percentile(baseline_sub, 50)),
                float(np.percentile(baseline_sub, 95)),
                float(np.percentile(baseline_sub, 99)),
                global_scale)
    logger.info("  frangi global_scale=%.4f | SEED=%.4f GROW=%.4f", global_scale, seed_t, grow_t)
    if seed_t > 0.9:
        logger.warning("  ⚠ Frangi seed ≈ 1.0 — bug? mask will be empty")

    return {"seed_threshold": seed_t, "grow_threshold": grow_t,
            "global_scale": global_scale, "baseline": baseline}


def normalize(raw, c):
    return np.clip((raw - c["lo"]) / (c["hi"] - c["lo"] + 1e-8), 0, 1)


def save_vis(image_id, original, maps, out_path):
    n = len(maps)
    fig, axes = plt.subplots(1, 1 + n * 2, figsize=(4 + n * 6, 4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input", fontsize=10); axes[0].axis("off")
    for idx, (name, (hm, m)) in enumerate(maps.items()):
        axes[1 + idx*2].imshow(hm, cmap="hot", vmin=0, vmax=1)
        axes[1 + idx*2].set_title(f"{name}\nheatmap", fontsize=10); axes[1 + idx*2].axis("off")
        axes[2 + idx*2].imshow(m, cmap="gray", vmin=0, vmax=255)
        axes[2 + idx*2].set_title(f"{name}\nmask ({100*(m>0).mean():.1f}%)", fontsize=10)
        axes[2 + idx*2].axis("off")
    fig.suptitle(image_id, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def pick_vis(df):
    df = df.copy()
    df["prob_group"] = df["defect_probability"].apply(
        lambda p: "normal" if p <= PROB_ZERO_THRESHOLD
        else ("defect" if p >= PROB_ONE_THRESHOLD else "borderline"))
    rng = np.random.default_rng(RANDOM_SEED)
    parts = []
    for (_, pg), sub in df.groupby(["cell_type", "prob_group"]):
        n = min(N_VIS_PER_GROUP, len(sub))
        parts.append(sub.iloc[rng.choice(len(sub), n, replace=False)])
    return pd.concat(parts, ignore_index=True)


def main():
    setup_logging()
    models = ["patchcore", "ae", "vae"]
    use_fr = USE_FRANGI and HAS_SKIMAGE
    if USE_FRANGI and not HAS_SKIMAGE:
        logger.warning("USE_FRANGI=True but skimage not installed.")
    if use_fr:
        models.append("frangi")

    logger.info("=" * 62); logger.info("GENERATE PSEUDO-GT (v5.2)"); logger.info("=" * 62)
    logger.info("Models: %s", models)
    logger.info("Frangi global_pct=%.1f", FRANGI_GLOBAL_PCT)

    t0 = time.time()
    df = pd.read_csv(LABELS_FILE); val_df = pd.read_csv(VAL_CSV)
    logger.info("Total: %d | Val: %d", len(df), len(val_df))

    missing = [fid for fid in df["image_id"] if not (PREPROCESSED_DIR / fid).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing. First: {missing[0]}")

    if "patchcore" in models:
        mb, ext, ae, vae = load_neural_models()

    logger.info("-" * 62); logger.info("STEP 1: Calibration"); logger.info("-" * 62)
    calibs = {}
    if "patchcore" in models:
        calibs["patchcore"] = calibrate_neural(
            "patchcore", val_df, lambda b: raw_patchcore(b, ext, mb),
            CALIB_DIR / "patchcore.json")
    if "ae" in models:
        calibs["ae"] = calibrate_neural("ae", val_df, lambda b: raw_ae(b, ae), CALIB_DIR / "ae.json")
    if "vae" in models:
        calibs["vae"] = calibrate_neural("vae", val_df, lambda b: raw_vae(b, vae), CALIB_DIR / "vae.json")
    if "frangi" in models:
        calibs["frangi"] = calibrate_frangi(
            val_df, CALIB_DIR / "frangi.json", CALIB_DIR / "frangi_baseline.npy")
        frangi_baseline = calibs["frangi"]["baseline"]
        frangi_global_scale = calibs["frangi"]["global_scale"]

    logger.info("-" * 62); logger.info("STEP 2: Inference"); logger.info("-" * 62)
    loader = make_loader(df, BATCH_SIZE); rows = []; offset = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Inference", ncols=80)):
        B = batch.shape[0]; ids = df["image_id"].iloc[offset:offset + B].tolist()
        raws = {}
        if "patchcore" in models: raws["patchcore"] = raw_patchcore(batch, ext, mb)
        if "ae" in models: raws["ae"] = raw_ae(batch, ae)
        if "vae" in models: raws["vae"] = raw_vae(batch, vae)
        if "frangi" in models:
            raws["frangi"] = np.stack([
                np.clip(compute_frangi_raw(load_image_uint8(PREPROCESSED_DIR / fid)) - frangi_baseline,
                        0, None)
                for fid in ids], 0)

        for i, iid in enumerate(ids):
            stem = Path(iid).stem
            row_meta = df.iloc[offset + i]
            row = {"image_id": iid, "cell_type": row_meta["cell_type"],
                   "defect_probability": float(row_meta["defect_probability"])}
            for m in models:
                raw = raws[m][i]
                if m == "frangi":
                    hm = np.clip(raw / frangi_global_scale, 0, 1)
                    score_map = raw
                else:
                    hm = normalize(raw, calibs[m])
                    score_map = raw
                np.save(HEATMAP_DIR / f"{stem}_{m}.npy", hm.astype(np.float16))
                mask = ((hm > calibs[m]["seed_threshold"]).astype(np.uint8)) * 255
                cv2.imwrite(str(MASK_DIR / f"{stem}_{m}.png"), mask)
                k = max(1, int(score_map.size * 0.01))
                row[f"{m}_score_raw_max"] = float(score_map.max())
                row[f"{m}_score_top1pct"] = float(np.partition(score_map.flatten(), -k)[-k:].mean())
                row[f"{m}_mask_area"] = float((mask > 0).mean())
            rows.append(row)
        offset += B
        if (batch_idx + 1) % 20 == 0:
            el = time.time() - t0
            logger.info("  %d/%d | %.1f img/s", offset, len(df), offset / el)

    t_inf = time.time() - t0
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    logger.info("-" * 62); logger.info("STEP 3: AUROC"); logger.info("-" * 62)
    eval_df = summary_df[(summary_df["defect_probability"] < PROB_ZERO_THRESHOLD) |
                         (summary_df["defect_probability"] > PROB_ONE_THRESHOLD)].copy()
    eval_df["label"] = (eval_df["defect_probability"] > PROB_ONE_THRESHOLD).astype(int)
    val_ids = set(val_df["image_id"])
    held_df = eval_df[(eval_df["label"] == 1) | (eval_df["image_id"].isin(val_ids))].copy()

    auroc_full, auroc_held = {}, {}
    for m in models:
        try:
            auroc_full[m] = float(roc_auc_score(eval_df["label"], eval_df[f"{m}_score_raw_max"]))
            auroc_held[m] = float(roc_auc_score(held_df["label"], held_df[f"{m}_score_raw_max"]))
            logger.info("  %-10s full=%.4f held=%.4f", m, auroc_full[m], auroc_held[m])
        except Exception as e:
            logger.warning("  %s failed: %s", m, e)
            auroc_full[m] = auroc_held[m] = None

    normal_df = summary_df[summary_df["defect_probability"] < PROB_ZERO_THRESHOLD]
    defect_df = summary_df[summary_df["defect_probability"] > PROB_ONE_THRESHOLD]
    stats = {"n_images": len(summary_df), "models": models, "use_frangi": use_fr,
             "frangi_global_pct": FRANGI_GLOBAL_PCT,
             "inference_time_seconds": round(t_inf, 1),
             "auroc_including_train": auroc_full, "auroc_heldout": auroc_held, "per_model": {}}
    for m in models:
        stats["per_model"][m] = {
            "normal": {"n": int(len(normal_df)),
                       "mask_area_mean": round(float(normal_df[f"{m}_mask_area"].mean()), 4)},
            "defect": {"n": int(len(defect_df)),
                       "mask_area_mean": round(float(defect_df[f"{m}_mask_area"].mean()), 4)}}
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    logger.info("-" * 62); logger.info("STEP 4: Visualization"); logger.info("-" * 62)
    vis_df = pick_vis(df)
    for _, row in tqdm(vis_df.iterrows(), total=len(vis_df), desc="Vis"):
        iid = row["image_id"]; stem = Path(iid).stem
        orig = load_image_uint8(PREPROCESSED_DIR / iid)
        maps = {}
        for m in models:
            hm = np.load(HEATMAP_DIR / f"{stem}_{m}.npy").astype(np.float32)
            mk = cv2.imread(str(MASK_DIR / f"{stem}_{m}.png"), cv2.IMREAD_GRAYSCALE)
            maps[m] = (hm, mk)
        save_vis(iid, orig, maps, VIS_DIR / f"{stem}.png")

    report = ["# Pseudo-GT Generation Report (v5.2)", "",
              f"**Total images:** {stats['n_images']}",
              f"**Models:** {', '.join(models)}",
              f"**Frangi global_pct:** {FRANGI_GLOBAL_PCT}",
              f"**Inference time:** {stats['inference_time_seconds']} s", "",
              "## Calibration", "",
              "| Model | Seed | Grow |", "|---|---:|---:|"]
    for m in models:
        c = calibs[m]
        report.append(f"| {m} | {c['seed_threshold']:.4f} | {c['grow_threshold']:.4f} |")
    report += ["", "## AUROC", "", "| Model | Full | Held-out |", "|---|---:|---:|"]
    for m in models:
        report.append(f"| {m} | {auroc_full[m]:.4f} | {auroc_held[m]:.4f} |"
                      if auroc_full[m] else f"| {m} | — | — |")
    report += ["", "## Mask areas", "", "| Model | Normal | Defect | Gap |", "|---|---:|---:|---:|"]
    for m in models:
        n = stats["per_model"][m]["normal"]["mask_area_mean"]
        d = stats["per_model"][m]["defect"]["mask_area_mean"]
        report.append(f"| {m} | {n:.4f} | {d:.4f} | {d-n:+.4f} |")
    report += ["", "## Frangi v5.2",
               "- global_pct=99.9 → seed ≈ 0.33 (not ~1.0)",
               "- calibrate_neural returns lo, hi (fix KeyError in v5.1)"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")
    logger.info("DONE.")


if __name__ == "__main__":
    main()