"""
06_consensus_pseudo_gt.py (v5.1)

v5.1 fixes:
  - FIX walrus operator in vis (np.zeros(hm.shape))
  - FRANGI_NOISE_FRANGI_THR: 0.15 → 0.10 (conservative with new scale)
  - Log frangi_seed_thr clearly at start
"""

from __future__ import annotations
import json, logging, sys, time
from pathlib import Path
import cv2, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS_FILE = PROJECT_ROOT / "data/raw/elpv/labels.csv"
PREPROCESSED_DIR = PROJECT_ROOT / "data/processed/elpv_preprocessed"
STEP04_DIR = PROJECT_ROOT / "reports/04_unsupervised_models"
STEP05_DIR = PROJECT_ROOT / "reports/05_pseudo_gt"
OUTPUT_DIR = PROJECT_ROOT / "reports/06_consensus_pseudo_gt"
HEATMAP_DIR = STEP05_DIR / "heatmaps"
CALIB_DIR = STEP05_DIR / "calibration"
VAL_CSV = STEP04_DIR / "normal_val.csv"
MASK_DIR = OUTPUT_DIR / "masks"
VIS_DIR = OUTPUT_DIR / "visualization"

CORE_MODELS = ("patchcore", "ae", "vae")

USE_FRANGI_RESCUE = True
USE_FRANGI_NOISE_FILTER = True

# Pass B (rescue)
FRANGI_RESCUE_PC_THR = 0.70
FRANGI_RESCUE_MIN_AREA = 30
FRANGI_RESCUE_MAX_AREA = 500

# Pass C (noise filter) — v5.1: 0.15 → 0.10
FRANGI_NOISE_FRANGI_THR = 0.10     # was 0.15 — more conservative with new scale
FRANGI_NOISE_PC_THR = 0.30
FRANGI_NOISE_MIN_AREA = 200
FRANGI_NOISE_MAX_ASPECT = 3.0

# Core consensus
PATCHCORE_BORDER_PCT = 0.05
MAX_FINAL_AREA_PCT = 0.40
SEED_VOTES_REQUIRED = 2
GROW_VOTES_REQUIRED = 2

OPEN_KERNEL_SIZE = 3; OPEN_ITER = 1
CLOSE_KERNEL_SIZE = 3; CLOSE_ITER = 2
LARGEST_CC_DOMINANCE = 0.60
MIN_COMPONENT_AREA = 0.001
OUTPUT_SIZE = 300

PROB_ZERO_THRESHOLD = 0.05
PROB_ONE_THRESHOLD = 0.95
N_VIS_PER_GROUP = 6
RANDOM_SEED = 42
LOG_LEVEL = logging.INFO


def setup_logging():
    for d in (OUTPUT_DIR, MASK_DIR, VIS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(OUTPUT_DIR / "consensus.log", encoding="utf-8")],
        force=True)


logger = logging.getLogger(__name__)


def load_heatmap(image_id, model):
    p = HEATMAP_DIR / f"{Path(image_id).stem}_{model}.npy"
    return np.load(p).astype(np.float32) if p.exists() else None


def make_pc_valid(hm_shape):
    h, w = hm_shape
    ph, pw = int(h * PATCHCORE_BORDER_PCT), int(w * PATCHCORE_BORDER_PCT)
    valid = np.ones((h, w), dtype=bool)
    valid[:ph, :] = False; valid[-ph:, :] = False
    valid[:, :pw] = False; valid[:, -pw:] = False
    return valid


def keep_cc_touching_seed(grow, seed):
    n, labels = cv2.connectedComponents(grow.astype(np.uint8), connectivity=8)
    if n <= 1: return np.zeros_like(grow, dtype=bool)
    keep = set()
    for cid in range(1, n):
        if ((labels == cid) & seed).any(): keep.add(cid)
    return np.isin(labels, list(keep)) if keep else np.zeros_like(grow, dtype=bool)


def morph_cleanup(mask_bool):
    m = mask_bool.astype(np.uint8) * 255
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ko, iterations=OPEN_ITER)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kc, iterations=CLOSE_ITER)
    return m


def filter_components(mask_u8):
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1: return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0: return mask_u8
    largest = int(areas.max()); total = int(areas.sum())
    H, W = mask_u8.shape
    if total > 0 and largest / total >= LARGEST_CC_DOMINANCE:
        return ((labels == 1 + int(np.argmax(areas))) * 255).astype(np.uint8)
    min_a = max(10, int(H * W * MIN_COMPONENT_AREA))
    keep = np.zeros_like(mask_u8)
    for cid in range(1, n):
        if stats[cid, cv2.CC_STAT_AREA] >= min_a: keep[labels == cid] = 255
    return keep


def resize_300(m):
    r = cv2.resize(m, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_NEAREST)
    return ((r > 127).astype(np.uint8)) * 255


def pass_a_core(hm_pc, hm_ae, hm_va, thresholds, pc_valid):
    seed_pc = (hm_pc > thresholds["patchcore"]["seed_threshold"]) & pc_valid
    seed_ae = hm_ae > thresholds["ae"]["seed_threshold"]
    seed_va = hm_va > thresholds["vae"]["seed_threshold"]
    seed = (seed_pc.astype(int) + seed_ae.astype(int) + seed_va.astype(int)) >= SEED_VOTES_REQUIRED

    grow_pc = (hm_pc > thresholds["patchcore"]["grow_threshold"]) & pc_valid
    grow_ae = hm_ae > thresholds["ae"]["grow_threshold"]
    grow_va = hm_va > thresholds["vae"]["grow_threshold"]
    grow = (grow_pc.astype(int) + grow_ae.astype(int) + grow_va.astype(int)) >= GROW_VOTES_REQUIRED

    final = keep_cc_touching_seed(grow, seed)
    used_fb = False
    if final.mean() > MAX_FINAL_AREA_PCT:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final = cv2.dilate(seed.astype(np.uint8), k, iterations=2).astype(bool)
        used_fb = True

    mask_u8 = morph_cleanup(final)
    mask_u8 = filter_components(mask_u8)
    return mask_u8, used_fb


def pass_c_noise_filter(mask, hm_frangi, hm_pc):
    if hm_frangi is None: return mask, 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 1: return mask, 0
    keep = mask.copy(); removed = 0
    for cid in range(1, n):
        a = stats[cid, cv2.CC_STAT_AREA]
        if a >= FRANGI_NOISE_MIN_AREA: continue
        w = stats[cid, cv2.CC_STAT_WIDTH]; h = stats[cid, cv2.CC_STAT_HEIGHT]
        asp = max(w, h) / (min(w, h) + 1e-6)
        if asp >= FRANGI_NOISE_MAX_ASPECT: continue
        comp = labels == cid
        frangi_mean = float(hm_frangi[comp].mean())
        pc_mean = float(hm_pc[comp].mean())
        if frangi_mean < FRANGI_NOISE_FRANGI_THR and pc_mean < FRANGI_NOISE_PC_THR:
            keep[comp] = 0; removed += 1
    return keep, removed


def pass_b_rescue(mask_core, hm_frangi, hm_pc, frangi_seed_thr, pc_valid):
    if hm_frangi is None: return mask_core, 0
    cand = ((hm_frangi > frangi_seed_thr) &
            (hm_pc > FRANGI_RESCUE_PC_THR) &
            pc_valid &
            ~(mask_core > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    rescue = np.zeros_like(cand); cnt = 0
    for cid in range(1, n):
        a = stats[cid, cv2.CC_STAT_AREA]
        if FRANGI_RESCUE_MIN_AREA <= a <= FRANGI_RESCUE_MAX_AREA:
            rescue[labels == cid] = 255; cnt += 1
    return np.maximum(mask_core, rescue), cnt


def process_image(image_id, thresholds, frangi_seed_thr):
    stem = Path(image_id).stem
    hm_pc = load_heatmap(image_id, "patchcore")
    hm_ae = load_heatmap(image_id, "ae")
    hm_va = load_heatmap(image_id, "vae")
    if hm_pc is None or hm_ae is None or hm_va is None:
        return {"image_id": image_id, "status": "missing"}

    hm_fr = load_heatmap(image_id, "frangi")
    pc_valid = make_pc_valid(hm_pc.shape)

    mask_core, used_fb = pass_a_core(hm_pc, hm_ae, hm_va, thresholds, pc_valid)

    noise_rm = 0
    if USE_FRANGI_NOISE_FILTER and hm_fr is not None:
        mask_clean, noise_rm = pass_c_noise_filter(mask_core, hm_fr, hm_pc)
    else:
        mask_clean = mask_core

    rescue_cnt = 0
    if USE_FRANGI_RESCUE and hm_fr is not None:
        mask_rescued, rescue_cnt = pass_b_rescue(mask_clean, hm_fr, hm_pc, frangi_seed_thr, pc_valid)
    else:
        mask_rescued = mask_clean

    final = resize_300(mask_rescued)
    cv2.imwrite(str(MASK_DIR / f"{stem}.png"), final)

    return {"image_id": image_id, "status": "ok",
            "core_area": float((mask_core > 0).mean()),
            "final_area": float((final > 0).mean()),
            "used_fallback": used_fb,
            "rescue_count": rescue_cnt,
            "noise_removed": noise_rm,
            "has_frangi": hm_fr is not None}


def save_vis(image_id, original, model_heatmaps, model_masks, final_mask, stats, out_path):
    n = len(model_heatmaps)
    fig, axes = plt.subplots(1, 1 + n * 2 + 1, figsize=(5 + n * 6, 4.5))
    axes = np.atleast_1d(axes)
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Input", fontsize=10); axes[0].axis("off")
    for idx, m in enumerate(model_heatmaps.keys()):
        hm = model_heatmaps[m]; mk = model_masks[m]
        axes[1 + idx*2].imshow(hm, cmap="hot", vmin=0, vmax=1)
        axes[1 + idx*2].set_title(m, fontsize=10); axes[1 + idx*2].axis("off")
        axes[2 + idx*2].imshow(mk, cmap="gray", vmin=0, vmax=255)
        axes[2 + idx*2].set_title(f"{m}\nmask {100*(mk>0).mean():.1f}%", fontsize=9)
        axes[2 + idx*2].axis("off")
    ax_last = axes[-1]
    ax_last.imshow(final_mask, cmap="gray", vmin=0, vmax=255)
    ax_last.set_title(f"FINAL\n{100*(final_mask>0).mean():.1f}%", fontsize=10, fontweight="bold")
    ax_last.axis("off")
    info = (f"core:{100*stats['core_area']:.1f}% "
            f"final:{100*stats['final_area']:.1f}% "
            f"fb:{stats['used_fallback']} "
            f"resc:{stats['rescue_count']} "
            f"rm:{stats['noise_removed']}")
    fig.suptitle(f"{image_id}  [{info}]", fontsize=10, family="monospace")
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
    logger.info("=" * 62); logger.info("CONSENSUS (v5.1)"); logger.info("=" * 62)
    logger.info("Frangi rescue: %s | noise filter: %s", USE_FRANGI_RESCUE, USE_FRANGI_NOISE_FILTER)
    logger.info("Frangi noise threshold: %.2f (was 0.15)", FRANGI_NOISE_FRANGI_THR)

    thresholds = {}
    for m in CORE_MODELS:
        p = CALIB_DIR / f"{m}.json"
        if not p.exists(): raise FileNotFoundError(f"Missing: {p}")
        with open(p) as f: thresholds[m] = json.load(f)

    frangi_calib = CALIB_DIR / "frangi.json"
    frangi_seed_thr = 0.3
    if frangi_calib.exists():
        with open(frangi_calib) as f:
            fc = json.load(f)
            frangi_seed_thr = fc["seed_threshold"]
            logger.info("Frangi seed threshold: %.4f (from calib)", frangi_seed_thr)
            logger.info("Frangi global_scale: %.4f", fc.get("global_scale", 0))
            if frangi_seed_thr > 0.9:
                logger.warning("⚠ Frangi seed ≈ 1.0 — rerun step 05 with FRANGI_GLOBAL_PCT=99.9")

    df = pd.read_csv(LABELS_FILE)
    t0 = time.time()
    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Consensus", ncols=80):
        s = process_image(row["image_id"], thresholds, frangi_seed_thr)
        s["cell_type"] = row["cell_type"]
        s["defect_probability"] = float(row["defect_probability"])
        rows.append(s)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    t_el = time.time() - t0

    ok = summary_df[summary_df["status"] == "ok"]
    normal = ok[ok["defect_probability"] < PROB_ZERO_THRESHOLD]
    defect = ok[ok["defect_probability"] > PROB_ONE_THRESHOLD]
    border = ok[(ok["defect_probability"] >= PROB_ZERO_THRESHOLD) &
                (ok["defect_probability"] <= PROB_ONE_THRESHOLD)]

    stats = {
        "n_images": len(ok), "processing_time_seconds": round(t_el, 1),
        "use_frangi_rescue": USE_FRANGI_RESCUE,
        "use_frangi_noise_filter": USE_FRANGI_NOISE_FILTER,
        "frangi_seed_thr_used": frangi_seed_thr,
        "fallback_count": int(ok["used_fallback"].sum()),
        "total_rescued": int(ok["rescue_count"].sum()),
        "total_noise_removed": int(ok["noise_removed"].sum()),
        "images_with_frangi": int(ok["has_frangi"].sum()),
        "groups": {
            "normal": {"n": int(len(normal)),
                       "final_area_mean": round(float(normal["final_area"].mean()), 4),
                       "empty_pct": round(float((normal["final_area"] == 0).mean()), 4)},
            "defect": {"n": int(len(defect)),
                       "final_area_mean": round(float(defect["final_area"].mean()), 4),
                       "empty_pct": round(float((defect["final_area"] == 0).mean()), 4)},
            "borderline": {"n": int(len(border)),
                           "final_area_mean": round(float(border["final_area"].mean()), 4),
                           "empty_pct": round(float((border["final_area"] == 0).mean()), 4)}}}
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)

    vis_df = pick_vis(df)
    for _, row in tqdm(vis_df.iterrows(), total=len(vis_df), desc="Vis"):
        iid = row["image_id"]; stem = Path(iid).stem
        orig = np.asarray(
            Image.open(PREPROCESSED_DIR / iid).convert("L").resize(
                (OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR),
            dtype=np.uint8)
        model_heatmaps, model_masks = {}, {}
        for m in ("patchcore", "ae", "vae", "frangi"):
            hm = load_heatmap(iid, m)
            if hm is None: continue
            mk_path = STEP05_DIR / "masks" / f"{stem}_{m}.png"
            mk = cv2.imread(str(mk_path), cv2.IMREAD_GRAYSCALE) if mk_path.exists() \
                 else np.zeros(hm.shape, dtype=np.uint8)
            model_heatmaps[m] = cv2.resize(hm, (OUTPUT_SIZE, OUTPUT_SIZE))
            model_masks[m] = cv2.resize(mk, (OUTPUT_SIZE, OUTPUT_SIZE),
                                        interpolation=cv2.INTER_NEAREST)
        final_mask = cv2.imread(str(MASK_DIR / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if final_mask is None:
            final_mask = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
        row_stats = ok[ok["image_id"] == iid].iloc[0].to_dict() if not ok[ok["image_id"] == iid].empty else {
            "core_area": 0, "final_area": 0, "used_fallback": False,
            "rescue_count": 0, "noise_removed": 0}
        save_vis(iid, orig, model_heatmaps, model_masks, final_mask, row_stats, VIS_DIR / f"{stem}.png")

    report = ["# Consensus Pseudo-GT Report (v5.1)", "",
              f"**Total:** {stats['n_images']}",
              f"**Time:** {stats['processing_time_seconds']}s",
              f"**Frangi seed threshold used:** {frangi_seed_thr:.4f}",
              f"**Fallback:** {stats['fallback_count']}",
              f"**Total rescued:** {stats['total_rescued']}",
              f"**Total noise removed:** {stats['total_noise_removed']}", "",
              "## Mask areas by group", "",
              "| Group | n | Final area | Empty % |", "|---|---:|---:|---:|"]
    for g, s in stats["groups"].items():
        report.append(f"| {g} | {s['n']} | {s['final_area_mean']:.4f} | {s['empty_pct']*100:.1f}% |")
    report += ["", "## v5.1 fixes",
               "- Pass C frangi threshold 0.15 → 0.10 (conservative with new scale)",
               "- Walrus operator removed in vis",
               "",
               "## What to check",
               "- `frangi_seed_thr` should be 0.2-0.6 (not ~1.0)",
               "- `total_rescued` should be > 0 (Pass B fires)",
               "- `total_noise_removed` should be > 0 (Pass C fires)",
               "- `groups.normal.empty_pct` should stay > 85%"]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")

    logger.info("DONE. %s", OUTPUT_DIR)
    for g, s in stats["groups"].items():
        logger.info("  %-12s n=%d | final=%.4f | empty=%.1f%%",
                    g, s["n"], s["final_area_mean"], s["empty_pct"]*100)
    logger.info("Rescued total: %d | Noise removed total: %d",
                stats["total_rescued"], stats["total_noise_removed"])


if __name__ == "__main__":
    main()