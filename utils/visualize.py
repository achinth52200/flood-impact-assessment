"""
Visualization utilities for flood impact maps.
Generates RGB composites, flood overlays, and metric plots.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import rasterio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PRED_DIR, OUTPUT_DIR


FLOOD_COLOR    = (0.0, 0.5, 1.0, 0.6)   # semi-transparent blue
NO_FLOOD_COLOR = (0.9, 0.9, 0.7, 0.0)   # transparent


# ─── RGB composite ────────────────────────────────────────────────────────────

def tif_to_rgb(img_path: Path,
               rgb_channels: tuple[int, int, int] = (4, 3, 2)) -> np.ndarray:
    """
    Load a multi-channel GeoTIFF and return an (H, W, 3) uint8 RGB image.
    Default channels assume 7-band stack: S1(0,1) + S2(B2=2,B3=3,B4=4,B8=5,B11=6)
    so rgb_channels=(4,3,2) → S2 natural colour (R,G,B).
    """
    with rasterio.open(img_path) as src:
        arr = src.read().astype(np.float32)   # (C, H, W)

    r, g, b = [arr[c] for c in rgb_channels]

    def norm(x):
        lo, hi = np.percentile(x, [2, 98])
        return np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)

    rgb = np.stack([norm(r), norm(g), norm(b)], axis=-1)
    return (rgb * 255).astype(np.uint8)


def sar_to_gray(img_path: Path, channel: int = 0) -> np.ndarray:
    """Return a single SAR channel as (H, W) float32 in [0,1]."""
    with rasterio.open(img_path) as src:
        arr = src.read(channel + 1).astype(np.float32)
    lo, hi = np.percentile(arr, [2, 98])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


# ─── Overlay plots ────────────────────────────────────────────────────────────

def plot_flood_overlay(img_path: Path,
                       mask_path: Path,
                       pred_path: Path | None = None,
                       out_file: Path | None = None,
                       title: str = "") -> plt.Figure:
    """
    Three-panel figure:
      1. Sentinel-2 RGB composite
      2. Ground-truth flood mask overlay
      3. Predicted flood mask overlay  (if pred_path given)
    """
    rgb  = tif_to_rgb(img_path)

    with rasterio.open(mask_path) as src:
        gt_mask = src.read(1).astype(np.uint8)

    n_cols = 3 if pred_path else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    fig.suptitle(title or img_path.stem, fontsize=14, fontweight="bold")

    # Panel 1 — RGB
    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 RGB")
    axes[0].axis("off")

    # Panel 2 — Ground-truth
    axes[1].imshow(rgb)
    flood_rgba = np.zeros((*gt_mask.shape, 4))
    flood_rgba[gt_mask == 1] = FLOOD_COLOR
    axes[1].imshow(flood_rgba, interpolation="nearest")
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Panel 3 — Prediction
    if pred_path:
        with rasterio.open(pred_path) as src:
            pred_mask = src.read(1).astype(np.uint8)

        axes[2].imshow(rgb)
        pred_rgba = np.zeros((*pred_mask.shape, 4))
        pred_rgba[pred_mask == 1] = FLOOD_COLOR
        axes[2].imshow(pred_rgba, interpolation="nearest")

        # Confusion: TP=blue, FP=red, FN=orange
        tp = (pred_mask == 1) & (gt_mask == 1)
        fp = (pred_mask == 1) & (gt_mask == 0)
        fn = (pred_mask == 0) & (gt_mask == 1)
        conf_rgba = np.zeros((*pred_mask.shape, 4))
        conf_rgba[tp] = (0.1, 0.5, 1.0, 0.7)
        conf_rgba[fp] = (1.0, 0.1, 0.1, 0.7)
        conf_rgba[fn] = (1.0, 0.6, 0.0, 0.7)
        axes[2].imshow(conf_rgba, interpolation="nearest")

        legend_els = [
            mpatches.Patch(color=(0.1, 0.5, 1.0), label="True Positive"),
            mpatches.Patch(color=(1.0, 0.1, 0.1), label="False Positive"),
            mpatches.Patch(color=(1.0, 0.6, 0.0), label="False Negative"),
        ]
        axes[2].legend(handles=legend_els, loc="lower right", fontsize=8)
        axes[2].set_title("Prediction (TP/FP/FN)")
        axes[2].axis("off")

    plt.tight_layout()
    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_file}")
    return fig


# ─── Training curves ──────────────────────────────────────────────────────────

def plot_training_curves(log_file: Path,
                         out_file: Path | None = None) -> plt.Figure:
    """
    Read a simple CSV training log and plot loss + IoU curves.
    Expected columns: epoch, train_loss, val_loss, train_iou, val_iou
    """
    import csv
    rows = []
    with open(log_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    epochs     = [r["epoch"]     for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    val_loss   = [r["val_loss"]   for r in rows]
    train_iou  = [r["train_iou"]  for r in rows]
    val_iou    = [r["val_iou"]    for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train", color="steelblue")
    axes[0].plot(epochs, val_loss,   label="Val",   color="tomato",
                 linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_iou, label="Train", color="steelblue")
    axes[1].plot(epochs, val_iou,   label="Val",   color="tomato",
                 linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU Curves"); axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
    return fig


# ─── Flood-area bar chart ─────────────────────────────────────────────────────

def plot_flood_area_comparison(event_names: list,
                               gt_pct: list, pred_pct: list,
                               out_file: Path | None = None) -> plt.Figure:
    """Bar chart comparing ground-truth vs predicted flooded area (%) per event."""
    x = np.arange(len(event_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(event_names) * 0.9), 5))
    ax.bar(x - w/2, gt_pct,   w, label="Ground Truth", color="steelblue")
    ax.bar(x + w/2, pred_pct, w, label="Predicted",    color="tomato",
           alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(event_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Flooded area (%)")
    ax.set_title("Flooded Area: Ground Truth vs Prediction")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if out_file:
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
    return fig
