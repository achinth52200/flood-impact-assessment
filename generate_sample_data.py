"""
Generate synthetic Sentinel-1 + Sentinel-2 tiles for testing.
Uses tifffile (pure Python) instead of rasterio.

Usage:
    python generate_sample_data.py --n_tiles 100
"""
import argparse
import numpy as np
import tifffile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROC_DIR, TILE_SIZE

RNG = np.random.default_rng(42)


def _noise(h, w, seed=0):
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    n = np.zeros((h, w))
    for freq in [3, 7, 15, 31]:
        amp = 1.0 / freq
        ph = RNG.uniform(0, 2*np.pi, 2)
        n += amp * np.sin(2*np.pi*freq*xx + ph[0]) * np.cos(2*np.pi*freq*yy + ph[1])
    return (n - n.min()) / (n.max() - n.min() + 1e-9)


def make_tile(h, w, flood_frac=0.15):
    base = _noise(h, w)
    thresh = np.percentile(base, (1 - flood_frac) * 100)
    mask = (base > thresh).astype(np.uint8)

    land = _noise(h, w, seed=1)
    veg  = _noise(h, w, seed=2)

    # Sentinel-1: water is dark in SAR
    s1_vv = np.where(mask, RNG.uniform(0.05,0.15), land*0.4+0.3).astype(np.float32)
    s1_vh = np.where(mask, RNG.uniform(0.03,0.10), land*0.3+0.15).astype(np.float32)
    s1_vv = np.clip(s1_vv + RNG.normal(0, 0.02, (h,w)), 0, 1).astype(np.float32)
    s1_vh = np.clip(s1_vh + RNG.normal(0, 0.02, (h,w)), 0, 1).astype(np.float32)

    # Sentinel-2: water = high blue, low NIR/SWIR
    def band(land_v, water_v):
        b = np.where(mask, water_v, land_v*land+0.1)
        return np.clip(b + RNG.normal(0,0.03,(h,w)), 0, 1).astype(np.float32)

    s2_b2  = band(0.3,  0.6)
    s2_b3  = band(0.35, 0.55)
    s2_b4  = band(0.25, 0.1)
    s2_b8  = band(0.5+0.3*veg, 0.05)
    s2_b11 = band(0.4,  0.08)

    img = np.stack([s1_vv, s1_vh, s2_b2, s2_b3, s2_b4, s2_b8, s2_b11])  # (7,H,W)
    return img, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_tiles", type=int, default=100)
    parser.add_argument("--size",    type=int, default=TILE_SIZE)
    parser.add_argument("--out_dir", default=str(PROC_DIR))
    args = parser.parse_args()

    img_dir  = Path(args.out_dir) / "images"
    mask_dir = Path(args.out_dir) / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    flood_fracs = RNG.uniform(0.03, 0.40, args.n_tiles)
    regions = ["India","Bangladesh","Myanmar","Vietnam","Ghana","Spain"]

    print(f"Generating {args.n_tiles} synthetic {args.size}×{args.size} tiles …")
    for i in range(args.n_tiles):
        region   = RNG.choice(regions)
        event_id = f"{region}_synth_{i:05d}"
        img, mask = make_tile(args.size, args.size, flood_fracs[i])

        # Save as multi-page TIFF: (C,H,W) for image, (H,W) for mask
        tifffile.imwrite(str(img_dir  / f"{event_id}.tif"),  img,  photometric='minisblack')
        tifffile.imwrite(str(mask_dir / f"{event_id}_mask.tif"), mask)

        if (i+1) % 20 == 0 or i == args.n_tiles-1:
            print(f"  {i+1}/{args.n_tiles} tiles done")

    print(f"\nSaved to {Path(args.out_dir)}")
    print("Next: python train.py")

if __name__ == "__main__":
    main()
