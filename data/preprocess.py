"""
Preprocessing pipeline for the Sen1Floods11 dataset.

Steps:
  1. Read paired Sentinel-1 (.tif 2-band) + Sentinel-2 (.tif 13-band) + mask
  2. Select S2 bands [B2,B3,B4,B8,B11] → 5 channels
  3. Stack S1 (2ch) + S2 (5ch) → 7-channel image
  4. Reproject everything to EPSG:4326 at 10 m
  5. Tile into 256×256 patches (skip >90% nodata)
  6. Save processed patches to data/processed/{images,masks}/

Usage:
    python data/preprocess.py
    python data/preprocess.py --raw_dir data/raw --out_dir data/processed
"""
import argparse
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, PROC_DIR, TILE_SIZE, OVERLAP, S2_BAND_IDX, NUM_CHANNELS


DST_CRS = "EPSG:4326"
TARGET_RES = 10.0   # metres — resample everything to 10 m


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def read_tif(path: Path, band_idx: list[int] | None = None) -> tuple:
    """
    Read a GeoTIFF.  Returns (array CHW float32, profile).
    band_idx: 0-based list of bands to keep (None → keep all).
    """
    with rasterio.open(path) as src:
        if band_idx is None:
            arr = src.read().astype(np.float32)
        else:
            arr = src.read([i + 1 for i in band_idx]).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def reproject_array(arr: np.ndarray, src_profile: dict,
                    dst_crs: str = DST_CRS) -> tuple:
    """Reproject a CHW array to dst_crs at a fixed pixel size."""
    C, H, W = arr.shape
    transform, width, height = calculate_default_transform(
        src_profile["crs"], dst_crs,
        src_profile["width"], src_profile["height"],
        *rasterio.transform.array_bounds(
            src_profile["height"], src_profile["width"],
            src_profile["transform"]
        ),
    )
    dst_profile = src_profile.copy()
    dst_profile.update(crs=dst_crs, transform=transform,
                       width=width, height=height)

    dst = np.zeros((C, height, width), dtype=np.float32)
    for c in range(C):
        reproject(
            source=arr[c], destination=dst[c],
            src_transform=src_profile["transform"],
            src_crs=src_profile["crs"],
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return dst, dst_profile


def save_tif(arr: np.ndarray, profile: dict, path: Path):
    """Save a CHW array as a GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    C, H, W = arr.shape
    profile = profile.copy()
    profile.update(count=C, height=H, width=W,
                   dtype="float32", driver="GTiff",
                   compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def save_mask(arr: np.ndarray, profile: dict, path: Path):
    """Save a HW uint8 array as a single-band GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    H, W = arr.shape
    profile = profile.copy()
    profile.update(count=1, height=H, width=W,
                   dtype="uint8", driver="GTiff", compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(np.uint8), 1)


# ─── Normalisation ────────────────────────────────────────────────────────────

def percentile_clip(arr: np.ndarray,
                    lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-channel percentile normalisation to [0, 1]."""
    out = np.empty_like(arr)
    for c in range(arr.shape[0]):
        vmin, vmax = np.percentile(arr[c], [lo, hi])
        rng = vmax - vmin if vmax != vmin else 1.0
        out[c] = np.clip((arr[c] - vmin) / rng, 0.0, 1.0)
    return out


def sar_preprocess(arr: np.ndarray, db_scale: bool = True) -> np.ndarray:
    """
    Sentinel-1 SAR backscatter preprocessing:
      1. Convert linear scale → dB  (10 * log10)
      2. Clip to [-30, 0] dB range
      3. Normalise to [0, 1]
    """
    arr = np.nan_to_num(arr, nan=-30.0)
    # Sen1Floods11 S1Hand images are already in dB (mostly negative values);
    # only convert when the input is in linear power scale.
    if db_scale and np.nanmin(arr) >= 0:
        arr = np.where(arr > 0, arr, 1e-10)   # avoid log(0)
        arr = 10.0 * np.log10(arr)
    arr = np.clip(arr, -30.0, 0.0)
    arr = (arr + 30.0) / 30.0             # → [0, 1]
    return arr


# ─── Tiling ───────────────────────────────────────────────────────────────────

def tile_pair(img: np.ndarray, mask: np.ndarray, profile: dict,
              out_img_dir: Path, out_mask_dir: Path, event_id: str,
              tile_size: int = TILE_SIZE, overlap: int = OVERLAP) -> int:
    """Slice a (C,H,W) image and (H,W) mask into fixed-size patches."""
    C, H, W = img.shape
    step = tile_size - overlap
    tid  = 0

    for y in range(0, max(H - tile_size + 1, 1), step):
        for x in range(0, max(W - tile_size + 1, 1), step):
            y2 = min(y + tile_size, H)
            x2 = min(x + tile_size, W)
            ph, pw = y2 - y, x2 - x

            img_tile  = img[:, y:y2, x:x2]
            mask_tile = mask[y:y2, x:x2]

            # Skip tiles that are mostly nodata or have zero valid mask pixels
            nodata_frac = (img_tile == 0).all(axis=0).mean()
            if nodata_frac > 0.9:
                continue

            # Pad to tile_size if near the edge
            if ph < tile_size or pw < tile_size:
                img_p  = np.zeros((C, tile_size, tile_size), dtype=np.float32)
                mask_p = np.zeros((tile_size, tile_size), dtype=np.uint8)
                img_p[:, :ph, :pw]  = img_tile
                mask_p[:ph, :pw]    = mask_tile
                img_tile, mask_tile = img_p, mask_p

            # Adjust geotransform
            win_transform = rasterio.transform.xy(
                profile["transform"], y, x)
            tile_profile = profile.copy()
            tile_profile["transform"] = rasterio.transform.from_origin(
                win_transform[0], win_transform[1],
                profile["transform"].a,
                -profile["transform"].e,
            )

            fname = f"{event_id}_{tid:05d}"
            save_tif(img_tile,    tile_profile, out_img_dir  / f"{fname}.tif")
            save_mask(mask_tile,  tile_profile, out_mask_dir / f"{fname}_mask.tif")
            tid += 1

    return tid


# ─── Per-event processing ─────────────────────────────────────────────────────

def process_event(event_id: str, raw_dir: Path, out_dir: Path) -> int:
    """
    Find the S1, S2, and label files for one event and produce tiles.
    Returns the number of tiles generated.
    """
    s1_dir    = raw_dir / "S1"
    s2_dir    = raw_dir / "S2"
    mask_dir  = raw_dir / "labels"
    img_out   = out_dir / "images"
    mask_out  = out_dir / "masks"

    # Locate files (Sen1Floods11 naming: <Country>_<id>_S1Hand.tif etc.)
    s1_path   = next(s1_dir.glob(f"{event_id}*.tif"),   None)
    s2_path   = next(s2_dir.glob(f"{event_id}*.tif"),   None)
    mask_path = next(mask_dir.glob(f"{event_id}*.tif"), None)

    if not all([s1_path, s2_path, mask_path]):
        missing = [n for n, p in [("S1", s1_path), ("S2", s2_path),
                                   ("mask", mask_path)] if p is None]
        print(f"  ⚠  {event_id}: missing {missing}, skipping")
        return 0

    # Read
    s1_arr,   s1_prof  = read_tif(s1_path)                # (2, H, W)
    s2_arr,   s2_prof  = read_tif(s2_path, S2_BAND_IDX)   # (5, H, W)
    mask_arr, mask_prof= read_tif(mask_path)               # (1, H, W)
    mask_arr = (mask_arr[0] > 0).astype(np.uint8)         # → (H, W) binary

    # Reproject to common CRS
    if s1_prof.get("crs") is None or str(s1_prof["crs"]) != DST_CRS:
        s1_arr, s1_prof = reproject_array(s1_arr, s1_prof)
    if s2_prof.get("crs") is None or str(s2_prof["crs"]) != DST_CRS:
        s2_arr, s2_prof = reproject_array(s2_arr, s2_prof)

    # Resample S2 to match S1 spatial extent (S1 is already at 10 m)
    if s2_arr.shape[1:] != s1_arr.shape[1:]:
        import cv2
        H, W = s1_arr.shape[1], s1_arr.shape[2]
        s2_arr = np.stack([
            cv2.resize(s2_arr[c], (W, H), interpolation=cv2.INTER_LINEAR)
            for c in range(s2_arr.shape[0])
        ])
        # Also resize mask
        mask_arr = cv2.resize(mask_arr, (W, H), interpolation=cv2.INTER_NEAREST)

    # Preprocess
    s1_arr = sar_preprocess(s1_arr)
    s2_arr = percentile_clip(s2_arr)

    # Stack S1 + S2 → (7, H, W)
    img = np.concatenate([s1_arr, s2_arr], axis=0)

    n_tiles = tile_pair(img, mask_arr, s1_prof,
                        img_out, mask_out, event_id)
    return n_tiles


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Sen1Floods11 Sentinel-1/2 imagery into tiles"
    )
    parser.add_argument("--raw_dir", default=str(RAW_DIR))
    parser.add_argument("--out_dir", default=str(PROC_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # Discover events from the S1 directory
    s1_dir = raw_dir / "S1"
    if not s1_dir.exists():
        print(f"S1 directory not found: {s1_dir}")
        print("Run  python data/download_data.py  first.")
        return

    events = sorted(p.stem.replace("_S1Hand", "")
                    for p in s1_dir.glob("*.tif"))
    print(f"Found {len(events)} flood events\n")

    total = 0
    for ev in tqdm(events, desc="Events"):
        n = process_event(ev, raw_dir, out_dir)
        total += n
        tqdm.write(f"  {ev}: {n} tiles")

    print(f"\nDone. {total} tiles saved to {out_dir}")
    print("\nNext step:")
    print("  python train.py")


if __name__ == "__main__":
    main()
