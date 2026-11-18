"""
Building impact estimation from 7-channel satellite tile.

Uses spectral indices derived from S2 bands:
  NDBI  = (SWIR - NIR) / (SWIR + NIR)   -> built-up land (positive = urban)
  NDWI  = (Green - NIR) / (Green + NIR) -> open water (cross-check)
  NDVI  = (NIR - Red) / (NIR + Red)     -> vegetation (negative = not veg = potential urban)

Channel layout expected: [S1-VV, S1-VH, B2-Blue, B3-Green, B4-Red, B8-NIR, B11-SWIR]
                           ch0    ch1    ch2      ch3       ch4    ch5     ch6
"""
import numpy as np

# Rough pixel-to-building conversion for 10 m/pixel Sentinel data.
# Average building footprint ~150 m² → ~1.5 pixels at 10 m resolution.
# We use 2 pixels per building as a conservative estimate.
PIXELS_PER_BUILDING = 2
PIXEL_AREA_M2 = 100          # 10 m × 10 m
AVG_BUILDING_SIZE_M2 = 150   # average urban building footprint


def compute_indices(arr: np.ndarray) -> dict:
    """
    arr: (C, H, W) float32, channels normalised to [0, 1].
    Returns dict of 2-D index maps.
    """
    eps = 1e-6
    nir  = arr[5]
    swir = arr[6]
    red  = arr[4]
    grn  = arr[3]

    ndbi = (swir - nir)  / (swir + nir  + eps)
    ndvi = (nir  - red)  / (nir  + red  + eps)
    ndwi = (grn  - nir)  / (grn  + nir  + eps)

    return {"NDBI": ndbi, "NDVI": ndvi, "NDWI": ndwi}


def detect_buildup(arr: np.ndarray, ndbi_thresh: float = 0.05) -> np.ndarray:
    """
    Returns a binary (H, W) uint8 mask: 1 = built-up pixel.
    Built-up pixels satisfy:
      NDBI > ndbi_thresh  AND  NDVI < 0.2  (not vegetation)
    """
    idx = compute_indices(arr)
    buildup = ((idx["NDBI"] > ndbi_thresh) & (idx["NDVI"] < 0.2)).astype(np.uint8)
    return buildup


def estimate_buildings(buildup_mask: np.ndarray) -> int:
    """Convert pixel count to estimated building count."""
    return max(1, int(buildup_mask.sum() / PIXELS_PER_BUILDING))


def building_impact(arr: np.ndarray,
                    flood_mask: np.ndarray,
                    ndbi_thresh: float = 0.05) -> dict:
    """
    Full building impact analysis.

    Parameters
    ----------
    arr        : (C, H, W) float32 tile
    flood_mask : (H, W) binary uint8 flood prediction
    ndbi_thresh: NDBI threshold for built-up detection

    Returns
    -------
    dict with keys:
        buildup_mask     - (H, W) uint8 built-up pixels
        affected_mask    - (H, W) uint8 flooded AND built-up pixels
        total_buildings  - estimated total buildings in scene
        affected_buildings - estimated buildings under flood water
        affected_pct     - percentage of buildings affected
        flooded_area_km2 - flooded area in km²
        affected_area_km2- affected built-up area in km²
        indices          - dict of NDBI/NDVI/NDWI maps
    """
    H, W = flood_mask.shape
    total_pixels = H * W

    idx          = compute_indices(arr)
    buildup_mask = detect_buildup(arr, ndbi_thresh)
    affected     = ((buildup_mask == 1) & (flood_mask == 1)).astype(np.uint8)

    total_bld    = estimate_buildings(buildup_mask)
    affected_bld = estimate_buildings(affected)

    affected_pct = (affected_bld / max(total_bld, 1)) * 100

    flooded_area_km2  = float(flood_mask.sum()  * PIXEL_AREA_M2 / 1e6)
    affected_area_km2 = float(affected.sum()    * PIXEL_AREA_M2 / 1e6)

    return {
        "buildup_mask":       buildup_mask,
        "affected_mask":      affected,
        "total_buildings":    total_bld,
        "affected_buildings": affected_bld,
        "affected_pct":       affected_pct,
        "flooded_area_km2":   flooded_area_km2,
        "affected_area_km2":  affected_area_km2,
        "indices":            idx,
    }
