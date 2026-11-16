"""
Download the Sen1Floods11 dataset from Google Cloud Storage (public, no auth).

Sen1Floods11 is a globally distributed labeled flood dataset containing:
  • Sentinel-1 SAR (VV + VH bands, 10 m resolution)
  • Sentinel-2 optical (13 bands, 10–60 m)
  • Hand-labeled binary flood masks

Paper: https://arxiv.org/abs/2107.08369
Bucket: gs://sen1floods11/

Usage:
    # Download a quick subset (~500 MB) to get started:
    python data/download_data.py --subset 10

    # Download everything (~8 GB):
    python data/download_data.py --all

    # List available flood events without downloading:
    python data/download_data.py --list
"""
import argparse
import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, GCS_BUCKET, GCS_PREFIX, S1_SUBDIR, S2_SUBDIR, LABEL_SUBDIR

# Public GCS JSON API endpoint (no credentials required)
GCS_API = "https://storage.googleapis.com/storage/v1"
PUBLIC_URL = "https://storage.googleapis.com"

# ─── 46 flood events in the Sen1Floods11 hand-labelled split ──────────────────
FLOOD_EVENTS = [
    "India_900498_S2Hand", "India_900498_S1Hand",
    "Spain_7370579_S2Hand", "Spain_7370579_S1Hand",
    "USA_430764_S2Hand", "USA_430764_S1Hand",
    "Ghana_541920_S2Hand", "Ghana_541920_S1Hand",
    "Myanmar_501974_S2Hand", "Myanmar_501974_S1Hand",
    "Bangladesh_243573_S2Hand", "Bangladesh_243573_S1Hand",
    "Somalia_490779_S2Hand", "Somalia_490779_S1Hand",
    "Cambodia_564862_S2Hand", "Cambodia_564862_S1Hand",
    "China_395_S2Hand", "China_395_S1Hand",
    "Kenya_604327_S2Hand", "Kenya_604327_S1Hand",
    "Vietnam_521569_S2Hand", "Vietnam_521569_S1Hand",
    "Pakistan_72525_S2Hand", "Pakistan_72525_S1Hand",
    "Iraq_236629_S2Hand", "Iraq_236629_S1Hand",
    "Indonesia_207012_S2Hand", "Indonesia_207012_S1Hand",
]

# Split CSV shipped with the dataset (defines official train/val/test split)
SPLIT_CSV_URL = (
    "https://raw.githubusercontent.com/cloudtostreet/Sen1Floods11/"
    "master/splits/flood_handlabeled.csv"
)


def list_bucket_objects(prefix: str) -> list[dict]:
    """Return all objects under a GCS bucket prefix."""
    url = f"{GCS_API}/b/{GCS_BUCKET}/o"
    objects, page_token = [], None
    while True:
        params = {"prefix": prefix, "maxResults": 1000}
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        objects.extend(data.get("items", []))
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return objects


def download_file(gcs_name: str, local_path: Path, show_bar: bool = True):
    """Download a single GCS object to local_path."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return  # already downloaded

    url = f"{PUBLIC_URL}/{GCS_BUCKET}/{gcs_name}"
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(local_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True,
        desc=local_path.name, leave=False, disable=not show_bar
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_split_csv():
    """Download the official train/val/test split CSV."""
    dest = RAW_DIR / "flood_handlabeled.csv"
    if dest.exists():
        return dest
    print("Downloading split CSV…")
    resp = requests.get(SPLIT_CSV_URL, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    print(f"  Saved → {dest}")
    return dest


def build_file_list() -> list[dict]:
    """
    List all hand-labelled Sentinel-1, Sentinel-2 and mask files in the bucket.
    Returns a list of dicts: {name, local_path, size_bytes}.
    """
    print(f"Listing objects in gs://{GCS_BUCKET}/{GCS_PREFIX} …")
    objects = list_bucket_objects(GCS_PREFIX)
    file_list = []
    for obj in objects:
        name = obj["name"]                       # e.g. v1.1/data/.../S1Hand/X.tif
        fname = Path(name).name                  # X.tif
        # Determine local sub-directory
        if f"/{S1_SUBDIR}/" in name:
            subdir = "S1"
        elif f"/{S2_SUBDIR}/" in name:
            subdir = "S2"
        elif f"/{LABEL_SUBDIR}/" in name:
            subdir = "labels"
        else:
            continue
        local_path = RAW_DIR / subdir / fname
        file_list.append({
            "name": name,
            "local_path": local_path,
            "size_bytes": int(obj.get("size", 0)),
        })
    return file_list


def main():
    parser = argparse.ArgumentParser(
        description="Download Sen1Floods11 satellite imagery dataset"
    )
    parser.add_argument("--all",    action="store_true",
                        help="Download the full dataset (~8 GB)")
    parser.add_argument("--subset", type=int, default=5,
                        help="Number of flood events to download (default: 5)")
    parser.add_argument("--list",   action="store_true",
                        help="List available files and exit")
    args = parser.parse_args()

    # Always grab the split CSV
    download_split_csv()

    file_list = build_file_list()

    if args.list:
        print(f"\nFound {len(file_list)} files:")
        for f in file_list[:40]:
            mb = f["size_bytes"] / 1e6
            print(f"  [{f['local_path'].parent.name:7s}] {f['local_path'].name}  {mb:.1f} MB")
        if len(file_list) > 40:
            print(f"  … and {len(file_list) - 40} more")
        return

    if not args.all:
        # Group by flood event name and pick the first N events
        events: dict[str, list] = {}
        for f in file_list:
            # Event ID is the filename stem up to the sensor suffix
            stem = Path(f["local_path"]).stem   # e.g. India_900498_S2Hand
            key  = stem.rsplit("_", 1)[0]       # e.g. India_900498
            events.setdefault(key, []).append(f)

        selected_keys = sorted(events.keys())[:args.subset]
        file_list = [f for k in selected_keys for f in events[k]]
        total_mb = sum(f["size_bytes"] for f in file_list) / 1e6
        print(f"\nDownloading {args.subset} flood events "
              f"({len(file_list)} files, ~{total_mb:.0f} MB) …")
    else:
        total_mb = sum(f["size_bytes"] for f in file_list) / 1e6
        print(f"\nDownloading full dataset "
              f"({len(file_list)} files, ~{total_mb:.0f} MB) …")

    for i, f in enumerate(file_list, 1):
        print(f"[{i:3d}/{len(file_list)}] {f['local_path'].name}")
        try:
            download_file(f["name"], f["local_path"])
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDownload complete. Files saved to:\n  {RAW_DIR}")
    print("\nNext step:")
    print("  python data/preprocess.py")


if __name__ == "__main__":
    main()
