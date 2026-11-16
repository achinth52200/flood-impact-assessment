"""
Inference script — tifffile backend (no rasterio/DLL).

Usage:
    python predict.py --test
    python predict.py --input data/processed/images/India_synth_00001.tif
"""
import argparse, csv, sys
import numpy as np
import torch
import tifffile
from pathlib import Path
from tqdm import tqdm

from config import DEVICE, BEST_CKPT, PROC_DIR, PRED_DIR, NUM_CHANNELS, IOU_THRESHOLD
from models.unet import build_model
from utils.metrics import compute_all, MetricTracker


def load_model(ckpt_path, device):
    model = build_model().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {ckpt_path}  (val IoU {ckpt.get('best_iou',0):.4f})")
    return model


def read_tile(path):
    arr = tifffile.imread(str(path)).astype(np.float32)
    if arr.ndim == 2: arr = arr[np.newaxis]
    if arr.shape[0] < arr.shape[-1]: arr = np.transpose(arr,(1,2,0))
    # (H,W,C) → normalise if needed
    for c in range(arr.shape[-1]):
        lo,hi = arr[:,:,c].min(), arr[:,:,c].max()
        if hi > 1.0:
            arr[:,:,c] = np.clip((arr[:,:,c]-lo)/max(hi-lo,1e-6),0,1)
    return np.transpose(arr,(2,0,1))   # → (C,H,W)


@torch.no_grad()
def predict_tile(model, arr, device, threshold=IOU_THRESHOLD):
    t = torch.from_numpy(arr).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    return (prob >= threshold).astype(np.uint8), prob


def evaluate_test(model, device):
    from data.dataset import get_splits
    from config import BATCH_SIZE
    from torch.utils.data import DataLoader
    from data.dataset import FloodDataset

    _, _, test_ids = get_splits()
    if not test_ids:
        print("No test tiles. Run generate_sample_data.py first.")
        return

    loader  = DataLoader(FloodDataset(test_ids, "test"),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    tracker = MetricTracker()
    for imgs, masks in tqdm(loader, desc="Test"):
        m = compute_all(model(imgs.to(device)).detach().cpu(), masks.cpu())
        tracker.update(m, imgs.size(0))

    print("\n── Test Metrics ─────────────────────")
    for k,v in tracker.means().items():
        print(f"  {k:12s}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(BEST_CKPT))
    parser.add_argument("--input",  default=None)
    parser.add_argument("--out_dir",default=str(PRED_DIR))
    parser.add_argument("--test",   action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"No checkpoint at {ckpt}. Run python train.py first.")
        return

    model = load_model(ckpt, device)
    if args.test:
        evaluate_test(model, device)
        return

    if args.input:
        p = Path(args.input)
        paths = [p] if p.is_file() else sorted(p.glob("*.tif"))
        out   = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for tp in tqdm(paths, desc="Predicting"):
            arr  = read_tile(tp)
            mask, prob = predict_tile(model, arr[:NUM_CHANNELS], device)
            tifffile.imwrite(str(out / (tp.stem+"_flood.tif")), mask)
            tifffile.imwrite(str(out / (tp.stem+"_prob.tif")),  prob.astype(np.float32))
        print(f"Saved predictions → {out}")

if __name__ == "__main__":
    main()
