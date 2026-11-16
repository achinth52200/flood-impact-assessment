"""
FloodDataset — PyTorch Dataset using tifffile (pure Python, no DLL dependencies).
7-channel tiles: S1(VV,VH) + S2(B2,B3,B4,B8,B11).
"""
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROC_DIR, RAW_DIR, NUM_CHANNELS, AUG_PROB, SEED


def load_image(path: str) -> np.ndarray:
    """Read multi-band TIFF → (H, W, C) float32."""
    arr = tifffile.imread(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis]           # (1,H,W)
    if arr.shape[0] < arr.shape[-1]:    # already (C,H,W)
        arr = np.transpose(arr, (1,2,0))
    return arr  # (H,W,C)


def load_mask(path: str) -> np.ndarray:
    """Read single-band TIFF mask → (H, W) uint8."""
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr[:,:,0]
    return (arr > 0).astype(np.uint8)


def build_transforms(split: str):
    if split == "train":
        return A.Compose([
            A.RandomRotate90(p=AUG_PROB),
            A.HorizontalFlip(p=AUG_PROB),
            A.VerticalFlip(p=AUG_PROB),
            A.Affine(scale=(0.9,1.1), translate_percent=(-0.05,0.05),
                     rotate=(-30,30), mode=0, p=AUG_PROB),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.4),
            A.GaussNoise(p=0.3),
            ToTensorV2(),
        ])
    return A.Compose([ToTensorV2()])


class FloodDataset(Dataset):
    def __init__(self, tile_ids, split="train", img_dir=None, mask_dir=None):
        self.ids      = tile_ids
        self.img_dir  = Path(img_dir)  if img_dir  else PROC_DIR / "images"
        self.mask_dir = Path(mask_dir) if mask_dir else PROC_DIR / "masks"
        self.transform = build_transforms(split)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        tid  = self.ids[idx]
        img  = load_image(str(self.img_dir  / f"{tid}.tif"))       # (H,W,7)
        mask = load_mask(str(self.mask_dir  / f"{tid}_mask.tif"))  # (H,W)

        # Clip to expected channel count
        if img.shape[-1] > NUM_CHANNELS:
            img = img[:, :, :NUM_CHANNELS]

        aug = self.transform(image=img, mask=mask)
        return aug["image"].float(), aug["mask"].float().unsqueeze(0)


def collect_tile_ids(img_dir=None):
    img_dir = Path(img_dir) if img_dir else PROC_DIR / "images"
    return sorted(p.stem for p in img_dir.glob("*.tif"))


def random_split(all_ids, train_r=0.70, val_r=0.15, seed=SEED):
    rng = np.random.default_rng(seed)
    ids = np.array(all_ids); rng.shuffle(ids)
    n, t, v = len(ids), int(len(ids)*train_r), int(len(ids)*val_r)
    return ids[:t].tolist(), ids[t:t+v].tolist(), ids[t+v:].tolist()


def get_splits(proc_dir=None, csv_path=None):
    all_tiles = collect_tile_ids(proc_dir)
    return random_split(all_tiles)
