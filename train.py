"""
Training script for the flood impact segmentation model.

Usage:
    python train.py
    python train.py --epochs 30 --batch_size 4 --lr 1e-4
    python train.py --resume models/checkpoints/best_model.pth
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import (DEVICE, BATCH_SIZE, NUM_EPOCHS, LR, WEIGHT_DECAY,
                    LR_MIN, WARMUP_EPOCHS, EARLY_STOP, POS_WEIGHT,
                    DICE_WEIGHT, BEST_CKPT, SAVE_EVERY, LOG_DIR,
                    MODEL_DIR, IOU_THRESHOLD)
from data.dataset import FloodDataset, get_splits
from models.unet import build_model, CombinedLoss
from utils.metrics import compute_all, MetricTracker


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("Using CPU")
    return d


def save_checkpoint(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_iou    = ckpt.get("best_iou", 0.0)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    print(f"Resumed from epoch {ckpt['epoch']}  (best IoU {best_iou:.4f})")
    return start_epoch, best_iou


# ─── Warmup cosine LR ─────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min_ratio):
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
            import math
            return lr_min_ratio + 0.5 * (1 - lr_min_ratio) * \
                   (1 + math.cos(math.pi * progress))
        super().__init__(optimizer, lr_lambda)


# ─── Single epoch ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device,
              is_train: bool) -> dict:
    model.train() if is_train else model.eval()
    tracker = MetricTracker()
    total_loss = 0.0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, desc="train" if is_train else "val",
                                leave=False):
            imgs  = imgs.to(device)   # (B, 7, H, W)
            masks = masks.to(device)  # (B, 1, H, W)

            logits = model(imgs)      # (B, 1, H, W)
            loss   = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Metrics on CPU to save GPU memory
            m = compute_all(logits.detach().cpu(),
                            masks.detach().cpu(),
                            threshold=IOU_THRESHOLD)
            m["loss"] = loss.item()
            tracker.update(m, n=imgs.size(0))
            total_loss += loss.item() * imgs.size(0)

    return tracker.means()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--resume",      type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--workers",     type=int,   default=4)
    args = parser.parse_args()

    device = get_device(DEVICE)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ids, val_ids, test_ids = get_splits()

    if not train_ids:
        print("\nNo processed tiles found.")
        print("Run these first:")
        print("  python data/download_data.py --subset 10")
        print("  python data/preprocess.py")
        print("\nOr generate synthetic test data:")
        print("  python generate_sample_data.py")
        sys.exit(1)

    print(f"Tiles — train: {len(train_ids)}, val: {len(val_ids)}, "
          f"test: {len(test_ids)}")

    train_ds = FloodDataset(train_ids, split="train")
    val_ds   = FloodDataset(val_ids,   split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.workers,
                              pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model().to(device)

    criterion = CombinedLoss(dice_w=DICE_WEIGHT, pos_weight=POS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(
        optimizer, WARMUP_EPOCHS, args.epochs,
        lr_min_ratio=LR_MIN / args.lr)

    start_epoch, best_iou = 0, 0.0
    no_improve = 0

    if args.resume:
        start_epoch, best_iou = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer   = SummaryWriter(log_dir=str(LOG_DIR / run_name))

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs …\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_m = run_epoch(model, train_loader, criterion, optimizer,
                            device, is_train=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer,
                            device, is_train=False)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch+1:3d}/{args.epochs}  "
            f"Loss {train_m['loss']:.4f}/{val_m['loss']:.4f}  "
            f"IoU {train_m['iou']:.4f}/{val_m['iou']:.4f}  "
            f"F1 {val_m['f1']:.4f}  "
            f"LR {lr_now:.2e}  "
            f"({elapsed:.0f}s)"
        )

        # TensorBoard
        for k in train_m:
            writer.add_scalars(k, {"train": train_m[k], "val": val_m[k]}, epoch)
        writer.add_scalar("lr", lr_now, epoch)

        # Checkpoint
        state = dict(epoch=epoch, model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),
                     best_iou=best_iou, val_metrics=val_m)

        if val_m["iou"] > best_iou:
            best_iou   = val_m["iou"]
            no_improve = 0
            save_checkpoint(state, BEST_CKPT)
            print(f"  ✓ New best IoU: {best_iou:.4f} — checkpoint saved")
        else:
            no_improve += 1

        if (epoch + 1) % SAVE_EVERY == 0:
            save_checkpoint(state, MODEL_DIR / f"epoch_{epoch+1:03d}.pth")

        if no_improve >= EARLY_STOP:
            print(f"\nEarly stopping after {EARLY_STOP} epochs without "
                  f"validation improvement.")
            break

    writer.close()
    print(f"\nTraining complete. Best val IoU: {best_iou:.4f}")
    print(f"Best checkpoint: {BEST_CKPT}")
    print("\nNext step:")
    print("  python predict.py --checkpoint", BEST_CKPT)


if __name__ == "__main__":
    main()
