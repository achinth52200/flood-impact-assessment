"""
Startup helper for cloud deployments.
Called automatically when the app starts in an environment without a checkpoint.

Priority:
  1. Use existing checkpoint in models/checkpoints/best_model.pth
  2. Download from Hugging Face Hub  (set HF_REPO env var)
  3. Train on synthetic data as a last resort (slow, demo quality)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ensure_model():
    from config import BEST_CKPT, MODEL_DIR, PROC_DIR

    if BEST_CKPT.exists():
        print(f"Checkpoint found: {BEST_CKPT}")
        return True

    # ── Option 1: Download from HF Hub ────────────────────────────────────────
    hf_repo = os.environ.get("HF_REPO")        # e.g. "yourname/flood-model"
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            print(f"Downloading model from HF Hub: {hf_repo} …")
            path = hf_hub_download(
                repo_id=hf_repo,
                filename="best_model.pth",
                local_dir=str(MODEL_DIR),
            )
            print(f"Downloaded → {path}")
            return True
        except Exception as e:
            print(f"HF Hub download failed: {e}")

    # ── Option 2: Quick synthetic training ────────────────────────────────────
    print("No checkpoint found. Training on synthetic data (demo mode) …")
    print("Set HF_REPO env var to load a pretrained checkpoint instead.\n")

    # Generate tiles
    from generate_sample_data import main as gen_main
    import argparse
    sys.argv = ["generate_sample_data.py", "--n_tiles", "80"]
    gen_main()

    # Train for a few epochs
    sys.argv = ["train.py", "--epochs", "10", "--batch_size", "4", "--workers", "0"]
    from train import main as train_main
    train_main()
    return BEST_CKPT.exists()


if __name__ == "__main__":
    ok = ensure_model()
    sys.exit(0 if ok else 1)
