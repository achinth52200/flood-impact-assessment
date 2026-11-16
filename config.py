"""
Central configuration for the Flood Impact Assessment project.
Dataset: Sen1Floods11  (Sentinel-1 SAR + Sentinel-2 optical)
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"          # downloaded .tif files land here
PROC_DIR    = DATA_DIR / "processed"   # tiled patches
SAMPLE_DIR  = DATA_DIR / "sample"      # synthetic test data
MODEL_DIR   = BASE_DIR / "models" / "checkpoints"
OUTPUT_DIR  = BASE_DIR / "outputs"
PRED_DIR    = OUTPUT_DIR / "predictions"
METRICS_DIR = OUTPUT_DIR / "metrics"
LOG_DIR     = BASE_DIR / "logs"

for _d in [RAW_DIR, PROC_DIR, SAMPLE_DIR, MODEL_DIR,
           PRED_DIR, METRICS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── Sen1Floods11 Dataset ─────────────────────────────────────────────────────
# Public GCS bucket (no auth required for read)
GCS_BUCKET  = "sen1floods11"
GCS_PREFIX  = "v1.1/data/flood_events/HandLabeled"
# Sub-directories inside the bucket
S1_SUBDIR   = "S1Hand"      # Sentinel-1 SAR images  (VV, VH)
S2_SUBDIR   = "S2Hand"      # Sentinel-2 optical images (13 bands)
LABEL_SUBDIR= "LabelHand"   # Binary flood masks

# Which Sentinel-2 bands to keep (indices 0-based from the 13-band stack)
# B2(blue)=1, B3(green)=2, B4(red)=3, B8(NIR)=7, B11(SWIR1)=10
S2_BAND_IDX = [1, 2, 3, 7, 10]

# ─── Input channels ───────────────────────────────────────────────────────────
# Sentinel-1: VV, VH                        → 2 channels
# Sentinel-2: B2, B3, B4, B8, B11           → 5 channels
# Total                                      → 7 channels
NUM_S1_CHANNELS = 2
NUM_S2_CHANNELS = len(S2_BAND_IDX)
NUM_CHANNELS    = NUM_S1_CHANNELS + NUM_S2_CHANNELS   # 7
NUM_CLASSES     = 1   # binary segmentation (flood / no-flood)

# ─── Tiling ───────────────────────────────────────────────────────────────────
TILE_SIZE   = 256
OVERLAP     = 32

# ─── Train / Val / Test split (applied per flood event) ───────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test = 0.15
SEED        = 42

# ─── Model ────────────────────────────────────────────────────────────────────
ARCHITECTURE     = "unet"       # "unet" | "unetplusplus" | "deeplabv3plus"
ENCODER          = "resnet34"
ENCODER_WEIGHTS  = "imagenet"   # first 3 channels use pretrained weights
DROPOUT          = 0.3

# ─── Training ─────────────────────────────────────────────────────────────────
DEVICE        = "cuda"      # falls back to "cpu" in train.py if unavailable
BATCH_SIZE    = 8
NUM_EPOCHS    = 60
LR            = 3e-4
WEIGHT_DECAY  = 1e-5
SCHEDULER     = "cosine"    # "cosine" | "step" | "plateau"
LR_MIN        = 1e-6
WARMUP_EPOCHS = 5
EARLY_STOP    = 12          # patience epochs

# Flood pixels are ~5-20% of the image → up-weight positive class
POS_WEIGHT    = 5.0         # weight for flood class in BCE loss
DICE_WEIGHT   = 0.5         # combined loss = DICE_WEIGHT*Dice + (1-DICE_WEIGHT)*BCE

# ─── Augmentation ─────────────────────────────────────────────────────────────
AUG_PROB      = 0.5

# ─── Evaluation ───────────────────────────────────────────────────────────────
IOU_THRESHOLD = 0.5
METRICS       = ["iou", "f1", "precision", "recall", "accuracy"]

# ─── Checkpoint ───────────────────────────────────────────────────────────────
BEST_CKPT   = MODEL_DIR / "best_model.pth"
SAVE_EVERY  = 5     # also save a periodic checkpoint every N epochs
