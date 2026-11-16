"""
Flood segmentation model: U-Net with a pretrained ResNet encoder.

We use segmentation_models_pytorch (smp) which supports:
  • unet, unetplusplus, deeplabv3plus, fpn, pspnet …
  • encoders: resnet18/34/50, efficientnet-b0…b7, mit-b0…b5 …

Because our input has 7 channels (S1 VV+VH + S2 B2/B3/B4/B8/B11) but
ImageNet pretrained weights expect 3, we patch the first conv layer:
  - Copy RGB weights into the first 3 channels.
  - Initialise the remaining 4 channels with the mean of the RGB weights
    (a standard approach that preserves scale and variance).
"""
import sys
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (ARCHITECTURE, ENCODER, ENCODER_WEIGHTS,
                    NUM_CHANNELS, NUM_CLASSES, DROPOUT)


# ─── Channel adapter for N-channel input ──────────────────────────────────────

def adapt_first_conv(model: nn.Module, in_channels: int) -> nn.Module:
    """
    Patch the first convolutional layer of *any* smp model so it accepts
    `in_channels` input channels while reusing ImageNet pretrained weights.
    """
    if in_channels == 3:
        return model

    # smp places the first conv at model.encoder.model.conv1  (ResNet)
    # or model.encoder.features[0][0]  (EfficientNet) etc.
    # Walk the module tree to find the first Conv2d.
    first_conv: nn.Conv2d | None = None
    first_name: str = ""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            first_name = name
            break

    if first_conv is None:
        raise RuntimeError("Could not find first Conv2d in model.")

    old_weight = first_conv.weight.data   # (out, 3, kH, kW)
    out_ch, _, kH, kW = old_weight.shape

    new_conv = nn.Conv2d(
        in_channels, out_ch,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=first_conv.bias is not None,
    )

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_weight
        # Remaining channels: mean of the RGB weights
        mean_w = old_weight.mean(dim=1, keepdim=True)
        for c in range(3, in_channels):
            new_conv.weight[:, c:c+1, :, :] = mean_w

        if first_conv.bias is not None:
            new_conv.bias.data = first_conv.bias.data.clone()

    # Replace the module by walking the name path
    parts  = first_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_conv)

    return model


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(architecture: str = ARCHITECTURE,
                encoder: str = ENCODER,
                encoder_weights: str = ENCODER_WEIGHTS,
                in_channels: int = NUM_CHANNELS,
                num_classes: int = NUM_CLASSES,
                dropout: float = DROPOUT) -> nn.Module:
    """
    Build and return a segmentation model.

    Args:
        architecture    : one of 'unet', 'unetplusplus', 'deeplabv3plus', 'fpn'
        encoder         : backbone, e.g. 'resnet34', 'efficientnet-b2', 'mit-b1'
        encoder_weights : 'imagenet' | None
        in_channels     : number of input channels (7 for S1+S2)
        num_classes     : 1 for binary flood segmentation
    """
    arch_map = {
        "unet":         smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus":smp.DeepLabV3Plus,
        "fpn":          smp.FPN,
        "pspnet":       smp.PSPNet,
        "pan":          smp.PAN,
    }
    if architecture not in arch_map:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                         f"Choose from: {list(arch_map)}")

    model = arch_map[architecture](
        encoder_name    = encoder,
        encoder_weights = encoder_weights,
        in_channels     = 3,            # start with 3 to load ImageNet weights
        classes         = num_classes,
        activation      = None,         # raw logits; we apply sigmoid in loss
    )

    # Adapt first conv for N-channel satellite input
    model = adapt_first_conv(model, in_channels)

    return model


# ─── Loss functions ────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)
        inter = (probs * targets).sum()
        return 1.0 - (2.0 * inter + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """
    Dice + weighted BCE.
    flood class up-weighted with pos_weight to handle class imbalance.
    """
    def __init__(self, dice_w: float = 0.5, pos_weight: float = 5.0):
        super().__init__()
        self.dice_w  = dice_w
        self.dice    = DiceLoss()
        self.bce     = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        d = self.dice(logits, targets)
        b = self.bce(logits, targets)
        return self.dice_w * d + (1 - self.dice_w) * b


# ─── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model()
    model.eval()
    x = torch.randn(2, NUM_CHANNELS, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")          # expected (2, 1, 256, 256)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
