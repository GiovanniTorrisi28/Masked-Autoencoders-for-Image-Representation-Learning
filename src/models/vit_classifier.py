from __future__ import annotations

import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16


def build_vit_classifier(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "vit_b_16":
        return vit_b_16(weights=None, num_classes=num_classes)
    if model_name == "vit_b_32":
        return vit_b_32(weights=None, num_classes=num_classes)
    if model_name == "vit_l_16":
        return vit_l_16(weights=None, num_classes=num_classes)
    raise ValueError(f"Unsupported model_name: {model_name}")
