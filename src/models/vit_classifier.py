import torch
import torch.nn as nn

from .vit_encoder import ViTEncoder


class ViTClassifier(nn.Module):
    """
    ViT encoder with a linear classification head.

    Used for two purposes:
    1. Supervised baseline — encoder trained from scratch end-to-end.
    2. Linear probe — encoder weights loaded from MAE checkpoint and frozen;
       only the head is trained.

    Args:
        encoder: ViTEncoder backbone
        num_classes: number of output classes (100 for ImageNet100)
        dropout: dropout applied before the classification head
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        num_classes: int = 100,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.patch_embed.proj.out_channels, num_classes)

        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (for linear probing)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters (e.g., for fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W)
        Returns: logits (B, num_classes)
        """
        features = self.encoder(x)          # (B, N+1, D)
        cls_token = features[:, 0]          # (B, D) — CLS token
        cls_token = self.dropout(cls_token)
        return self.head(cls_token)


def build_vit_classifier(
    cfg: dict,
    pretrained_encoder_path: str | None = None,
    device: torch.device | None = None,
) -> "ViTClassifier":
    """
    Build a ViTClassifier from config.

    If pretrained_encoder_path is provided, loads encoder weights from an MAE
    checkpoint and freezes the encoder (linear probe mode).
    Otherwise returns a fresh model for supervised training.
    """
    enc_cfg = cfg["model"]["encoder"]
    encoder = ViTEncoder(
        img_size=cfg["model"].get("img_size", 224),
        patch_size=cfg["model"].get("patch_size", 16),
        in_channels=3,
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
        dropout=enc_cfg.get("dropout", 0.0),
    )
    model = ViTClassifier(
        encoder=encoder,
        num_classes=cfg["model"].get("num_classes", 100),
        dropout=cfg["model"].get("head_dropout", 0.0),
    )

    if pretrained_encoder_path is not None:
        from src.utils.checkpoint import load_encoder_weights
        load_encoder_weights(pretrained_encoder_path, model, device=device)
        model.freeze_encoder()
        print("[Model] Encoder frozen for linear probing.")

    return model
