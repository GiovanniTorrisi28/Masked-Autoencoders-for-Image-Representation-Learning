import torch
import torch.nn as nn

from .mae_decoder import MAEDecoder
from .vit_encoder import ViTEncoder


class MAE(nn.Module):
    """
    Full Masked Autoencoder for self-supervised pre-training.

    Wraps ViTEncoder + MAEDecoder, implements random patch masking,
    patchification/unpatchification, and pixel normalization targets.

    Args:
        encoder: ViTEncoder instance
        decoder: MAEDecoder instance
        mask_ratio: fraction of patches to mask (default 0.75)
        norm_pix_loss: if True, normalize pixel targets per-patch (MAE paper default)
    """

    def __init__(
        self,
        encoder: ViTEncoder,
        decoder: MAEDecoder,
        mask_ratio: float = 0.75,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = encoder.patch_embed.patch_size
        self.num_patches = encoder.patch_embed.num_patches

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: (B, 3, H, W)
        Returns: (B, num_patches, patch_size**2 * 3)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)     # (B, h, w, p, p, C)
        x = x.reshape(B, h * w, p * p * C)  # (B, N, patch_dim)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, num_patches, patch_size**2 * 3)
        Returns: (B, 3, H, W)  — used only for visualization
        """
        p = self.patch_size
        B, N, _ = patches.shape
        h = w = int(N ** 0.5)
        C = 3
        x = patches.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4)     # (B, C, h, p, w, p)
        x = x.reshape(B, C, h * p, w * p)
        return x

    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking via shuffle + slice.

        Args:
            x: patch embeddings (B, N, D) — CLS token NOT included
            mask_ratio: fraction of patches to mask

        Returns:
            visible_tokens: (B, num_visible, D)
            mask: (B, N) — binary, 1=masked 0=visible
            visible_indices: (B, num_visible) — original positions of visible patches
            mask_indices: (B, num_masked) — original positions of masked patches
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random shuffle for each sample in the batch
        noise = torch.rand(B, N, device=x.device)           # (B, N)
        shuffle_ids = torch.argsort(noise, dim=1)            # (B, N) — ascending

        visible_indices = shuffle_ids[:, :num_keep]          # (B, num_visible)
        mask_indices = shuffle_ids[:, num_keep:]             # (B, num_masked)

        # Gather visible tokens
        visible_tokens = torch.gather(
            x,
            dim=1,
            index=visible_indices.unsqueeze(-1).expand(-1, -1, D),
        )

        # Build binary mask: 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask.scatter_(1, visible_indices, 0.0)

        return visible_tokens, mask, visible_indices, mask_indices

    def forward(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full MAE forward pass.

        Returns:
            loss: scalar MSE reconstruction loss on masked patches only
            pred: (B, num_patches, patch_dim) decoder predictions
            mask: (B, num_patches) binary mask (1=masked)
        """
        B = imgs.shape[0]

        # 1. Patchify and compute normalized targets
        target = self.patchify(imgs)   # (B, N, patch_dim)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        # 2. Patch embedding + positional embedding (encoder handles this internally)
        #    but masking happens BEFORE the transformer blocks, so we need to:
        #    a) embed patches
        #    b) apply masking
        #    c) run transformer blocks

        x = self.encoder.patch_embed(imgs)   # (B, N, D_enc)

        # Add positional embeddings (only patch positions, skip index 0 = CLS in pos_embed)
        x = x + self.encoder.pos_embed[:, 1:, :]

        # 3. Random masking
        visible_tokens, mask, visible_indices, mask_indices = self.random_masking(x, self.mask_ratio)

        # 4. Prepend CLS token to visible tokens and run encoder transformer blocks
        cls = self.encoder.cls_token.expand(B, -1, -1)
        visible_with_cls = torch.cat([cls, visible_tokens], dim=1)  # (B, 1+num_visible, D)
        # CLS pos embed (index 0) is zeros in get_2d_sincos_pos_embed — already handled
        visible_with_cls = visible_with_cls + torch.cat([
            self.encoder.pos_embed[:, :1, :].expand(B, -1, -1),
            torch.zeros_like(visible_tokens),  # already added pos_embed above
        ], dim=1)

        for blk in self.encoder.blocks:
            visible_with_cls = blk(visible_with_cls)
        visible_with_cls = self.encoder.norm(visible_with_cls)

        # Drop CLS token before passing to decoder
        encoded_visible = visible_with_cls[:, 1:, :]  # (B, num_visible, D_enc)

        # 5. Decode
        pred = self.decoder(encoded_visible, visible_indices, mask_indices)  # (B, N, patch_dim)

        # 6. Loss: MSE on masked patches only
        loss_per_patch = ((pred - target) ** 2).mean(dim=-1)   # (B, N)
        loss = (loss_per_patch * mask).sum() / mask.sum()

        return loss, pred, mask


def build_mae(cfg: dict) -> "MAE":
    """
    Build MAE from a config dict.

    Expected keys:
        model.img_size, model.patch_size,
        model.encoder.{embed_dim, depth, num_heads, mlp_ratio, dropout},
        model.decoder.{embed_dim, depth, num_heads, mlp_ratio},
        model.mask_ratio, model.norm_pix_loss
    """
    enc_cfg = cfg["model"]["encoder"]
    dec_cfg = cfg["model"]["decoder"]
    img_size = cfg["model"].get("img_size", 224)
    patch_size = cfg["model"].get("patch_size", 16)
    grid_size = img_size // patch_size
    num_patches = grid_size ** 2

    encoder = ViTEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        embed_dim=enc_cfg["embed_dim"],
        depth=enc_cfg["depth"],
        num_heads=enc_cfg["num_heads"],
        mlp_ratio=enc_cfg.get("mlp_ratio", 4.0),
        dropout=enc_cfg.get("dropout", 0.0),
    )
    decoder = MAEDecoder(
        num_patches=num_patches,
        grid_size=grid_size,
        encoder_embed_dim=enc_cfg["embed_dim"],
        decoder_embed_dim=dec_cfg["embed_dim"],
        decoder_depth=dec_cfg["depth"],
        decoder_num_heads=dec_cfg["num_heads"],
        patch_size=patch_size,
        in_channels=3,
        mlp_ratio=dec_cfg.get("mlp_ratio", 4.0),
    )
    return MAE(
        encoder=encoder,
        decoder=decoder,
        mask_ratio=cfg["model"].get("mask_ratio", 0.75),
        norm_pix_loss=cfg["model"].get("norm_pix_loss", True),
    )
