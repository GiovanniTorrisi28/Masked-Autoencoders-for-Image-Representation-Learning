import numpy as np
import torch
import torch.nn as nn

from .vit_encoder import TransformerBlock, get_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """
    Lightweight ViT decoder for MAE pre-training.

    Receives the encoder's visible token outputs plus mask tokens inserted at
    the masked positions (with positional embeddings added to both) and
    reconstructs per-patch pixel values for ALL patches.

    Args:
        num_patches: total number of patches (196 for 224×224 / 16×16)
        grid_size: sqrt(num_patches), used for 2D pos embed
        encoder_embed_dim: dimension of incoming encoder tokens (768)
        decoder_embed_dim: internal decoder dimension (512)
        decoder_depth: number of Transformer blocks (8)
        decoder_num_heads: attention heads in decoder (16)
        patch_size: 16
        in_channels: 3
        mlp_ratio: 4.0
    """

    def __init__(
        self,
        num_patches: int = 196,
        grid_size: int = 14,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        patch_size: int = 16,
        in_channels: int = 3,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.patch_dim = patch_size * patch_size * in_channels

        # Project encoder dim → decoder dim
        self.embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        # Learned mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Fixed sinusoidal pos embed for the decoder (num_patches positions only, no CLS)
        pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).unsqueeze(0))  # (1, N, D)

        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.head = nn.Linear(decoder_embed_dim, self.patch_dim, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        visible_tokens: torch.Tensor,    # (B, num_visible, encoder_embed_dim)
        visible_indices: torch.Tensor,   # (B, num_visible) — patch positions in [0, N)
        mask_indices: torch.Tensor,      # (B, num_masked)  — patch positions in [0, N)
    ) -> torch.Tensor:
        """
        Returns:
            pred: (B, num_patches, patch_dim) — predictions for ALL patches.
                  The loss will select only masked positions.
        """
        B = visible_tokens.shape[0]
        N = self.num_patches

        # 1. Project visible tokens to decoder dim
        visible_tokens = self.embed(visible_tokens)   # (B, num_visible, D_dec)

        # 2. Add positional embedding to visible tokens
        vis_pos = self.pos_embed.expand(B, -1, -1)    # (B, N, D_dec)
        visible_pos_emb = torch.gather(
            vis_pos,
            dim=1,
            index=visible_indices.unsqueeze(-1).expand(-1, -1, vis_pos.shape[-1]),
        )
        visible_tokens = visible_tokens + visible_pos_emb

        # 3. Expand mask token and add positional embedding
        mask_tokens = self.mask_token.expand(B, mask_indices.shape[1], -1)  # (B, num_masked, D_dec)
        mask_pos_emb = torch.gather(
            vis_pos,
            dim=1,
            index=mask_indices.unsqueeze(-1).expand(-1, -1, vis_pos.shape[-1]),
        )
        mask_tokens = mask_tokens + mask_pos_emb

        # 4. Reconstruct full sequence in original position order
        # Build an empty tensor and scatter both groups back to their positions
        full = torch.zeros(B, N, visible_tokens.shape[-1], device=visible_tokens.device, dtype=visible_tokens.dtype)
        full.scatter_(
            1,
            visible_indices.unsqueeze(-1).expand(-1, -1, full.shape[-1]),
            visible_tokens,
        )
        full.scatter_(
            1,
            mask_indices.unsqueeze(-1).expand(-1, -1, full.shape[-1]),
            mask_tokens,
        )

        # 5. Pass through decoder blocks
        for blk in self.blocks:
            full = blk(full)

        full = self.norm(full)
        pred = self.head(full)   # (B, N, patch_dim)
        return pred
