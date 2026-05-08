import numpy as np
import torch
import torch.nn as nn

from .patch_embed import PatchEmbed


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Build sinusoidal 2D positional embeddings.

    Returns:
        numpy array of shape (grid_size**2, embed_dim)
    """
    half_dim = embed_dim // 2
    # 1D sincos for each spatial axis
    omega = np.arange(half_dim // 2, dtype=np.float64)
    omega /= half_dim // 2
    omega = 1.0 / (10000 ** omega)  # (D/4,)

    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid_h, grid_w = np.meshgrid(grid_h, grid_w, indexing="ij")  # (H, W)
    grid_h = grid_h.reshape(-1)  # (N,)
    grid_w = grid_w.reshape(-1)  # (N,)

    out_h = np.outer(grid_h, omega)  # (N, D/4)
    out_w = np.outer(grid_w, omega)  # (N, D/4)

    emb_h = np.concatenate([np.sin(out_h), np.cos(out_h)], axis=1)  # (N, D/2)
    emb_w = np.concatenate([np.sin(out_w), np.cos(out_w)], axis=1)  # (N, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (N, D)
    return emb.astype(np.float32)


class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block:
    LN → MHSA → residual → LN → MLP → residual

    Args:
        embed_dim: token dimension
        num_heads: number of attention heads
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio
        dropout: applied to attention weights and MLP output
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, embed_dim) → (B, N, embed_dim)"""
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder.

    Used as the heavy backbone for both MAE pre-training and supervised baseline.
    During MAE pre-training the caller passes only the visible patch tokens
    (masking is handled externally in MAE.forward).
    During supervised training all 196 tokens are passed.

    Args:
        img_size: 224
        patch_size: 16
        in_channels: 3
        embed_dim: 768 for ViT-Base
        depth: 12
        num_heads: 12
        mlp_ratio: 4.0
        dropout: 0.0
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed sinusoidal positional embeddings — registered as buffer (no grad)
        pos_embed = get_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        # Shape: (1, num_patches + 1, embed_dim)  (+1 for CLS)
        pos_embed_with_cls = np.zeros((1, num_patches + 1, embed_dim), dtype=np.float32)
        pos_embed_with_cls[0, 1:] = pos_embed
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed_with_cls))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) — full image; masking is done externally for MAE.
        Returns: (B, num_patches + 1, embed_dim)  [index 0 = CLS token]
        """
        B = x.shape[0]
        x = self.patch_embed(x)          # (B, N, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)            # (B, N+1, D)

        # Add positional embeddings
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        return self.norm(x)
