import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """
    Splits a (B, C, H, W) image into non-overlapping patches and projects
    each patch to embed_dim via a single Conv2d.

    Args:
        img_size: assumed square input size (default 224)
        patch_size: patch side length (default 16)
        in_channels: number of input channels (default 3)
        embed_dim: output embedding dimension
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self._num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    @property
    def num_patches(self) -> int:
        return self._num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        x = self.proj(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)          # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, embed_dim)
        return x
