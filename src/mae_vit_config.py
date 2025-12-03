"""Small ViT-based MAE Model Configuration

This module provides PyTorch implementations of small Vision Transformer-based
Masked Autoencoders (MAE) for 224x224 images, following architecture choices from
facebookresearch/mae repository.

Architecture references:
- Paper: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
- Repository: https://github.com/facebookresearch/mae

Key design decisions:
1. Patch embedding: 16x16 patches for 224x224 images (14x14 = 196 patches)
2. Encoder: ViT-Small (384-dim hidden, 6 layers, 6 heads)
3. Decoder: Lightweight decoder (8 layers, symmetric to encoder)
4. Masking: 75% masking ratio for self-supervised pretraining
"""

import torch
import torch.nn as nn
from functools import partial
import numpy as np


class PatchEmbedding(nn.Module):
    """Convert image to patch embeddings.
    
    Args:
        img_size (int): Input image size (224)
        patch_size (int): Patch size (16)
        in_channels (int): Number of input channels (3)
        embed_dim (int): Embedding dimension
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14*14 = 196
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, num_patches_h, num_patches_w) -> (B, num_patches, embed_dim)
        x = self.proj(x)  # (B, embed_dim, 14, 14)
        x = x.flatten(2)  # (B, embed_dim, 196)
        x = x.transpose(1, 2)  # (B, 196, embed_dim)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block from facebookresearch/mae.
    
    Follows: LayerNorm -> MultiHeadAttention -> LayerNorm -> MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Pre-norm architecture (like in MAE)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone.
    
    Configuration based on ViT-Small from facebookresearch/mae:
    - Encoder: 6 transformer blocks, 384 hidden dim, 6 heads
    - Decoder: 8 transformer blocks, 192 hidden dim, 3 heads
    
    Args:
        img_size (int): Input image size (224)
        patch_size (int): Patch size (16)
        in_channels (int): Input channels (3)
        embed_dim (int): Encoder embedding dimension (384)
        encoder_depth (int): Number of encoder blocks (6)
        encoder_num_heads (int): Number of attention heads in encoder (6)
        decoder_embed_dim (int): Decoder embedding dimension (192)
        decoder_depth (int): Number of decoder blocks (8)
        decoder_num_heads (int): Number of attention heads in decoder (3)
        mlp_ratio (float): MLP hidden dim ratio (4.0)
        norm_layer (nn.Module): Normalization layer (LayerNorm)
        masking_ratio (float): Ratio of patches to mask (0.75)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        masking_ratio=0.75,
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Encoder
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_encoder = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=encoder_num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = norm_layer(embed_dim)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed_decoder = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)
        
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights following MAE paper."""
        # Positional embeddings (normal initialization)
        nn.init.normal_(self.pos_embed_encoder, std=0.02)
        nn.init.normal_(self.pos_embed_decoder, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def mask_patches(self, x):
        """Random masking: reshuffle patches and mask.
        
        Args:
            x: (B, num_patches, embed_dim)
        
        Returns:
            x_masked: (B, num_patches_keep, embed_dim) - unmasked patches
            mask: (B, num_patches) - binary mask
            ids_restore: (B, num_patches) - indices to restore original order
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.masking_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first len_keep patches after shuffling
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Generate mask: 1 is masked, 0 is kept
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore

    def encode(self, x):
        """Encoder: patch embedding + transformer blocks.
        
        Args:
            x: (B, 3, 224, 224)
        
        Returns:
            Encoded features for decoder
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, 196, embed_dim)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 197, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed_encoder
        x = self.pos_drop(x)
        
        # Apply encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.encoder_norm(x)
        return x

    def forward(self, x):
        """Forward pass with masking.
        
        Args:
            x: (B, 3, 224, 224)
        
        Returns:
            Loss prediction (reconstructed patches)
        """
        # Encode
        x_encoded = self.encode(x)  # (B, 197, embed_dim)
        
        # Remove cls token for masking
        x_patches = x_encoded[:, 1:, :]  # (B, 196, embed_dim)
        
        # Mask patches
        x_masked, mask, ids_restore = self.mask_patches(x_patches)
        
        # Project to decoder dimension
        x_dec = self.decoder_embed(x_masked)  # (B, len_keep, decoder_embed_dim)
        
        # Add mask tokens for masked patches
        mask_tokens = self.mask_token.expand(
            x_dec.shape[0],
            ids_restore.shape[1] - x_dec.shape[1],
            -1
        )  # (B, len_mask, decoder_embed_dim)
        
        x_dec = torch.cat([x_dec, mask_tokens], dim=1)  # (B, 196, decoder_embed_dim)
        x_dec = torch.gather(
            x_dec, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_dec.shape[2])
        )  # Restore order
        
        # Add decoder positional embedding (without cls token)
        x_dec = x_dec + self.pos_embed_decoder[:, 1:, :]
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x_dec = block(x_dec)
        
        x_dec = self.decoder_norm(x_dec)
        
        # Predict patches
        x_pred = self.decoder_pred(x_dec)  # (B, 196, patch_size^2 * channels)
        
        return x_pred, mask


def mae_vit_small(masking_ratio=0.75, **kwargs):
    """Small MAE-ViT configuration (ViT-S/16).
    
    Encoder: 6 blocks, 384-dim, 6 heads
    Decoder: 8 blocks, 192-dim, 3 heads
    """
    model = MaskedAutoencoderViT(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        encoder_depth=6,
        encoder_num_heads=6,
        decoder_embed_dim=192,
        decoder_depth=8,
        decoder_num_heads=3,
        mlp_ratio=4.0,
        masking_ratio=masking_ratio,
        **kwargs
    )
    return model


if __name__ == '__main__':
    # Example usage
    model = mae_vit_small(masking_ratio=0.75)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    pred, mask = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Prediction shape: {pred.shape}')
    print(f'Mask shape: {mask.shape}')
    print(f'Masking ratio: {mask.float().mean():.3f}')
