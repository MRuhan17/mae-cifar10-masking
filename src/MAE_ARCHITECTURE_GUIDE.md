# Configuring Small ViT-based MAE for 224×224 Images

## Overview

This guide explains how to configure a small Vision Transformer-based Masked Autoencoder (MAE) for 224×224 images following architecture choices from **[facebookresearch/mae](https://github.com/facebookresearch/mae)**.

The reference implementation from Meta uses ViT-Base and ViT-Large models with asymmetric encoder-decoder architectures optimized for self-supervised learning.

## Key Architecture Decisions

### 1. **Patch Embedding (Patch Size = 16)**

```
224x224 image -> 16x16 patches -> 14x14 grid -> 196 patches
```

For 224×224 input images:
- **Patch size**: 16×16 (chosen for 14×14 patch grid)
- **Number of patches**: (224÷16)² = 14² = **196 patches**
- **Patch embedding dim**: Project 3-channel patches to 384-dim (encoder) or 192-dim (decoder)

The patch size of 16 is optimal for balancing:
- Computational efficiency (196 tokens vs 3136 for 8×8 patches)
- Semantic information preservation
- Compatibility with ImageNet pretraining conventions

### 2. **Encoder Architecture (ViT-Small)**

The **small ViT (ViT-S)** configuration from facebookresearch/mae uses:

```python
encode_dim = 384              # Hidden dimension
encoder_depth = 6             # Number of transformer blocks
encoder_num_heads = 6         # Attention heads
mlp_ratio = 4.0               # MLP expansion ratio
```

**Encoder structure**:
```
Input (224x224x3)
    ↓
Patch Embedding (196x384)
    ↓
CLS Token + Positional Embedding (197x384)
    ↓
[TransformerBlock]×6
    - MultiHeadAttention (384-dim, 6 heads, 64-dim per head)
    - MLP (384 → 1536 → 384)
    ↓
LayerNorm
    ↓
Encoded features (197x384)
```

**Why 6 layers?**
- Lightweight encoder reduces computational cost during pretraining
- facebookresearch/mae shows that encoding only ~25% of patches works well
- Asymmetric design: encoder processes all visible patches, decoder reconstructs masked ones

### 3. **Decoder Architecture (Lightweight)**

The decoder in MAE is intentionally lightweight:

```python
decoder_embed_dim = 192       # Reduced dimension
decoder_depth = 8             # More layers but smaller dimension
decoder_num_heads = 3         # Fewer attention heads
```

**Decoder structure**:
```
Encoded features (196x384 without CLS)
    ↓
Linear projection to 192-dim
    ↓
Add mask tokens for masked patches (196x192)
    ↓
[TransformerBlock]×8
    - MultiHeadAttention (192-dim, 3 heads, 64-dim per head)
    - MLP (192 → 768 → 192)
    ↓
LayerNorm
    ↓
Linear projection to patch_size²×channels (196x768)
    ↓
Patch reconstruction (224x224x3)
```

**Design rationale**:
- Decoder dimension reduced to 192-dim (~50% of encoder)
- More layers (8 vs 6) compensates for lower dimension
- facebookresearch/mae paper shows decoder asymmetry is key to MAE efficiency
- Mask tokens are learnable parameters that guide reconstruction of masked patches

### 4. **Masking Strategy (75% Masking)**

```python
masking_ratio = 0.75  # Mask 75% of patches
```

**Masking process**:
1. Generate random noise for each patch
2. Sort patches by noise (shuffle)
3. Keep first 25% of shuffled patches (49 patches)
4. Feed 49 visible patches to encoder
5. Decoder receives:
   - Encoded features of 49 visible patches
   - 147 learnable mask tokens for masked patches
6. Reconstruct all 196 patches

**Why 75%?**
- facebookresearch/mae experiments show 75% is optimal
- Forces model to learn meaningful representations
- Reduces computational load (3.3x reduction vs no masking)
- Enables faster training

### 5. **Positional Embeddings**

```python
pos_embed_encoder = nn.Parameter(torch.zeros(1, 197, 384))
pos_embed_decoder = nn.Parameter(torch.zeros(1, 196, 192))
```

- **Learned positional embeddings** (not sinusoidal)
- Initialized with normal distribution N(0, 0.02)
- Separate embeddings for encoder (with CLS token) and decoder (patches only)
- facebookresearch/mae shows learned embeddings work better than sinusoidal

## Implementation Details from facebookresearch/mae

### Pre-norm Architecture

facebookresearch/mae uses **pre-normalization** in transformer blocks:

```python
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Pre-norm: normalize before attention
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

Benefits:
- More stable training
- Better gradient flow
- Standard in modern Vision Transformers

### Weight Initialization

```python
# Positional embeddings: Normal(0, 0.02)
nn.init.normal_(pos_embed, std=0.02)
nn.init.normal_(cls_token, std=0.02)
nn.init.normal_(mask_token, std=0.02)

# Linear layers: Xavier uniform
nn.init.xavier_uniform_(linear.weight)
nn.init.constant_(linear.bias, 0)
```

### Reconstruction Loss

facebookresearch/mae uses **MSE loss on normalized patches**:

```python
loss = F.mse_loss(pred, target, reduction='mean')
# Where pred and target are normalized patch values
# Only computed on masked patches (important!)
```

## Configuration Comparison

| Component | ViT-Small (MAE) | ViT-Base (MAE) | ViT-Large (MAE) |
|-----------|-----------------|----------------|------------------|
| Embed Dim | 384 | 768 | 1024 |
| Encoder Depth | 6 | 12 | 24 |
| Decoder Dim | 192 | 384 | 512 |
| Decoder Depth | 8 | 8 | 8 |
| Num Heads | 6 | 12 | 16 |
| Parameters | ~36M | ~111M | ~306M |

## Using the Implementation

### Loading the Model

```python
from src.mae_vit_config import mae_vit_small

model = mae_vit_small(masking_ratio=0.75)
```

### Forward Pass

```python
import torch

# Input: batch of images
x = torch.randn(batch_size, 3, 224, 224)

# Forward pass returns:
# - pred: reconstructed patch values (batch_size, 196, 768)
# - mask: binary mask (batch_size, 196) where 1=masked, 0=visible
pred, mask = model(x)

# Compute loss only on masked patches
loss = F.mse_loss(pred[mask.bool()], target[mask.bool()])
```

## Key References from facebookresearch/mae

1. **Paper**: "Masked Autoencoders Are Scalable Vision Learners" (He et al., 2021)
2. **Architecture decisions**:
   - Asymmetric encoder-decoder design
   - 75% masking ratio
   - Pre-norm transformer blocks
   - Learned positional embeddings
   - Patch-based reconstruction (not pixel-level)

3. **Implementation patterns**:
   - Mask token learning
   - Random patch shuffling
   - Efficient encoder-only inference after pretraining

## Training Optimization Tips

1. **Masking ratio**: 75% is optimal, but 50-90% works well
2. **Batch size**: 256-512 typical for CIFAR-10 (smaller dataset)
3. **Learning rate**: 1e-4 to 5e-4 with cosine annealing
4. **Warmup**: 40 epochs warmup recommended
5. **Weight decay**: 0.05 for regularization

## Scaling to Different Sizes

For experimenting with different configurations:

```python
# Ultra-small (faster experimentation)
model = MaskedAutoencoderViT(
    embed_dim=256,
    encoder_depth=3,
    encoder_num_heads=4,
    decoder_embed_dim=128,
    decoder_depth=4,
    decoder_num_heads=2
)

# ViT-Base equivalent
model = MaskedAutoencoderViT(
    embed_dim=768,
    encoder_depth=12,
    encoder_num_heads=12,
    decoder_embed_dim=384,
    decoder_depth=8,
    decoder_num_heads=6
)
```

## Summary

The small ViT-based MAE configuration is designed for:
- **Efficiency**: 36M parameters for CIFAR-10 pretraining
- **Scalability**: Architecture works for any image size >= 224×224
- **Effectiveness**: Shows strong transfer learning performance

Key design principles from facebookresearch/mae:
1. Asymmetric encoder-decoder (light encoder, heavy decoder)
2. High masking ratio (75%) with random shuffling
3. Reconstruction-only loss on masked patches
4. Learned components (positional embeddings, mask tokens)
5. Pre-norm architecture for stability
