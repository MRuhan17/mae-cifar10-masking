"""Masking strategies for MAE pretraining.

Implements uniform random patch masking and masking utilities
for Vision Transformer patch grids.
"""

import torch
import numpy as np


def random_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.75,
    seed: int = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform random patch masking for MAE.
    
    Selects patches uniformly at random from a ViT patch grid to achieve
    target mask_ratio. Compatible with vision transformer patch embeddings.
    
    Args:
        x: Input tensor of shape (B, N, C) where:
            - B: batch size
            - N: number of patches (e.g., 196 for 14x14 grid from 224x224 input with 16x16 patches)
            - C: patch embedding dimension
        mask_ratio: Proportion of patches to mask [0.0, 1.0]. Default 0.75 (MAE standard).
        seed: Random seed for reproducibility. If None, uses non-deterministic random state.
    
    Returns:
        - x_masked: Unmasked patches only, shape (B, N*(1-mask_ratio), C)
        - mask: Binary mask indicating masked patches, shape (B, N) where 1 = masked, 0 = visible
        - ids_restore: Indices to restore original order, shape (B, N)
    
    Example:
        >>> B, N, C = 32, 196, 768  # batch_size=32, patches=196, embed_dim=768
        >>> x = torch.randn(B, N, C)
        >>> x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75)
        >>> print(x_masked.shape)  # (32, 49, 768)
        >>> print(mask.sum(dim=1))  # tensor([147, 147, ...]) - 75% of 196
    """
    B, N, C = x.shape  # (batch_size, num_patches, embed_dim)
    
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Number of patches to mask
    len_keep = int(N * (1 - mask_ratio))
    
    # Generate random noise for each patch in each batch element
    # Shape: (B, N)
    noise = torch.rand(B, N, device=x.device)
    
    # Sort noise: argsort returns indices that would sort the noise
    # ids_shuffle: (B, N) - indices sorted by noise (ascending)
    ids_shuffle = torch.argsort(noise, dim=1)  # Ascending
    ids_restore = torch.argsort(ids_shuffle, dim=1)  # Indices to restore original order
    
    # Keep indices (first len_keep patches after sorting = unmasked)
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Create binary mask: 1 for masked patches, 0 for visible
    # Initialize as all masked
    mask = torch.ones([B, N], device=x.device)
    # Set kept patches to 0 (not masked)
    mask.scatter_(1, ids_keep, 0)
    
    # Gather unmasked patches
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
    
    return x_masked, mask, ids_restore


def unmask_patches(
    x_masked: torch.Tensor,
    mask: torch.Tensor,
    ids_restore: torch.Tensor,
    mask_token: torch.Tensor = None
) -> torch.Tensor:
    """
    Restore masked patches back to full sequence order.
    
    Takes unmasked patches and reinserts mask tokens at masked positions
    to reconstruct the full patch sequence for decoder input.
    
    Args:
        x_masked: Unmasked patches, shape (B, N_visible, C)
        mask: Binary mask from masking step, shape (B, N) where 1 = masked
        ids_restore: Restore indices from masking step, shape (B, N)
        mask_token: Learnable mask token, shape (C,). If None, uses zeros.
    
    Returns:
        x_unmasked: Full sequence with mask tokens, shape (B, N, C)
    
    Example:
        >>> B, N, C = 32, 196, 768
        >>> x_masked, mask, ids_restore = random_masking(torch.randn(B, N, C))
        >>> mask_token = torch.nn.Parameter(torch.zeros(C))
        >>> x_full = unmask_patches(x_masked, mask, ids_restore, mask_token)
        >>> print(x_full.shape)  # (32, 196, 768)
    """
    B, N, C = x_masked.shape[0], mask.shape[1], x_masked.shape[2]
    
    if mask_token is None:
        mask_token = torch.zeros(C, device=x_masked.device)
    
    # Initialize output with mask tokens
    x_unmasked = mask_token.unsqueeze(0).unsqueeze(0).expand(B, N, -1).clone()
    
    # Place unmasked patches at positions where mask == 0
    x_unmasked[:, mask == 0] = x_masked.reshape(B * (N - mask.sum()), C).reshape(-1, C)
    
    # Restore original order using ids_restore
    x_unmasked = torch.gather(
        x_unmasked, 1,
        ids_restore.unsqueeze(-1).expand(-1, -1, C)
    )
    
    return x_unmasked


def create_patch_grid(image_size: int, patch_size: int) -> tuple[int, int]:
    """
    Calculate ViT patch grid dimensions.
    
    Args:
        image_size: Input image resolution (e.g., 224)
        patch_size: Patch size (e.g., 16)
    
    Returns:
        (num_patches_h, num_patches_w) or equivalently (grid_h, grid_w)
    
    Example:
        >>> h, w = create_patch_grid(224, 16)
        >>> print(f"Grid: {h}x{w}, Total patches: {h*w}")  # Grid: 14x14, Total patches: 196
    """
    assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"
    grid_size = image_size // patch_size
    return grid_size, grid_size


def visualize_mask(
    mask: torch.Tensor,
    grid_h: int,
    grid_w: int,
    save_path: str = None
) -> np.ndarray:
    """
    Visualize masking pattern for a single image.
    
    Args:
        mask: Binary mask, shape (B, N) or (N,) where 1 = masked
        grid_h: Grid height
        grid_w: Grid width
        save_path: Optional path to save visualization
    
    Returns:
        Visualization as numpy array (grid_h, grid_w)
    
    Example:
        >>> mask_single = mask[0]  # Take first batch element
        >>> vis = visualize_mask(mask_single, 14, 14)
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(vis, cmap='gray')
        >>> plt.savefig('mask_pattern.png')
    """
    if mask.dim() > 1:
        mask = mask[0]
    
    mask_grid = mask.reshape(grid_h, grid_w).cpu().numpy()
    
    if save_path:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(mask_grid, cmap='gray', vmin=0, vmax=1)
        plt.title('Patch Masking Pattern (white=masked, black=visible)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return mask_grid


if __name__ == "__main__":
    # Test random masking
    B, N, C = 4, 196, 768
    x = torch.randn(B, N, C)
    
    print(f"Input shape: {x.shape}")
    
    x_masked, mask, ids_restore = random_masking(x, mask_ratio=0.75, seed=42)
    
    print(f"Masked patches shape: {x_masked.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Masked ratio (actual): {mask.sum().item() / (B * N):.2%}")
    print(f"Restore indices shape: {ids_restore.shape}")
    
    # Test patch grid
    grid_h, grid_w = create_patch_grid(224, 16)
    print(f"\nPatch grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches")
    
    # Test visualization
    visualize_mask(mask, grid_h, grid_w, save_path="mask_pattern.png")
    print("Mask visualization saved to mask_pattern.png")




def block_masking(
        x: torch.Tensor,
    mask_ratio: float = 0.75,
    grid_h: int = 14,
    grid_w: int = 14,
    block_h_min: int = 2,
    block_h_max: int = 8,
    block_w_min: int = 2,
    block_w_max: int = 8,
    seed: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Block masking strategy for MAE pretraining.
    
    Samples a random rectangular region in patch space and masks all patches
    within that block. Dynamically adjusts block size to approximately match
    the target mask_ratio. More representative of real occlusions than random
    patch masking.
    
    Args:
        x: Input tensor of shape (B, N, C) where:
            - B: batch size
            - N: number of patches (grid_h * grid_w)
            - C: patch embedding dimension
        mask_ratio: Target proportion of patches to mask [0.0, 1.0]
        grid_h: Height of patch grid (e.g., 14 for 14x14)
        grid_w: Width of patch grid (e.g., 14 for 14x14)
        block_h_min: Minimum block height
        block_h_max: Maximum block height
        block_w_min: Minimum block width
        block_w_max: Maximum block width
        seed: Random seed for reproducibility
    
    Returns:
        - x_masked: Unmasked patches only, shape (B, N*(1-mask_ratio), C)
        - mask: Binary mask, shape (B, N) where 1 = masked, 0 = visible
        - ids_restore: Indices to restore original order, shape (B, N)
    
    Example:
        >>> B, N, C = 32, 196, 768  # 14x14 grid
        >>> x = torch.randn(B, N, C)
        >>> x_masked, mask, ids_restore = block_masking(
        ...     x, mask_ratio=0.75, grid_h=14, grid_w=14
        ... )
        >>> print(x_masked.shape)  # (32, ~49, 768)
        >>> print(f"Actual mask ratio: {mask.sum() / (B * N):.2%}")
    """
    B, N, C = x.shape
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Target number of patches to mask per image
    num_mask_patches = int(N * mask_ratio)
    
    # Initialize mask for entire batch
    mask = torch.zeros([B, grid_h, grid_w], device=x.device)
    
    for b in range(B):
        # Dynamically determine block size to match mask_ratio
        # Start with maximum block size and iteratively reduce if needed
        best_block_h, best_block_w = block_h_max, block_w_max
        min_diff = float('inf')
        
        # Try different block sizes to find closest match to mask_ratio
        for try_h in range(block_h_max, block_h_min - 1, -1):
            for try_w in range(block_w_max, block_w_min - 1, -1):
                num_masked = try_h * try_w
                diff = abs(num_masked - num_mask_patches)
                
                if diff < min_diff:
                    min_diff = diff
                    best_block_h, best_block_w = try_h, try_w
        
        # Sample random top-left corner for the block
        # Ensure block doesn't exceed grid boundaries
        max_h_start = max(0, grid_h - best_block_h)
        max_w_start = max(0, grid_w - best_block_w)
        
        h_start = torch.randint(0, max_h_start + 1, (1,)).item()
        w_start = torch.randint(0, max_w_start + 1, (1,)).item()
        
        # Mask the rectangular block
        h_end = min(h_start + best_block_h, grid_h)
        w_end = min(w_start + best_block_w, grid_w)
        
        mask[b, h_start:h_end, w_start:w_end] = 1
    
    # Reshape mask to (B, N)
    mask = mask.reshape(B, -1)
    
    # Create restore indices using noise-based sorting for robustness
    noise = torch.rand(B, N, device=x.device)
    # Make masked patches have high noise (will be sorted to end)
    noise[mask == 1] = noise[mask == 1] + 2.0
    
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep unmasked patches (minimum across batch for consistency)
    len_keep = (mask == 0).sum(dim=1).min().item()
    ids_keep = ids_shuffle[:, :len_keep]
    
    # Gather unmasked patches
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
    
    return x_masked, mask, ids_restore

def adaptive_block_masking(
    x: torch.Tensor,
    mask_ratio: float = 0.75,
    grid_h: int = 14,
    grid_w: int = 14,
    num_blocks: int = 1,
    seed: int = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Advanced block masking with multiple random blocks per image.
    
    Allows masking multiple non-overlapping rectangular regions to achieve
    more flexible mask ratios. Useful for varying difficulty levels during
    pretraining.
    
    Args:
        x: Input tensor of shape (B, N, C)
        mask_ratio: Target proportion of patches to mask
        grid_h: Height of patch grid
        grid_w: Width of patch grid
        num_blocks: Number of rectangular blocks to sample (non-overlapping)
        seed: Random seed for reproducibility
    
    Returns:
        - x_masked: Unmasked patches
        - mask: Binary mask indicating masked patches
        - ids_restore: Restore indices
    
    Example:
        >>> x_masked, mask, ids_restore = adaptive_block_masking(
        ...     x, mask_ratio=0.75, grid_h=14, grid_w=14, num_blocks=2
        ... )
    """
    B, N, C = x.shape
    
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Initialize mask
    mask = torch.zeros([B, grid_h, grid_w], device=x.device)
    num_mask_patches = int(N * mask_ratio)
    
    for b in range(B):
        # Target patches per block
        patches_per_block = num_mask_patches // num_blocks
        
        # Sample multiple blocks
        occupied = torch.zeros([grid_h, grid_w], device=x.device)
        
        for block_idx in range(num_blocks):
            # Try to find valid region
            max_attempts = 10
            placed = False
            
            for attempt in range(max_attempts):
                # Random block size (approximately square)
                block_size = int(np.sqrt(patches_per_block))
                block_h = torch.randint(max(1, block_size // 2), block_size + 2, (1,)).item()
                block_w = torch.randint(max(1, block_size // 2), block_size + 2, (1,)).item()
                
                # Random position
                max_h_start = max(1, grid_h - block_h)
                max_w_start = max(1, grid_w - block_w)
                h_start = torch.randint(0, max_h_start, (1,)).item()
                w_start = torch.randint(0, max_w_start, (1,)).item()
                
                h_end = min(h_start + block_h, grid_h)
                w_end = min(w_start + block_w, grid_w)
                
                # Check for overlap
                region = occupied[h_start:h_end, w_start:w_end]
                if region.sum() == 0:  # No overlap
                    mask[b, h_start:h_end, w_start:w_end] = 1
                    occupied[h_start:h_end, w_start:w_end] = 1
                    placed = True
                    break
            
            if not placed and attempt == max_attempts - 1:
                # Fallback: place block anyway (allow overlap)
                block_h = max(1, block_size // 2)
                block_w = max(1, block_size // 2)
                h_start = torch.randint(0, max(1, grid_h - block_h), (1,)).item()
                w_start = torch.randint(0, max(1, grid_w - block_w), (1,)).item()
                h_end = min(h_start + block_h, grid_h)
                w_end = min(w_start + block_w, grid_w)
                mask[b, h_start:h_end, w_start:w_end] = 1
    
    # Reshape and create restore indices
    mask = mask.reshape(B, -1)
    
    noise = torch.rand(B, N, device=x.device)
    noise[mask == 1] = noise[mask == 1] + 2.0
    
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    len_keep = (mask == 0).sum(dim=1).min().item()
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
    
    return x_masked, mask, ids_restore
