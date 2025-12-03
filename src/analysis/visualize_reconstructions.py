import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from cifar10_loader import get_sample_images
from mae_vit_config import build_mae_full_model

OUTPUT_DIR = "results/reconstructions"

MASK_TYPES = ["random", "block"]
MASK_RATIOS = ["0.4", "0.6", "0.75", "0.9"]

device = "cuda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

images = get_sample_images(8).to(device)  # your helper that loads few CIFAR10 samples

for mtype in MASK_TYPES:
    for mratio in MASK_RATIOS:

        encoder_path = f"results/pretrain/{mtype}_{mratio}/encoder.pth"
        
        model = build_mae_full_model(mask_type=mtype, mask_ratio=float(mratio))
        model.encoder.load_state_dict(torch.load(encoder_path))
        model.to(device)
        model.eval()

        with torch.no_grad():
            reconstructed, mask = model(images)

        # save grid
        fig, axes = plt.subplots(3, 1, figsize=(8, 6))

        axes[0].imshow(make_grid(images.cpu(), nrow=4).permute(1, 2, 0))
        axes[0].set_title("Original Images")
        axes[0].axis("off")

        axes[1].imshow(make_grid(mask.cpu(), nrow=4).permute(1, 2, 0))
        axes[1].set_title("Masked Images")
        axes[1].axis("off")

        axes[2].imshow(make_grid(reconstructed.cpu(), nrow=4).permute(1, 2, 0))
        axes[2].set_title(f"Reconstructed: {mtype} {mratio}")
        axes[2].axis("off")

        save_path = os.path.join(OUTPUT_DIR, f"{mtype}_{mratio}.png")
        plt.savefig(save_path)
        plt.close()
