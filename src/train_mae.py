import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# assume you have defined: MaskedAutoencoder class that implements MAE (encoder + decoder + mask logic)
# For simplicity: MaskedAutoencoder(image_size=32, patch_size=…) -> returns model with .encoder + .decoder + mask logic

def train_mae_on_cifar10(batch_size=128, mask_ratio=0.75, num_epochs=100, device='cuda'):
    # 1. Data loading and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        # possibly add minimal augmentation like random crop / horizontal flip
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
    ])
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Build model
    model = MaskedAutoencoder(
        img_size=32,
        patch_size=4,   # just example; choose patch_size dividing 32
        mask_ratio=mask_ratio,
        embed_dim=...,  # depends on your ViT variant
        encoder_depth=..., 
        decoder_depth=..., 
    ).to(device)

    criterion = nn.MSELoss()  # reconstruction loss on masked patches (pixel space)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # 3. Training loop
    loss_log = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)

            # forward pass: model should apply random masking internally (or you apply before)
            reconstructed, mask = model(images)  
            # assume reconstructed => full image, and mask indicates which patches were masked
            # compute loss only over masked patches
            loss = model.masked_mse_loss(images, reconstructed, mask)  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        loss_log.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}  Loss: {epoch_loss:.4f}")

    # 4. Save encoder weights and loss logs
    torch.save(model.encoder.state_dict(), "mae_encoder_cifar10.pth")
    torch.save({"loss": loss_log}, "mae_pretrain_loss_log.pth")
    return loss_log
