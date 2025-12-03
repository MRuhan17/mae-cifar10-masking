import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from cifar10_loader import get_cifar10_loaders
from mae_vit_config import build_mae_encoder


class LinearProbe(nn.Module):
    def __init__(self, encoder, embed_dim, num_classes=10):
        super().__init__()
        self.encoder = encoder  # frozen MAE encoder
        self.classifier = nn.Linear(embed_dim, num_classes)

        # freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)  # feature vector from MAE encoder
        logits = self.classifier(feats)
        return logits


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


def train_linear_probe(encoder_path, batch_size, epochs, lr, device):
    # 1. Load CIFAR-10 with labels
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    # 2. Build and load pretrained encoder
    encoder, embed_dim = build_mae_encoder()
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    encoder.eval()

    # 3. Create linear probe model
    model = LinearProbe(encoder, embed_dim).to(device)

    # 4. Train only classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs}  |  Loss: {epoch_loss:.4f}  |  Test Acc: {test_acc*100:.2f}")

    return test_acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--encoder_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    acc = train_linear_probe(
        encoder_path=args.encoder_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

    print(f"\nFinal linear probe accuracy: {acc*100:.2f}")


if __name__ == "__main__":
    main()
