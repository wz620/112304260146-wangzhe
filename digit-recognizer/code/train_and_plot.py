import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    return (pred.argmax(dim=1) == labels).float().mean().item()


def main() -> None:
    set_seed(42)
    device = torch.device("cpu")

    base_dir = r"d:\邪恶冥刻模组\实验\kaggle\手写数字\digit-recognizer"
    train_path = f"{base_dir}/train.csv"
    test_path = f"{base_dir}/test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = train_df.drop(columns=["label"]).values.astype(np.float32) / 255.0
    y = train_df["label"].values.astype(np.int64)
    X_test = test_df.values.astype(np.float32) / 255.0

    X = X.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(torch.from_numpy(X_test))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    train_losses = []
    val_losses = []
    epochs = []

    best_acc = 0.0
    best_state = None
    patience = 4
    no_improve = 0

    for epoch in range(1, 9):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_steps = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy(logits, yb)
            train_steps += 1

        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits, yb)
                val_steps += 1

        scheduler.step()

        train_loss /= train_steps
        train_acc /= train_steps
        val_loss /= val_steps
        val_acc /= val_steps

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs.append(epoch)

        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.", flush=True)
                break

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{base_dir}/loss_curve.png", dpi=150, bbox_inches='tight')
    print(f"\nLoss curve saved to {base_dir}/loss_curve.png")

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb in test_loader:
            xb = xb[0].to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())

    submission = pd.DataFrame({
        "ImageId": range(1, len(preds) + 1),
        "Label": preds
    })
    submission.to_csv(f"{base_dir}/submission.csv", index=False)
    print(f"Submission saved to {base_dir}/submission.csv")


if __name__ == "__main__":
    main()
