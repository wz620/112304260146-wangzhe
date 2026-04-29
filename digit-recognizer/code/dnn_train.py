import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

DATA_DIR = r"d:\邪恶冥刻模组\实验\kaggle\手写数字\digit-recognizer"

print("Loading data...")
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

X_train = train_df.drop("label", axis=1).values.astype(np.float32) / 255.0
y_train = train_df["label"].values
X_test = test_df.values.astype(np.float32) / 255.0

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        return self.fc(x)

model = SimpleDNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

print("\nTraining DNN...")
model.train()
for epoch in range(15):
    total_loss = 0
    correct = 0
    total = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()
    
    scheduler.step()
    acc = 100.0 * correct / total
    print(f"Epoch {epoch+1}/15 - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.2f}%")

print("\nPredicting...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predictions = outputs.max(1)
    predictions = predictions.numpy()

submission = pd.DataFrame({
    "ImageId": range(1, len(predictions) + 1),
    "Label": predictions
})

submission.to_csv(f"{DATA_DIR}/submission.csv", index=False)
print(f"\nSubmission saved to {DATA_DIR}/submission.csv")
print(submission.head(10))
