import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("--- AMC CNN TRAINING (FAST MODE) ---")

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), '..', 'amc_dataset.pt')
if not os.path.exists(dataset_path):
    print(f"ERROR: {dataset_path} not found. Run generate_dataset.py first.")
    sys.exit(1)

data = torch.load(dataset_path, weights_only=True)
X, Y = data['X'], data['Y']
print(f"Original shape: {X.shape}")

# ---- FORCE to (N, 2, 128) ----
if X.shape[1] == 128 and X.shape[2] == 2:
    X = X.permute(0, 2, 1)          # (N, 2, 128)
elif X.shape[1] != 2 or X.shape[2] != 128:
    raise ValueError(f"Bad shape: {X.shape} – expected (N,2,128) or (N,128,2)")

# ---- L2 normalise per sample (across channels & time) ----
norms = torch.norm(X, dim=(1,2), keepdim=True)
X = X / (norms + 1e-8)

# Train/val split
split = int(0.8 * len(X))
train_ds = TensorDataset(X[:split], Y[:split])
val_ds = TensorDataset(X[split:], Y[split:])
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# ---- Model (same as before) ----
class AMCCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.drop(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.drop(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.drop(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AMCCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

print(f"Training on {device} for 30 epochs...")

best_val = 0.0
for epoch in range(30):
    model.train()
    train_correct, train_total = 0, 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(out, 1)
        train_total += by.size(0)
        train_correct += (pred == by).sum().item()

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            _, pred = torch.max(model(bx), 1)
            val_total += by.size(0)
            val_correct += (pred == by).sum().item()

    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    scheduler.step(val_acc)

    if val_acc > best_val:
        best_val = val_acc
        save_path = os.path.join(os.path.dirname(__file__), '..', 'amc_cnn_weights.pth')
        torch.save(model.state_dict(), save_path)
        # Save class labels
        labels = ['BPSK', 'QPSK', 'QAM16', 'QAM64', 'BFSK', 'GFSK']
        with open(save_path.replace('.pth', '.labels.json'), 'w') as f:
            json.dump(labels, f)
        print(f"Saved best model (val {val_acc:.2f}%)")

    print(f"Epoch {epoch+1}/30 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val:.2f}%")

print("--- TRAINING DONE ---")