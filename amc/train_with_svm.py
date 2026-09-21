# amc/train_with_svm.py
import sys
import os
# Add the current directory to Python path so we can import model.py
sys.path.append(os.path.dirname(__file__))

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pickle

from model import CNNFeatureExtractor

print("--- AMC CNN + SVM TRAINING (START) ---")  # <-- immediate output

# Dataset is in the root folder, not '..'
dataset_path = 'amc_dataset.pt'
if not os.path.exists(dataset_path):
    print(f"ERROR: Dataset not found at {dataset_path}")
    print("Run 'python amc/generate_dataset.py' first.")
    sys.exit(1)

print("Loading dataset...")
data = torch.load(dataset_path, weights_only=True)
X, Y = data['X'], data['Y']
print(f"Original shape: {X.shape}")

# Fix shape to (N, 2, 128)
if X.shape[1] == 128 and X.shape[2] == 2:
    X = X.permute(0, 2, 1)
elif X.shape[1] != 2 or X.shape[2] != 128:
    raise ValueError(f"Bad shape: {X.shape}")

# Normalise per sample
norms = torch.norm(X, dim=(1,2), keepdim=True)
X = X / (norms + 1e-8)

# Split
split = int(0.8 * len(X))
train_X, train_Y = X[:split], Y[:split]
val_X, val_Y = X[split:], Y[split:]

train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(val_X, val_Y), batch_size=256, shuffle=False)

# ---- CNN Model (with classifier head) ----
class CNNwithHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = CNNFeatureExtractor()
        self.fc = nn.Linear(64*16, 6)  # 6 classes

    def forward(self, x):
        feat = self.features(x)
        return self.fc(feat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNwithHead().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Train the CNN (30 epochs)
print("Training CNN (30 epochs)...")
best_val_acc = 0
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
            out = model(bx)
            _, pred = torch.max(out, 1)
            val_total += by.size(0)
            val_correct += (pred == by).sum().item()
    
    train_acc = 100 * train_correct / train_total
    val_acc = 100 * val_correct / val_total
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'amc_cnn_with_head.pth')
        print(f"Best model saved (val {val_acc:.2f}%)")
    
    print(f"Epoch {epoch+1}/30 | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%")

# ---- Extract features for SVM ----
print("Extracting features for SVM...")
model.eval()
feature_extractor = model.features

def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            feat = feature_extractor(bx).cpu().numpy()
            features.append(feat)
            labels.append(by.numpy())
    return np.vstack(features), np.hstack(labels)

train_feat, train_labels = extract_features(train_loader)
val_feat, val_labels = extract_features(val_loader)

print(f"Train features shape: {train_feat.shape}")

# ---- Train SVM ----
scaler = StandardScaler()
train_feat_scaled = scaler.fit_transform(train_feat)
val_feat_scaled = scaler.transform(val_feat)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm.fit(train_feat_scaled, train_labels)

val_pred = svm.predict(val_feat_scaled)
val_acc_svm = 100 * np.mean(val_pred == val_labels)
print(f"SVM validation accuracy: {val_acc_svm:.2f}%")

# ---- Save everything ----
torch.save(feature_extractor.state_dict(), 'amc_feature_extractor.pth')
with open('amc_svm.pkl', 'wb') as f:
    pickle.dump(svm, f)
with open('amc_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

labels = ['BPSK', 'QPSK', 'QAM16', 'QAM64', 'BFSK', 'GFSK']
with open('amc_labels.json', 'w') as f:
    json.dump(labels, f)

print("SVM model and feature extractor saved.")
print("--- TRAINING COMPLETE ---")