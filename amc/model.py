# amc/model.py
"""
AMC — CNN model definitions.
Includes:
  - ModulationCNN: original classifier with a final linear layer (for end‑to‑end training).
  - CNNFeatureExtractor: same convolutional backbone but without the classifier head,
                         outputs a 1024‑dimensional feature vector for SVM.
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    Convolutional feature extractor.
    Input:  (batch, 2, 128)  (I/Q)
    Output: (batch, 1024)    flattened features (64 channels × 16 time steps)
    """
    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv1d(2, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        # Block 2
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        # Block 3
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(self.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)   # flatten -> (batch, 64*16)
        return x


class ModulationCNN(nn.Module):
    """
    Full CNN classifier (original) that includes the final linear layers.
    Used for end‑to‑end training with cross‑entropy loss.
    """
    def __init__(self, num_classes: int = 6, input_len: int = 128):
        super().__init__()
        # Feature extractor part
        self.features = CNNFeatureExtractor()
        # Classifier head
        self.fc1 = nn.Linear(64 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (batch, 1024)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_model(weights_path: str,
               num_classes: int = 6,
               input_len: int = 128,
               device: str = "cpu") -> ModulationCNN:
    """
    Load a trained ModulationCNN model for inference.
    (Used by the original AMCEngine if you switch back.)
    """
    model = ModulationCNN(num_classes=num_classes, input_len=input_len)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
