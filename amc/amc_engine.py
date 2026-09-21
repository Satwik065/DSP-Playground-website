from __future__ import annotations
import json, os, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
from .model import CNNFeatureExtractor  # we'll define this in model.py

_DEFAULT_CLASS_LABELS = ["BPSK", "QPSK", "QAM16", "QAM64", "BFSK", "GFSK"]
_INPUT_LEN = 128

class AMCEngine:
    def __init__(self, 
                 feature_weights: str = "amc_feature_extractor.pth",
                 svm_path: str = "amc_svm.pkl",
                 scaler_path: str = "amc_scaler.pkl",
                 labels_path: str = "amc_labels.json",
                 device: str = "cpu"):
        self.device = device
        
        # Load class labels
        if os.path.exists(labels_path):
            with open(labels_path) as f:
                self.class_labels = json.load(f)
        else:
            self.class_labels = _DEFAULT_CLASS_LABELS
            warnings.warn("Labels file missing, using defaults.")
        
        # Load feature extractor
        self.feature_extractor = CNNFeatureExtractor().to(device)
        self.feature_extractor.load_state_dict(torch.load(feature_weights, map_location=device))
        self.feature_extractor.eval()
        
        # Load SVM and scaler
        with open(svm_path, 'rb') as f:
            self.svm = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.eps = 1e-8

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(x, dim=(1,2), keepdim=True)
        return x / (norms + self.eps)

    def classify(self, iq: np.ndarray) -> dict:
        iq = np.asarray(iq, dtype=np.float32)
        if iq.shape != (2, _INPUT_LEN):
            raise ValueError(f"Expected (2,{_INPUT_LEN}), got {iq.shape}")
        
        x = torch.from_numpy(iq).unsqueeze(0).to(self.device)
        x = self._normalize(x)
        
        with torch.no_grad():
            feat = self.feature_extractor(x).cpu().numpy()  # (1, 1024)
        
        feat_scaled = self.scaler.transform(feat)
        probs = self.svm.predict_proba(feat_scaled)[0]   # (num_classes,)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        
        return {
            "predicted_modulation": self.class_labels[pred_idx],
            "confidence": confidence,
            "probabilities": {l: float(p) for l, p in zip(self.class_labels, probs)}
        }

    def classify_batch(self, iq_batch: np.ndarray) -> list[dict]:
        iq_batch = np.asarray(iq_batch, dtype=np.float32)
        x = torch.from_numpy(iq_batch).to(self.device)
        x = self._normalize(x)
        with torch.no_grad():
            feats = self.feature_extractor(x).cpu().numpy()
        feats_scaled = self.scaler.transform(feats)
        probs = self.svm.predict_proba(feats_scaled)
        
        results = []
        for row in probs:
            idx = int(np.argmax(row))
            results.append({
                "predicted_modulation": self.class_labels[idx],
                "confidence": float(row[idx]),
                "probabilities": {l: float(p) for l, p in zip(self.class_labels, row)}
            })
        return results
