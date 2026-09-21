# generate_amc_dataset.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
from dcp.digital_comm import generate_iq_window

# --- Configuration (reduce samples for faster training) ---
MODULATIONS = ['bpsk', 'qpsk', '16qam', '64qam', 'bfsk', 'gfsk']
SNR_RANGE = list(range(-20, 20, 2))          # -20 to 18 dB
SAMPLES_PER_CLASS_PER_SNR = 200              # <-- reduce to 200 for speed
WINDOW_LEN = 128

def main():
    print("Generating dataset (fast mode: 200 per SNR per class)...")
    X_data, Y_data = [], []
    for mod_idx, mod in enumerate(MODULATIONS):
        print(f"Generating {mod.upper()}...")
        for snr in SNR_RANGE:
            for _ in range(SAMPLES_PER_CLASS_PER_SNR):
                iq = generate_iq_window(mod, snr_db=snr, channel='awgn', window_len=WINDOW_LEN)
                iq = iq.T   # (2, 128) – correct shape
                X_data.append(iq)
                Y_data.append(mod_idx)

    X = torch.tensor(np.array(X_data), dtype=torch.float32)
    Y = torch.tensor(Y_data, dtype=torch.long)

    idx = torch.randperm(len(X))
    X, Y = X[idx], Y[idx]

    torch.save({'X': X, 'Y': Y}, 'amc_dataset.pt')
    print(f"Saved dataset shape: {X.shape}  (N, 2, 128)")

if __name__ == "__main__":
    main()
