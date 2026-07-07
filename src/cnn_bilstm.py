"""
CNN-BiLSTM model for Remaining Useful Life (RUL) prediction on NASA CMAPSS.

This is one of the three SOTA methods in the project. It mirrors the structure
of baseline.py and reuses data_loader.py (for loading / RUL / dead-sensor logic)
and scoring.py (for the RMSE + NASA score).

Pipeline:
    1. Load train + test, compute RUL on train, drop "dead" (constant) sensors.
    2. Clip RUL with a piecewise-linear cap (standard trick for CMAPSS).
    3. Min-Max scale sensors (fit on train, apply to test).
    4. Build sliding windows of length WINDOW so the network sees a short
       time-history instead of a single cycle.
    5. Train a CNN -> BiLSTM -> Dense regressor.
    6. Predict one RUL per test engine from its LAST window, then score.
"""

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from data_loader import data_load, compute_rul, get_dead_sensors
from scoring import score

# ----------------------------- Config --------------------------------------
project_dir = Path(__file__).parent.parent

WINDOW     = 30      # number of cycles fed to the model at once
MAX_RUL    = 125     # piecewise-linear RUL cap (engine is "healthy" above this)
BATCH_SIZE = 256
EPOCHS     = 20      # ~18-20 is the sweet spot on FD001 (longer can overfit)
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------- Windowing helpers --------------------------------
def make_train_windows(df, sensor_cols, window=WINDOW):
    """Slide a window over each engine. Label = clipped RUL at the window's
    last cycle. Returns X (N, window, n_features) and y (N,)."""
    X, y = [], []
    for _, g in df.groupby("Engine Number"):
        g = g.sort_values("Cycle")
        feats = g[sensor_cols].values
        ruls  = g["RUL"].values
        # one window for every position where a full window fits
        for i in range(len(g) - window + 1):
            X.append(feats[i:i + window])
            y.append(ruls[i + window - 1])      # RUL at the end of the window
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def make_test_windows(df, sensor_cols, window=WINDOW):
    """Take the LAST window of each engine (that's where we must predict RUL).
    Engines shorter than `window` are front-padded by repeating their first row."""
    X = []
    for _, g in df.groupby("Engine Number"):
        g = g.sort_values("Cycle")
        feats = g[sensor_cols].values
        if len(feats) < window:
            pad = np.repeat(feats[:1], window - len(feats), axis=0)
            feats = np.vstack([pad, feats])
        X.append(feats[-window:])               # most recent `window` cycles
    return np.asarray(X, dtype=np.float32)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------- Model ----------------------------------------
class CNN_BiLSTM(nn.Module):
    """1-D CNN extracts local degradation patterns across the window;
    a BiLSTM models how those patterns evolve in time; a small MLP head
    regresses the RUL.

    NOTE: BatchNorm after each conv is important here. Without it, the
    unnormalized ReLU activations feeding the LSTM make it collapse to
    predicting the mean RUL (train RMSE stuck ~= target std)."""

    def __init__(self, n_features, cnn_ch=64, lstm_hidden=96, lstm_layers=1):
        super().__init__()
        # CNN feature extractor (operates on (batch, channels=features, time))
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, cnn_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_ch),
            nn.ReLU(),
            nn.Conv1d(cnn_ch, cnn_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_ch),
            nn.ReLU(),
        )
        # Bidirectional LSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=cnn_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        # Regression head (2 * lstm_hidden because BiLSTM concatenates directions)
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):                 # x: (batch, window, features)
        x = x.transpose(1, 2)             # -> (batch, features, window) for Conv1d
        x = self.cnn(x)                   # -> (batch, cnn_ch, window)
        x = x.transpose(1, 2)             # -> (batch, window, cnn_ch) for LSTM
        out, _ = self.lstm(x)             # -> (batch, window, 2*hidden)
        out = out[:, -1, :]               # last timestep's representation
        return self.head(out).squeeze(-1)  # -> (batch,)


# ------------------------------ Train ---------------------------------------
def train_model(model, loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # stabilize LSTM
            opt.step()
            running += loss.item() * len(xb)
        rmse = (running / len(loader.dataset)) ** 0.5
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  train RMSE {rmse:6.2f}")
    return model


# ------------------------------ Main ----------------------------------------
def main(train_file_name="train_FD001.txt"):
    # 1. Load + RUL + drop dead sensors (same logic as baseline.py)
    train = data_load(train_file_name)
    train = compute_rul(train)

    dead_sensors = get_dead_sensors(train)
    train = train.drop(columns=dead_sensors)

    test_file_name = train_file_name.replace("train", "test")
    test = data_load(test_file_name)
    test = test.drop(columns=dead_sensors)

    sensor_cols = [c for c in train.columns if "Sensor" in c]

    # 2. Piecewise-linear RUL cap on the TRAIN labels
    train["RUL"] = train["RUL"].clip(upper=MAX_RUL)

    # 3. Min-Max scale sensors (fit on train, transform test)
    scaler = MinMaxScaler()
    train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
    test[sensor_cols]  = scaler.transform(test[sensor_cols])

    # 4. Build windows
    X_train, y_train = make_train_windows(train, sensor_cols, WINDOW)
    X_test = make_test_windows(test, sensor_cols, WINDOW)
    print(f"train windows: {X_train.shape}  test windows: {X_test.shape}  "
          f"(features={len(sensor_cols)})")

    loader = DataLoader(SeqDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

    # 5. Train
    model = CNN_BiLSTM(n_features=len(sensor_cols))
    print(f"Training CNN-BiLSTM on {DEVICE} ...")
    train_model(model, loader)

    # 6. Predict last window per engine, clip to [0, MAX_RUL], and score
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
    y_pred = np.clip(y_pred, 0, MAX_RUL)

    RUL_file_name = train_file_name.replace("train", "RUL")
    y_true = pd.read_csv(project_dir / "CMAPSSData" / RUL_file_name,
                         header=None, names=["RUL"])["RUL"].values

    print("\n=== CNN-BiLSTM results ===")
    score(y_pred, y_true)


if __name__ == "__main__":
    main("train_FD001.txt")
