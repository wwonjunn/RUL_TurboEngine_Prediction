"""
CNN-BiLSTM model for Remaining Useful Life (RUL) prediction on NASA CMAPSS.

One of the three SOTA methods in the project. Uses the SHARED pipeline in
data_loader.py (loading / RUL / dead sensors / windowing) and scoring.py
(RMSE + NASA score), so the only thing that differs between this and the
Transformer is the model in Stage 4.

Pipeline:
    1. Load train + test, compute RUL on train, drop "dead" (constant) sensors.
    2. Feature columns = 3 operation settings + surviving sensors.
    3. Min-Max scale features (fit on train, apply to test).
    4. Clip RUL with a piecewise-linear cap (standard trick for CMAPSS).
    5. Sliding windows of length WINDOW so the net sees a short time-history.
    6. Train CNN -> BiLSTM -> Dense regressor.
    7. Predict one RUL per test engine from its LAST window, then score.

Run:
    python cnn_bilstm.py              # all four sub-datasets, prints a table
    python cnn_bilstm.py FD001        # just one
    python cnn_bilstm.py FD001 FD003  # a subset
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

from data_loader import (data_load, compute_rul, get_dead_sensors,
                         get_feature_cols, make_windows, make_test_windows)
from scoring import score

# ----------------------------- Config --------------------------------------
project_dir = Path(__file__).parent.parent

WINDOW     = 30      # cycles fed to the model at once
MAX_RUL    = 125     # piecewise-linear RUL cap (matches DA_transformer.py)
BATCH_SIZE = 256
EPOCHS     = 20
LR         = 1e-3
SEED       = 42

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

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

    def forward(self, x):                  # x: (batch, window, features)
        x = x.transpose(1, 2)              # -> (batch, features, window) for Conv1d
        x = self.cnn(x)                    # -> (batch, cnn_ch, window)
        x = x.transpose(1, 2)              # -> (batch, window, cnn_ch) for LSTM
        out, _ = self.lstm(x)              # -> (batch, window, 2*hidden)
        out = out[:, -1, :]                # last timestep's representation
        return self.head(out).squeeze(-1)  # -> (batch,)


# ------------------------------ Train ---------------------------------------
def train_model(model, loader, epochs=EPOCHS, lr=LR):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rmse = float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # stabilize LSTM
            opt.step()
            running += loss.item() * len(xb)
        rmse = (running / len(loader.dataset)) ** 0.5
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  epoch {epoch:3d}/{epochs}  train RMSE {rmse:6.2f}")
    return model, rmse   # final train RMSE is reported in the paper


# ------------------------------ One dataset ---------------------------------
def run_one(fd_name="FD001"):
    set_seed()
    print(f"\n{'='*60}\nCNN-BiLSTM  |  {fd_name}\n{'='*60}")

    # 1. Load + RUL + drop dead sensors (identical to baseline / DA_transformer)
    train = data_load(f"train_{fd_name}.txt")
    train = compute_rul(train)

    dead_sensors = get_dead_sensors(train)
    train = train.drop(columns=dead_sensors)

    test = data_load(f"test_{fd_name}.txt")
    test = test.drop(columns=dead_sensors)      # same list derived from train

    # 2. Operation settings + surviving sensors
    feature_cols = get_feature_cols(train)

    # 3. Min-Max scale (fit on train, transform test)
    scaler = MinMaxScaler()
    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    # 4 + 5. Windows, then cap the TRAIN labels
    X_train, y_train = make_windows(train, feature_cols, WINDOW)
    X_test = make_test_windows(test, feature_cols, WINDOW)
    y_train = np.minimum(y_train, MAX_RUL)

    print(f"features={len(feature_cols)} "
          f"(dead sensors dropped: {len(dead_sensors)})")
    print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

    loader = DataLoader(SeqDataset(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

    # 6. Train
    print(f"Training on: {DEVICE}")
    model = CNN_BiLSTM(n_features=len(feature_cols))
    model, train_rmse = train_model(model, loader)

    # 7. Predict last window per engine, clip to [0, MAX_RUL], score
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_pred = model(X_tensor).cpu().numpy()
    y_pred = np.clip(y_pred, 0, MAX_RUL)

    y_true = pd.read_csv(project_dir / "CMAPSSData" / f"RUL_{fd_name}.txt",
                         header=None, names=["RUL"])["RUL"].values

    assert len(y_pred) == len(y_true), \
        f"{fd_name}: {len(y_pred)} predictions vs {len(y_true)} ground-truth RULs"

    print(f"\nCNN-BiLSTM {fd_name}:")
    rmse, nasa = score(y_pred, y_true)

    # how many predictions overestimate RUL? these drive the NASA score
    late = int(np.sum(y_pred - y_true > 0))
    print(f"late predictions: {late}/{len(y_true)}")

    return {"dataset": fd_name, "n_features": len(feature_cols),
            "train_rmse": round(train_rmse, 2), "rmse": round(rmse, 2),
            "nasa": round(nasa, 2), "late": late, "n_engines": len(y_true)}


# ------------------------------ Main ----------------------------------------
def main(datasets=DATASETS):
    results = [run_one(fd) for fd in datasets]

    print(f"\n{'='*76}")
    print("CNN-BiLSTM SUMMARY  (paste this to Claude for the results table)")
    print(f"{'='*76}")
    print(f"{'Dataset':<10}{'Feats':>7}{'TrainRMSE':>12}{'TestRMSE':>11}"
          f"{'NASA Score':>16}{'Late':>10}")
    print("-" * 76)
    for r in results:
        print(f"{r['dataset']:<10}{r['n_features']:>7}{r['train_rmse']:>12.2f}"
              f"{r['rmse']:>11.2f}{r['nasa']:>16.2f}"
              f"{str(r['late']) + '/' + str(r['n_engines']):>10}")
    print("=" * 76)

    out = project_dir / "cnn_bilstm_results.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"saved -> {out}")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args if args else DATASETS)
