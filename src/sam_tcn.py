"""
SAM-TCN model (Self-Attention Mechanism + Temporal Convolutional Network)
for Remaining Useful Life (RUL) prediction on NASA CMAPSS.

This is one of the three SOTA methods in the project. It mirrors the structure
of baseline.py and reuses data_loader.py and scoring.py.

Architecture:
    - Temporal Convolutional Network (TCN) with dilated convolutions and 
      residual connections for efficient temporal feature extraction.
    - Self-Attention Mechanism (SAM) to learn long-range dependencies across 
      time steps.
    - Regression head for RUL prediction.

Pipeline:
    1. Load train + test, compute RUL on train, drop "dead" (constant) sensors.
    2. Clip RUL with a piecewise-linear cap (standard trick for CMAPSS).
    3. Min-Max scale sensors (fit on train, apply to test).
    4. Build sliding windows of length WINDOW.
    5. Train SAM-TCN regressor.
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
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for temporal dependency modeling."""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Combine heads
        x_attn = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        return x_attn


class TemporalConvBlock(nn.Module):
    """Residual temporal convolutional block with dilation."""
    
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, 
            padding=pad, dilation=dilation, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, 
            padding=pad, dilation=dilation, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection adjustment (1x1 conv if channels change)
        self.residual = nn.Identity()
        if in_ch != out_ch:
            self.residual = nn.Conv1d(in_ch, out_ch, 1, bias=False)
    
    def forward(self, x):
        # x: (batch, in_ch, seq_len)
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        out = self.dropout(out)
        
        return out


class SAM_TCN(nn.Module):
    """
    Self-Attention Mechanism + Temporal Convolutional Network.
    
    Combines dilated TCN for capturing multi-scale temporal patterns with
    self-attention for learning long-range dependencies. The model efficiently
    processes sensor time-series to predict RUL.
    
    Args:
        n_features: Number of input sensor features
        tcn_channels: List of channel sizes for each TCN block (default: [64, 64, 64])
        tcn_kernel_size: Kernel size for temporal convolutions (default: 3)
        attention_dim: Embedding dimension for self-attention (default: 64)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.2)
    """
    
    def __init__(
        self, 
        n_features, 
        tcn_channels=[64, 64, 64], 
        tcn_kernel_size=3,
        attention_dim=64,
        num_heads=8,
        dropout=0.2
    ):
        super().__init__()
        
        # Initial projection to TCN channel size
        self.proj_in = nn.Linear(n_features, tcn_channels[0])
        
        # TCN backbone with increasing dilation rates
        self.tcn_blocks = nn.ModuleList()
        for i, out_ch in enumerate(tcn_channels):
            in_ch = tcn_channels[i-1] if i > 0 else tcn_channels[0]
            dilation = 2 ** i  # exponentially increasing dilation: 1, 2, 4, ...
            block = TemporalConvBlock(
                in_ch, out_ch, 
                kernel_size=tcn_kernel_size,
                dilation=dilation,
                dropout=dropout
            )
            self.tcn_blocks.append(block)
        
        # Self-attention layer (operates on (batch, seq_len, attention_dim))
        self.to_attn = nn.Linear(tcn_channels[-1], attention_dim)
        self.attention = MultiHeadSelfAttention(attention_dim, num_heads, dropout)
        self.from_attn = nn.Linear(attention_dim, tcn_channels[-1])
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(tcn_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    
    def forward(self, x):
        # x: (batch, window, n_features)
        batch_size, seq_len, n_features = x.shape
        
        # Project input to TCN dimension
        x = self.proj_in(x)  # (batch, seq_len, tcn_channels[0])
        x = x.transpose(1, 2)  # (batch, tcn_channels[0], seq_len) for Conv1d
        
        # Apply TCN blocks
        for tcn_block in self.tcn_blocks:
            x = tcn_block(x)  # (batch, out_ch, seq_len)
        
        # Transpose back to (batch, seq_len, channels) for attention
        x = x.transpose(1, 2)  # (batch, seq_len, tcn_channels[-1])
        
        # Self-attention: project -> attention -> project back
        attn_in = self.to_attn(x)  # (batch, seq_len, attention_dim)
        attn_out = self.attention(attn_in)  # (batch, seq_len, attention_dim)
        x = x + self.from_attn(attn_out)  # residual connection + (batch, seq_len, tcn_channels[-1])
        
        # Use last timestep for prediction
        x_last = x[:, -1, :]  # (batch, tcn_channels[-1])
        
        # Regression head
        out = self.head(x_last)  # (batch, 1)
        
        return out.squeeze(-1)  # (batch,)


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
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
    model = SAM_TCN(n_features=len(sensor_cols))
    print(f"Training SAM-TCN on {DEVICE} ...")
    train_model(model, loader)

    # 6. Predict last window per engine, clip to [0, MAX_RUL], and score
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_test).to(DEVICE)).cpu().numpy()
    y_pred = np.clip(y_pred, 0, MAX_RUL)

    RUL_file_name = train_file_name.replace("train", "RUL")
    y_true = pd.read_csv(project_dir / "CMAPSSData" / RUL_file_name,
                         header=None, names=["RUL"])["RUL"].values

    print("\n=== SAM-TCN results ===")
    score(y_pred, y_true)


if __name__ == "__main__":
    main("train_FD001.txt")
