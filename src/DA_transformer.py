# does not include domain part here.
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from data_loader import data_load, compute_rul, get_dead_sensors, make_windows, make_test_windows
from scoring import score

class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerRUL(nn.Module):
    def __init__(self, num_sensors, d_model=64, nhead=4, num_layers=2, window_size=30):
        super().__init__()
        self.input_proj = nn.Linear(num_sensors, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.output_head(x).squeeze(-1)


project_dir = Path(__file__).parent.parent
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 128        # doubled from 64 — fewer batches per epoch, faster
LEARNING_RATE = 1e-4   # lowered from 1e-3 — fixes the flatline on FD002

train_file_name = "train_FD004.txt"
train = data_load(train_file_name)
train = compute_rul(train)

dead_sensors = get_dead_sensors(train)
train = train.drop(columns=dead_sensors)

test_file_name = train_file_name.replace("train", "test")
test = data_load(test_file_name)
test = test.drop(columns=dead_sensors)

# include operation settings as features — helps model condition on flight state
# critical for FD002 which has 6 operating conditions
feature_cols = [c for c in train.columns if 'Sensor' in c or 'Operation' in c]

scaler = MinMaxScaler()
train[feature_cols] = scaler.fit_transform(train[feature_cols])
test[feature_cols] = scaler.transform(test[feature_cols])

X_train, y_train = make_windows(train, feature_cols, WINDOW_SIZE)
X_test = make_test_windows(test, feature_cols, WINDOW_SIZE)

RUL_file_name = train_file_name.replace("train", "RUL")
y_true = pd.read_csv(project_dir / "CMAPSSData" / RUL_file_name,
                     header=None, names=["RUL"])["RUL"].values

y_train = np.minimum(y_train, 125)

print(f"X_train shape: {X_train.shape}")
print(f"y_train max: {y_train.max()}, mean: {y_train.mean():.1f}")

train_dataset = CMAPSSDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# M4 Mac — use MPS, fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")

model = TransformerRUL(num_sensors=len(feature_cols), window_size=WINDOW_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {total_loss/len(train_loader):.4f}")

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred = model(X_test_tensor).cpu().numpy()

score(y_pred, y_true)