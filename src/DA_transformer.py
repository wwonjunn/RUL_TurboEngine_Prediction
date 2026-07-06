# dataset.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from data_loader import data_load, compute_rul, get_dead_sensors, make_windows, make_test_windows
from scoring import score

class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        # convert numpy arrays to PyTorch tensors
        # float32 is standard for neural network inputs
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)  # how many samples total

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # one sample + its label
    

class TransformerRUL(nn.Module):
    def __init__(self, num_sensors, d_model=64, nhead=4, num_layers=2, window_size=30):
        super().__init__()

        # project 14 sensors → 64 dimensional space
        self.input_proj = nn.Linear(num_sensors, d_model)

        # learned position signal, one vector per timestep
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, d_model))

        # multi-head self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,    # expects [batch, seq, features] not [seq, batch, features]
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # squeeze [64] → [32] → [1] = RUL prediction
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: [batch, 30, 14]
        x = self.input_proj(x)          # [batch, 30, 64]
        x = x + self.pos_embedding      # [batch, 30, 64] — add position info
        x = self.transformer(x)         # [batch, 30, 64] — self-attention
        x = x[:, -1, :]                 # [batch, 64] — last timestep only
        out = self.output_head(x)       # [batch, 1]
        return out.squeeze(-1)          # [batch]
    

project_dir = Path(__file__).parent.parent
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# load and preprocess — identical pipeline to baseline
train_file_name = "train_FD001.txt"
train = data_load(train_file_name)
train = compute_rul(train)

dead_sensors = get_dead_sensors(train)
train = train.drop(columns=dead_sensors)

test_file_name = train_file_name.replace("train", "test")
test = data_load(test_file_name)
test = test.drop(columns=dead_sensors)

sensor_cols = [c for c in train.columns if 'Sensor' in c]

scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
test[sensor_cols] = scaler.transform(test[sensor_cols])

# build windows
X_train, y_train = make_windows(train, sensor_cols, WINDOW_SIZE)
X_test = make_test_windows(test, sensor_cols, WINDOW_SIZE)

RUL_file_name = train_file_name.replace("train", "RUL")
y_true = pd.read_csv(project_dir / "CMAPSSData" / RUL_file_name,
                     header=None, names=["RUL"])["RUL"].values

# cap RUL at 125 — focuses learning on degradation phase
y_train = np.minimum(y_train, 125)

train_dataset = CMAPSSDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = TransformerRUL(num_sensors=len(sensor_cols), window_size=WINDOW_SIZE).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()       # clear gradients from last batch
        y_pred = model(X_batch)     # forward pass
        loss = criterion(y_pred, y_batch)  # compare to true RUL
        loss.backward()             # compute gradients
        optimizer.step()            # update weights

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

# evaluate
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred = model(X_test_tensor).cpu().numpy()

score(y_pred, y_true)