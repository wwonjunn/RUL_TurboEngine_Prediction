import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from data_loader import data_load, compute_rul, get_dead_sensors, get_feature_cols, make_windows, make_test_windows
from scoring import score

project_dir = Path(__file__).parent.parent

# ── DATASET ───────────────────────────────────────────────────────────────────

class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── GRADIENT REVERSAL ─────────────────────────────────────────────────────────

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


# ── MODEL ─────────────────────────────────────────────────────────────────────

class TransformerDAT(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, window_size=30):
        super().__init__()
        self.input_proj = nn.Linear(num_features, d_model)
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
        self.domain_discriminator = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # source=0, target=1
        )

    def forward(self, x, alpha=1.0):
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer(x)
        features = x[:, -1, :]
        rul_pred = self.output_head(features).squeeze(-1)
        domain_pred = self.domain_discriminator(grad_reverse(features, alpha))
        return rul_pred, domain_pred


# ── CONFIG ────────────────────────────────────────────────────────────────────

WINDOW_SIZE = 30
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE = 5e-5
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Training on: {device}")


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_and_preprocess(fd_name, dead_sensors=None, scaler=None, feature_cols=None, is_source=True):
    df = data_load(f"train_{fd_name}.txt")
    df = compute_rul(df)

    if dead_sensors is None:
        dead_sensors = get_dead_sensors(df)
    df = df.drop(columns=dead_sensors)

    if feature_cols is None:
        feature_cols = get_feature_cols(df)

    if scaler is None:
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    X, y = make_windows(df, feature_cols, WINDOW_SIZE)
    if is_source:
        y = np.minimum(y, 150)

    return X, y, dead_sensors, scaler, feature_cols


def load_test(fd_name, dead_sensors, scaler, feature_cols):
    test = data_load(f"test_{fd_name}.txt")
    test = test.drop(columns=dead_sensors)
    test[feature_cols] = scaler.transform(test[feature_cols])
    X_test = make_test_windows(test, feature_cols, WINDOW_SIZE)
    y_true = pd.read_csv(project_dir / "CMAPSSData" / f"RUL_{fd_name}.txt",
                         header=None, names=["RUL"])["RUL"].values
    return X_test, y_true


def train_dat(source, target, device, dead_sensors, scaler, feature_cols):
    X_src, y_src, _, _, _ = load_and_preprocess(
        source, dead_sensors, scaler, feature_cols, is_source=True)
    X_tgt, _, _, _, _ = load_and_preprocess(
        target, dead_sensors, scaler, feature_cols, is_source=False)

    source_loader = DataLoader(
        CMAPSSDataset(X_src, y_src),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    target_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tgt, dtype=torch.float32),
            torch.ones(len(X_tgt), dtype=torch.long)  # domain label = 1
        ),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    model = TransformerDAT(num_features=len(feature_cols), window_size=WINDOW_SIZE).to(device)
    rul_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        # alpha schedule from DANN paper — ramps 0→1 over training
        # RUL learning stabilizes first, domain adaptation kicks in gradually
        alpha = 2 / (1 + np.exp(-10 * epoch / EPOCHS)) - 1

        model.train()
        total_rul_loss = 0
        total_domain_loss = 0

        for (X_s, y_s), (X_t, d_t) in zip(source_loader, target_loader):
            X_s, y_s = X_s.to(device), y_s.to(device)
            X_t, d_t = X_t.to(device), d_t.to(device)
            d_src = torch.zeros(len(X_s), dtype=torch.long).to(device)

            optimizer.zero_grad()

            rul_pred, dp_src = model(X_s, alpha=alpha)
            _, dp_tgt = model(X_t, alpha=alpha)

            rul_loss = rul_criterion(rul_pred, y_s)
            domain_loss = domain_criterion(dp_src, d_src) + \
                          domain_criterion(dp_tgt, d_t)

            (rul_loss + domain_loss).backward()
            optimizer.step()

            total_rul_loss += rul_loss.item()
            total_domain_loss += domain_loss.item()

        print(f"  [{source}->{target}] Epoch {epoch+1}/{EPOCHS} "
              f"RUL: {total_rul_loss/len(source_loader):.4f} "
              f"Domain: {total_domain_loss/len(source_loader):.4f} "
              f"Alpha: {alpha:.3f}")

    return model


# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

results = {}

for source in DATASETS:
    print(f"\n{'='*50}")
    print(f"SOURCE: {source}")
    print(f"{'='*50}")

    _, _, dead_sensors, scaler, feature_cols = load_and_preprocess(source)

    targets = [fd for fd in DATASETS if fd != source]
    for target in targets:
        input(f"\nPress Enter to start DAT {source} → {target}...")

        print(f"\n[DAT] {source} → {target}")
        model = train_dat(source, target, device, dead_sensors, scaler, feature_cols)

        model.eval()
        X_test, y_true = load_test(target, dead_sensors, scaler, feature_cols)
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            y_pred, _ = model(X_tensor)
            y_pred = y_pred.cpu().numpy()

        # debug
        print(f"y_pred min: {y_pred.min():.2f}, max: {y_pred.max():.2f}, mean: {y_pred.mean():.2f}")
        print(f"y_true min: {y_true.min():.2f}, max: {y_true.max():.2f}, mean: {y_true.mean():.2f}")
        print(f"X_test shape: {X_test.shape}")
        print(f"num predictions: {len(y_pred)}, num true: {len(y_true)}")
        print(f"\nDAT {source}→{target}:")
        rmse, nasa = score(y_pred, y_true)
        results[f"{source}→{target}"] = (rmse, nasa)

        print(f"\nResults so far:")
        print(f"{'Task':<20} {'RMSE':>10} {'NASA Score':>12}")
        print("-" * 44)
        for task, (r, n) in results.items():
            print(f"{task:<20} {r:>10.2f} {n:>12.2f}")


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────

print(f"\n{'='*50}")
print("FINAL RESULTS SUMMARY")
print(f"{'='*50}")
print(f"{'Task':<20} {'RMSE':>10} {'NASA Score':>12}")
print("-" * 44)
for task, (rmse, nasa) in results.items():
    print(f"{task:<20} {rmse:>10.2f} {nasa:>12.2f}")