import pandas as pd
import numpy as np
from pathlib import Path

project_dir = Path(__file__).parent.parent

def data_load(file_name):
    # Column 1: Engine unit number (unique identifier)
    # Column 2: Time cycle (operational cycle number)
    # Columns 3-5: Operational settings (flight altitude, throttle resolver angle, etc.)
    # Columns 6-26: Sensor measurements (temperatures, pressures, speeds, ratios)
    cols = ["Engine Number", "Cycle"] + \
           [f'Operation Setting {i}' for i in range(1, 4)] + \
           [f'Sensor Measurement {i}' for i in range(1, 22)]
    df = pd.read_csv(project_dir / "CMAPSSData" / file_name,
                     sep=r'\s+', header=None, names=cols)
    return df

def compute_rul(df):
    max_cycle = df.groupby("Engine Number")["Cycle"].transform("max")
    df["RUL"] = max_cycle - df["Cycle"]
    return df

def get_dead_sensors(df, threshold=0.01):
    # by std() this drops any sensors that basically don't fluctuate at all
    sensor_cols = [f'Sensor Measurement {i}' for i in range(1, 22)]
    std = df[sensor_cols].std()
    return std[std < threshold].index.tolist()

def get_feature_cols(df):
    # includes operation settings so model can condition on flight state
    # critical for FD002/FD004 which have 6 operating conditions
    return [c for c in df.columns if 'Sensor' in c or 'Operation' in c]

def make_windows(df, feature_cols, window_size=30):
    X, y = [], []
    for engine_id, group in df.groupby("Engine Number"):
        group = group.sort_values("Cycle")
        features = group[feature_cols].values
        ruls = group["RUL"].values

        if len(group) < window_size:
            continue

        for i in range(len(group) - window_size + 1):
            X.append(features[i:i + window_size])
            y.append(ruls[i + window_size - 1])

    return np.array(X), np.array(y)

def make_test_windows(df, feature_cols, window_size=30):
    X = []
    for engine_id, group in df.groupby("Engine Number"):
        group = group.sort_values("Cycle")
        features = group[feature_cols].values

        if len(features) < window_size:
            # pad short engines by repeating first row
            pad = np.repeat(features[0:1], window_size - len(features), axis=0)
            features = np.vstack([pad, features])

        X.append(features[-window_size:])

    return np.array(X)