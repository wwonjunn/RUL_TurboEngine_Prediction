import pandas as pd
from pathlib import Path
import numpy as np

def data_load(file_name):
    project_dir = Path(__file__).parent.parent
    # Column 1: Engine unit number (unique identifier)
    # Column 2: Time cycle (operational cycle number)
    # Columns 3-5: Operational settings (flight altitude, throttle resolver angle, etc.)
    # Columns 6-26: Sensor measurements (temperatures, pressures, speeds, ratios)
    cols = ["Engine Number", "Cycle"] + \
           [f'Operation Setting {i}' for i in range(1, 4)] + \
           [f'Sensor Measurement {i}' for i in range(1, 22)]
    df = pd.read_csv(project_dir / "CMAPSSData" / file_name, sep=r'\s+', header=None, names=cols)
    return df

def compute_rul(train):
    max_cycle = train.groupby("Engine Number")["Cycle"].transform("max")
    train["RUL"] = max_cycle - train["Cycle"]
    return train

def get_dead_sensors(train, threshold=0.01):
    # by std() this drops any sensors that basically don't fluctuate at all
    sensor_cols = [f'Sensor Measurement {i}' for i in range(1, 22)]
    std = train[sensor_cols].std()
    return std[std < threshold].index.tolist()


# Jun - for DA transformer - making windows for training
# upon literature searching ~windows of 30 is pretty standard so lets use 30 for now

def make_windows(df, sensor_cols, window_size=30):
    X, y = [], []
    for engine_id, group in df.groupby("Engine Number"):
        group = group.sort_values("Cycle")
        sensors = group[sensor_cols].values  # shape [num_cycles x 14]
        ruls = group["RUL"].values

        if len(group) < window_size:
            continue  # engine too short, skip entirely

        for i in range(len(group) - window_size + 1):
            X.append(sensors[i:i + window_size])       # 30 rows of sensors
            y.append(ruls[i + window_size - 1])        # RUL at last cycle of window

    return np.array(X), np.array(y)  # shapes: [N x 30 x 14], [N]

def make_test_windows(df, sensor_cols, window_size=30):
    X = []
    for engine_id, group in df.groupby("Engine Number"):
        group = group.sort_values("Cycle")
        sensors = group[sensor_cols].values

        if len(sensors) < window_size:
            # pad short engines by repeating first row
            pad = np.repeat(sensors[0:1], window_size - len(sensors), axis=0)
            sensors = np.vstack([pad, sensors])

        X.append(sensors[-window_size:])  # always take the last 30 cycles

    return np.array(X)  # shape: [num_test_engines x 30 x 14]