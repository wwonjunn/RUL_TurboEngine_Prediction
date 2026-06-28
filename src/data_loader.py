import pandas as pd
from pathlib import Path

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