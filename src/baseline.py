import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

project_dir = Path(__file__).parent.parent

# Column 1: Engine unit number (unique identifier)
# Column 2: Time cycle (operational cycle number)
# Columns 3-5: Operational settings (flight altitude, throttle resolver angle, etc.)
# Columns 6-26: Sensor measurements (temperatures, pressures, speeds, ratios)
cols = ["Engine Number", "Cycle"] + \
       [f'Operation Setting {i}' for i in range(1, 4)] + \
       [f'Sensor Measurement {i}' for i in range(1, 22)]

train = pd.read_csv(project_dir / "CMAPSSData" / "train_FD001.txt",
                    sep=r'\s+', header=None, names=cols)

max_cycle = train.groupby("Engine Number")["Cycle"].transform("max")
train["RUL"] = max_cycle - train["Cycle"]

sensor_cols = [f'Sensor Measurement {i}' for i in range(1, 22)]
std = train[sensor_cols].std()
threshold = 0.01 # by std() this drops any sensors that basically don't fluctuate at all
dead_sensors = std[std < threshold].index.tolist()
train = train.drop(columns=dead_sensors)

sensor_cols = [c for c in train.columns if 'Sensor' in c]

scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])

X_train = train[sensor_cols]
y_train = train["RUL"]

model = LinearRegression()
model.fit(X_train, y_train)

test = pd.read_csv(project_dir / "CMAPSSData" / "test_FD001.txt",
                   sep=r'\s+', header=None, names=cols)

# apply same dead sensor drops and scaler from train, never refit on test
test = test.drop(columns=dead_sensors)
test[sensor_cols] = scaler.transform(test[sensor_cols])

# predict only on last observed cycle per engine
last_rows = test.groupby("Engine Number").last().reset_index()
X_test = last_rows[sensor_cols]
y_pred = model.predict(X_test)

y_true = pd.read_csv(project_dir / "CMAPSSData" / "RUL_FD001.txt",
                     header=None, names=["RUL"])["RUL"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Baseline Linear Regression RMSE: {rmse:.2f}")

# NASA asymmetric scoring function (Saxena et al. 2008, eq. 11)
def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    score = sum(
        np.exp(-di / 13) - 1 if di < 0
        else np.exp(di / 10) - 1
        for di in d
    )
    return score

score = nasa_score(y_true, y_pred)
print(f"Baseline NASA Score: {score:.2f}")