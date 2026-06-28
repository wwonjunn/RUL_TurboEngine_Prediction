import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from data_loader import data_load, compute_rul, get_dead_sensors
from scoring import score

project_dir = Path(__file__).parent.parent

train_file_name = "train_FD001.txt"
train = data_load(train_file_name)
train = compute_rul(train)

# get dead sensors from train only, apply to both
dead_sensors = get_dead_sensors(train)
train = train.drop(columns=dead_sensors)

test_file_name = train_file_name.replace("train", "test")
test = data_load(test_file_name)
test = test.drop(columns=dead_sensors) # same list derived from train

sensor_cols = [c for c in train.columns if 'Sensor' in c]

# fit scaler on train only, apply same scaler to test
scaler = MinMaxScaler()
train[sensor_cols] = scaler.fit_transform(train[sensor_cols])
test[sensor_cols] = scaler.transform(test[sensor_cols])

X_train = train[sensor_cols]
y_train = train["RUL"]

model = LinearRegression()
model.fit(X_train, y_train)

# predict only on last observed cycle per engine
last_rows = test.groupby("Engine Number").last().reset_index() #each engine resets 
X_test = last_rows[sensor_cols]
y_pred = model.predict(X_test)

RUL_file_name = train_file_name.replace("train", "RUL")
y_true = pd.read_csv(project_dir / "CMAPSSData" / RUL_file_name,
                     header=None, names=["RUL"])["RUL"].values

score(y_pred, y_true)
