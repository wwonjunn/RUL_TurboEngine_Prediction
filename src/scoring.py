# handles all scoring related tasks
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

def score(computed, actual):
    rmse = np.sqrt(mean_squared_error(actual, computed))
    print(f"RMSE: {rmse:.2f}")

    d = computed - actual
    total_score = 0
    for error in d:
        if error < 0:
            total_score += np.exp(-error / 13) - 1
        else:
            total_score += np.exp(error / 10) - 1

    print(f"NASA Score: {total_score:.2f}")
    return rmse, total_score 