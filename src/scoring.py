# handles all scoring related tasks
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

def score(computed, actual): #lower score for both is better
    #RSME METHOD
    rmse = np.sqrt(mean_squared_error(actual, computed))
    print(f"Baseline Linear Regression RMSE: {rmse:.2f}")

    #NASA METHOD
    d = computed - actual 
    total_score = 0 

    for error in d:
        if error < 0:
            total_score += np.exp(-error / 13) - 1 # early prediction, light
        else:
            total_score += np.exp(error / 10) - 1   # late prediction, heavy

    print(f"Baseline NASA Score: {total_score:.2f}")