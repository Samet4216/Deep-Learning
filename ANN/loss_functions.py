import numpy as np


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)