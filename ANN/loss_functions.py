import numpy as np


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)  #clip predictions to avoid log(0)
    sample_losses = -np.sum(y_true * np.log(y_pred), axis=1) #compute the loss for each sample
    batch_loss = np.mean(sample_losses)
    return batch_loss

def categorical_cross_entropy_derivative(y_true, y_pred):
    N = y_true.shape[0]
    return (y_pred-y_true) / N

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15) 
    return -(np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
# ex. A=[0.80, 0.20] Target=[1,0] => for 0.80 and 1 => -[y*log(p) + (1-y)*log(1-p)] = -[1*log(0.80) + (1-1)*log(1-0.80)] = 0.22314355 loss point

def binary_cross_entropy_derivative(y_true, y_pred):
    N = y_true.shape[0]
    return (y_pred-y_true) / N