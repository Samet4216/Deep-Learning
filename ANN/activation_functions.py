import numpy as np

def sigmoid(z, alpha=None):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z, alpha=None):
    y_pred = sigmoid(z, alpha)
    return y_pred * (1 - y_pred)

def relu(z, alpha=None):
    return np.maximum(0, z)

def relu_derivative(z, alpha=None):
    return np.where(np.asarray(z) > 0, 1.0, 0.0) # np.where(condition, value_if_true, value_if_false)

def leaky_relu(z, alpha=None):
    alpha = 0.01 if alpha is None else alpha
    return np.where(np.asarray(z) > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=None):
    alpha = 0.01 if alpha is None else alpha
    return np.where(np.asarray(z) > 0, 1, alpha)

def linear(z, alpha=None):
    return z

def linear_derivative(z, alpha=None):
    return np.ones_like(z)