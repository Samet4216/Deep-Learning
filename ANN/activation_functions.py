import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    y_pred = sigmoid(z)
    return y_pred * (1 - y_pred)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(np.asarray(z) > 0, 1.0, 0.0) # np.where(condition, value_if_true, value_if_false)

def leaky_relu(z, alpha):
    return np.where(np.asarray(z) > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha):
    return np.where(np.asarray(z) > 0, 1, alpha)