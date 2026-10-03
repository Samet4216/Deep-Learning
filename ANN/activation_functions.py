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

def softmax(z, alpha=None):
    z_stable = z - np.max(z, axis=1, keepdims=True) #[2000, 2001, 2002] - 2002 = [-2, -1, 0] # subtracting the max value for numerical stability
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True) #keepdims=True to maintain the same shape for broadcasting

def softmax_derivative(z, alpha=None):
    return 1