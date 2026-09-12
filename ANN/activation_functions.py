import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivate(z):
    y_pred = sigmoid(z)
    return y_pred * (1 - y_pred)

def relu(z):
    return np.maximum(0, z)

def relu_derivate(z):
    return np.where(np.asarray(z) > 0, 1.0, 0.0) #np.where(koşul, koşul_doğruysa, koşul_yanlışsa) 
