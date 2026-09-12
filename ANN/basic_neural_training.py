"""
def neuron(x1,x2,x3,w1,w2,w3,b):
    z=(x1*w1)+(x2*w2)+(x3*w3)+b
    return z

input=neuron(
    2,#x1
    3,#x2
    4,#x3
    5,#w1
    6,#w2
    0,#w3
    20,#b
)
print(input)
"""

import numpy as np
import random 
import activation_functions as af

class Neuron():

    def __init__(self, input_size, activation, derivative, alpha=0.01):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.1 # A negative initial bias can cause ReLU units to stop learning.
        self.activation = getattr(af, activation) # getattr receives the module and function name.
        self.derivative = getattr(af, derivative) # Activation and derivative names are passed as strings.
        self.alpha = alpha

    def forward(self,input):
        self.z = np.dot(input,self.weights)+self.bias # z = (x * w) + b
        self.output = self.activation(self.z)
        return self.output

    # def mse_loss(y_true, y_pred):

    def gradient(self, x, y_true):

        y_pred = self.output # Prediction returned by the activation function.

        dL_dy_pred = -2 * (y_true - y_pred) # Derivative of the loss with respect to the prediction.
        activation_func_der = self.derivative(self.z)

        # Chain rule
        ortak_turevler = dL_dy_pred * activation_func_der
        gradient = ortak_turevler * x #(dz/dw)

        bias_gradient = ortak_turevler * 1 # dz/db is 1.

        return gradient, bias_gradient

    def update(self, gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * bias_gradient

def mse_loss(y_true, y_pred):
    loss=(y_true-y_pred)**2
    return loss

class Layer():

    def __init__(self, input_size, neuron_count, activation, derivative):
        self.neurons = []
        for _ in range(neuron_count):
            neuron = Neuron(input_size, activation, derivative)
            self.neurons.append(neuron)

    def forward(self, input): #neuron = Neuron(input_size, activation, derivative)
        self.outputs=[] #forward results of all neurons in the layer
        for neuron in (self.neurons):
            self.output=neuron.forward(input) #neuron.forward(input) ==> Neuron(input_size, activation, derivative).forward(input)
            self.outputs.append(self.output)
        return self.outputs


layer = Layer(3, 4, "relu", "relu_derivative")

input = np.array([2.0, 1.0, 3.0])

output = layer.forward(input)

print(output)