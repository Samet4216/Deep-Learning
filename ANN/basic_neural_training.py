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

    def __init__(self, input_size, activation, derivative, alpha=None):
        self.weights = np.random.randn(input_size) * 0.01
        self.bias = 0.1 # A negative initial bias can cause ReLU units to stop learning.
        self.activation = getattr(af, activation) # getattr receives the module and function name.
        self.derivative = getattr(af, derivative) # Activation and derivative names are passed as strings.
        self.alpha = alpha

    def forward(self,input):
        self.z = np.dot(input,self.weights)+self.bias # z = (x * w) + b
        self.output = self.activation(self.z, alpha=self.alpha)
        return self.output

    # def mse_loss(y_true, y_pred):

    def gradient(self, input, y_true): #dA => gradient for activation func  ||  d => gradient

        A = self.output # Prediction returned by the activation function.

        dA = -2 * (y_true - A) # Derivative of the loss with respect to the activation.
        activation_derivative = self.derivative(self.z, alpha=self.alpha)

        # Chain rule
        dZ = dA * activation_derivative
        gradient = dZ * input #(dz/dw) gradient of weights

        bias_gradient = dZ * 1 # dz/db is 1.

        return gradient, bias_gradient

    def update(self, gradient, bias_gradient, learning_rate):
        self.weights -= learning_rate * gradient
        self.bias -= learning_rate * bias_gradient

def mse_loss(y_true, y_pred):
    loss=(y_true-y_pred)**2
    return loss

class Layer():

    def __init__(self, input_size, neuron_count, activation, derivative, alpha=None):
        self.neurons = []
        for _ in range(neuron_count):
            neuron = Neuron(input_size, activation, derivative, alpha)
            self.neurons.append(neuron)

    def forward(self, input): # neuron = Neuron(input_size, activation, derivative)
        self.input = input

        W = np.array([neuron.weights for neuron in self.neurons]).T # TRANPOSE(T)==> weight matrix 3*4 (3 in 4 out) to change 4*3 (4 in 3 out)
        self.W = W
        b = np.array([neuron.bias for neuron in self.neurons]) # bias matrix of the first layer
        Z = np.dot(input, W) + b # outputs matrix
        self.Z = Z
        self.W = W
        self.b = b

        activation = self.neurons[0].activation # assuming all neurons in the layer use the same activation function
        self.outputs = activation(Z, alpha=self.neurons[0].alpha) # apply activation function to the outputs
        return self.outputs

    def backward(self, dA):
        activation_derivative = self.neurons[0].derivative(
            self.Z,
            alpha=self.neurons[0].alpha
        ) # call the previously selected activation derivative with its parameter.
        dZ = dA * activation_derivative # aktivation derivative and output derivative for dot product :D
        dW = []
        for x in (self.input):
            row = []
            for dz in dZ:
                row.append(x * dz)
            dW.append(row) #numpy lib outher func some func 
            """ input = [2, 3, 1], dZ = [4, 22]
            #
            # Start:
            # dW = []
            #
            # x=2 → row=[] → 2*4=8 → 2*22=44 → row=[8,44] → dW=[[8,44]]
            # x=3 → row=[] → 3*4=12 → 3*22=66 → row=[12,66] → dW=[[8,44],[12,66]]
            # x=1 → row=[] → 1*4=4 → 1*22=22 → row=[4,22] → dW=[[8,44],[12,66],[4,22]]
            #
            # Final dW:
            # [[8,44],
            #  [12,66],
            #  [4,22]]
            """
        dW = np.array(dW)
        db = dZ #dL/db = dL/dZ * dZ/db(1) => dL/db = dL/dZ
        self.dW = dW
        self.db = db
        dA_pre = np.dot(dZ, self.W.T) # dL/dA => dL/dZ(dZ) * dZ/dW (W.transpoze) find weight gradient for last layer >>also<< input of first layer
        return dW, db, dA_pre 

    def update(self, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.weights -= learning_rate * self.dW[:, i] # For i=0, update the neuron's weights. dW[:, i] selects all rows in column i.
            neuron.bias -= learning_rate * self.db[i]
"""         # Example:
            # dW =
            # [[ 8, 44],
            #  [12, 66],
            #  [ 4, 22]]
            #
            #because TRANSPOSE(T) weight matrix 3*4 (3 in 4 out) to change 4*3 (4 in 3 out)
            3
            # dW[:,0] → [8,12,4]   → gradients of neuron 0
            # dW[:,1] → [44,66,22] → gradients of neuron 1
            #
            # db = [4,22]
            # db[0] → 4  → bias gradient of neuron 0
            # db[1] → 22 → bias gradient of neuron 1
            #
            # learning_rate = 0.1
            #
            # neuron 0:
            # weights [1,2,3] - 0.1*[8,12,4] = [0.2,0.8,2.6]
            # bias 1 - 0.1*4 = 0.6
            #
            # neuron 1:
            # weights [4,5,6] - 0.1*[44,66,22] = [-0.4,-1.6,3.8]
            # bias 2 - 0.1*22 = -0.2
"""


class NeuralNetwork():

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward (self, input):
        output = input # The output from the previous layer will be our input
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, dA):
        for layer in reversed(self.layers):
            dW, db, dA_pre = layer.backward(dA) #dA => last layer output
            dA = dA_pre # layer-1 output is layer-2 input
            #for ex. Layer 1.backward([92,118,144]) || [92,118,144] is layer-2 input
    
    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

#TEST
network = NeuralNetwork()

network.add(Layer(3, 4, "relu", "relu_derivative"))
network.add(Layer(4, 3, "relu", "relu_derivative"))

input = np.array([2.0, 1.0, 3.0])

output = network.forward(input)

print(output)
print(output.shape)

"""    
layer = Layer(input_size=3, neuron_count=4, activation="relu", derivative="relu_derivative")
input = np.array([0.2, 0.1, 0.3])
output = layer.forward(input)
print("Output:", output)
print("Output shape:", output.shape)
"""