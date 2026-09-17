import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, layer):
        for i,neuron in enumerate(layer.neurons):
            neuron.weights -= self.learning_rate * layer.dW[:, i] # For i=0, update the neuron's weights. dW[:, i] selects all rows in column i.
            neuron.bias -= self.learning_rate * layer.db[i]


class momentum:
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def update(self, layer):
        if layer not in self.velocity: # check if the layer is already in the velocity dictionary
            self.velocity[layer] = {
                "weights": np.zeros_like(layer.dW),
                "bias": np.zeros_like(layer.db) 
                            #np.zeros_like(layer.dW)
                            # [
                            #   [0, 0],
                            #   [0, 0],
                            #   [0, 0]
                            # ]
            }
        v = self.velocity[layer]
        v["weights"] = (
            self.beta * v["weights"] 
            - self.learning_rate * layer.dW
        )
        v["bias"] = (
            self.beta * v["bias"]
            - self.learning_rate * layer.db
        )
        for i, neuron in enumerate(layer.neurons):
            neuron.weights += v["weights"][:, i] # For i=0, update the neuron's weights. v["weights"][:, i] selects all rows in column i.
            neuron.bias += v["bias"][i]
        
