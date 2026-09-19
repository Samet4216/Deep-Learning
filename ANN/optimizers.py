import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self):
        pass  # No state to update for SGD

    def update(self, layer):
        for i,neuron in enumerate(layer.neurons):
            neuron.weights -= self.learning_rate * layer.dW[:, i] # For i=0, update the neuron's weights. dW[:, i] selects all rows in column i.
            neuron.bias -= self.learning_rate * layer.db[i]

class Momentum:
    def __init__(self, learning_rate, beta):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = {}

    def step(self):
        pass  # No state to update for momentum

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
        
class Adam:
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = {} 
        self.v_weights = {}
        self.m_bias = {}
        self.v_bias = {}
        self.time = 0 #optimizer update/step counter
        """ time=>
                Epoch 1
        ├─ batch 1 → time = 1
        ├─ batch 2 → time = 2
        ├─ ...
        └─ batch 10 → time = 10

        Epoch 2
        ├─ batch 1 → time = 11
        └─ ...
        """

    def step(self):
        self.time += 1 # Increment the time step for bias correction in the ADAM optimizer.

    def update(self, layer):
        if layer not in self.m_weights:
            self.m_weights[layer] = np.zeros_like(layer.dW)
            self.v_weights[layer] = np.zeros_like(layer.dW)
            self.m_bias[layer] = np.zeros_like(layer.db)
            self.v_bias[layer] = np.zeros_like(layer.db)
        self.m_weights[layer] = (
            self.m_weights[layer] * self.beta1 +
            (1-self.beta1) * layer.dW
        )
        self.v_weights[layer] = (
            self.v_weights[layer] * self.beta2 +
            (1-self.beta2) * (layer.dW ** 2)
        )
        self.m_bias[layer] = (
            self.m_bias[layer] * self.beta1 +
            (1-self.beta1) * layer.db
        )
        self.v_bias[layer] = (
            self.v_bias[layer] * self.beta2 +
            (1-self.beta2) * (layer.db ** 2)
        ) 
        #====bias correction====
        m_hat_weights = self.m_weights[layer] / (1 - self.beta1 ** self.time)
        v_hat_weights = self.v_weights[layer] / (1 - self.beta2 ** self.time)
        m_hat_bias = self.m_bias[layer] / (1 - self.beta1 ** self.time)
        v_hat_bias = self.v_bias[layer] / (1 - self.beta2 ** self.time)

        weight_update = (
            m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        )
        bias_update = (
            m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
        )
        for i,neuron in enumerate(layer.neurons):
            neuron.weights -= self.learning_rate * weight_update[:, i]
            neuron.bias -= self.learning_rate * bias_update[i]
        