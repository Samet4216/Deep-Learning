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
import activation_functions as af
import loss_functions 

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

class Layer():

    def __init__(self, input_size, neuron_count, activation, derivative, alpha=None, dropout_rate=0.0):
        self.neurons = []
        for _ in range(neuron_count):
            neuron = Neuron(input_size, activation, derivative, alpha)
            self.neurons.append(neuron)
        self.dropout_rate = dropout_rate
        self.mask = None # store the mask for use in backpropagation 

    def forward(self, input, training=True): # neuron = Neuron(input_size, activation, derivative)
        self.input = input # store the input for use in backpropagation
        self.W = np.array([neuron.weights for neuron in self.neurons]).T # TRANPOSE(T)==> weight matrix 3*4 (3 in 4 out) to change 4*3 (4 in 3 out)
        self.b = np.array([neuron.bias for neuron in self.neurons]) # bias matrix of the first layer
        self.Z = np.dot(input, self.W) + self.b # outputs matrix
        activation = self.neurons[0].activation # assuming all neurons in the layer use the same activation function
        self.outputs = activation(self.Z, alpha=self.neurons[0].alpha) # apply activation function to the outputs

        if training and self.dropout_rate > 0.0: # because only use dropout training and dont use test time
            self.mask = (np.random.rand(*self.outputs.shape) >= self.dropout_rate) / (1.0 - self.dropout_rate) #random.randn => generate number between 1 and 0
            # *self.outputs.shape => with * unpacking operation so ex. (18,36) => 18,36 because dont use tuple in random.randn
            self.outputs = self.outputs * self.mask
        else:
            self.mask = None 


        return self.outputs

    def backward(self, dA,L1_lambda=0.0, L2_lambda=0.0):

        if self.dropout_rate >= 0.0 and self.mask is not None:
            dA = dA * self.mask

        activation_derivative = self.neurons[0].derivative(
            self.Z,
            alpha=self.neurons[0].alpha
        ) # call the previously selected activation derivative with its parameter.
        dZ = dA * activation_derivative # aktivation derivative and output derivative for dot product gradient :D
        dW = np.dot(self.input.T, dZ)
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
        if L1_lambda > 0.0:
            dW += L1_lambda * np.sign(self.W)
        if L2_lambda > 0.0:
            dW += L2_lambda * self.W
        self.dW = dW
        self.db = np.sum(dZ, axis=0) # axis=0 specifies that the operation is performed along the columns.
        # it collects samples and produces a result for each neuron.
            # Example: dZ = [[1, 2, 3, 4], [10, 20, 30, 40], [100, 200, 300, 400]]
            # axis=0 sums the columns -> db = [111, 222, 333, 444] (one bias gradient per neuron)
        dA_pre = np.dot(dZ, self.W.T)
            # send the gradient back to the previous layer.
                # dA_pre = dL/dA_pre = dZ @ W.T
                # this tells us how much each input/previous-layer activation
        return dA_pre 
    """  NOW, WORKİNG=> ====NeuralNetwork.update() => optimizer.update(layer)=== 

    def update(self, learning_rate):
        for i, neuron in enumerate(self.neurons):
            neuron.weights -= learning_rate * self.dW[:, i] # For i=0, update the neuron's weights. dW[:, i] selects all rows in column i.
            # Example:
            # dW = [[ 8, 44],
                #  [12, 66],
                #  [ 4, 22]]
            #
            #because TRANSPOSE(T) weight matrix 3*4 (3 in 4 out) to change 4*3 (4 in 3 out)
            3*4 = 12 elements
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

    def __init__(self, optimizer, L1_lambda=0.0, L2_lambda=0.0):
        self.layers = []
        self.optimizer = optimizer #for ex. self.optimizer.time += 1 in update() method==> adam optimizer
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda

    def compile(self, loss_function, loss_derivative):
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative

    def add(self, layer):
        self.layers.append(layer)

    def forward (self, input, training=True):
        output = input # The output from the previous layer will be our input
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def predict(self, inputs):
        return self.forward(input=inputs, training=False)

    def backward(self, dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA, L1_lambda=self.L1_lambda, L2_lambda=self.L2_lambda) #dA => last layer output
            #for ex. Layer 1.backward([92,118,144]) || [92,118,144] is layer-2 input
    
    def update(self):
        self.optimizer.step()
        for layer in self.layers:
            self.optimizer.update(layer)

    def train_step(self, input, y_true):
        y_pred = self.forward(input, training=True) #use dropout for training step so training=True
        loss = self.loss_function(y_true, y_pred) #calculate loss for the current batch
        
        if self.L1_lambda > 0.0: loss += self.L1_lambda * sum(np.sum(np.abs(layer.W)) for layer in self.layers)
        if self.L2_lambda > 0.0: loss += self.L2_lambda * sum(np.sum(layer.W**2) for layer in self.layers) / 2
        dA = self.loss_derivative(y_true, y_pred) #dinamic loss function derivative for backpropagation.
        # y_true.size => gives exactly the total number of elements. For example, if y_true.shape == (3,4): gives 12.

        self.backward(dA) # definition for later use 
        self.update()
        return loss

    def fit(self, input, y_true, epochs, batch_size, validation_data=None, patience=None, min_delta=0.0, restore_best_weights=True):
        if validation_data is not None:
            validation_input, validation_target = validation_data # Unpack the validation data
        loss_data = []
        val_loss_data = []

        best_val_loss = np.inf #infinitw loss score at the start
        best_weights = None
        best_epoch = 0
        wait = 0

        for epoch in range(epochs):
            epoch_loss = 0
            indices = np.random.permutation(len(input)) # shuffle the indices of the input data
            for i in range(0, len(input), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_input = input[batch_indices]
                batch_y_true = y_true[batch_indices]
                loss = self.train_step(batch_input, batch_y_true)
                batch_num = batch_input.shape[0] # number of samples in current batch
                epoch_loss += loss * batch_num  # Multiply by batch_num to get the total loss for the batch
            loss_average = epoch_loss / len(input)  # Divide by the total number of samples to get the average loss for the epoch
            print(f"Epoch {epoch + 1}: Loss = {loss_average:.6f}")
            loss_data.append(loss_average)

            if validation_data is not None:
                validation_loss = self.evaluate(
                    validation_input,
                    validation_target,
                    func=self.loss_function
                )
                val_loss_data.append(validation_loss)
                print(f"Validation Loss = {validation_loss:.6f}")

                if patience is not None:
                    if validation_loss < (best_val_loss - min_delta):
                        best_val_loss = validation_loss
                        best_epoch = epoch + 1
                        wait = 0
                        if restore_best_weights:
                            best_weights = self.get_weights()
                    else:
                        wait += 1
                        if wait >= patience:
                            print(f"\n[EARLY STOPPING] Stopped at epoch {epoch + 1}")
                            print(f"[EARLY STOPPING] Best score was at epoch {best_epoch} (Val Loss: {best_val_loss:.6f})")
                            if restore_best_weights and best_weights is not None:
                                print(f"[EARLY STOPPING] Restored best weights from epoch {best_epoch}.")
                                self.set_weights(best_weights)
                            break
        history = {
            "loss": loss_data,
            "val_loss": val_loss_data
        }
        return history

    def evaluate (self, input, y_true, func=None): # func=> which loss function to use for evaluation, default is mean squared error
        if func is None:
            func = self.loss_function
        y_pred = self.forward(input, training=False) #dont use dropout => not training step so False
        loss = func(y_true, y_pred)
        return loss
    
    def get_weights(self): #create copy weights and biases of the neurons in all layers
        weights = []
        for layer in self.layers:
            layer_weight = [(neuron.weights.copy(), neuron.bias) for neuron in layer.neurons]
            weights.append(layer_weight)
        return weights

    def set_weights(self, saved_weights):
        for layer, layer_weight in zip(self.layers, saved_weights): #pair each layer with its saved weights
            for neuron, (weight, bias) in zip(layer.neurons, layer_weight): #step through each neuron in the layer and restore its weights and bias.. (weight, bias)=>saved_weights
                neuron.weights = weight.copy()
                neuron.bias = bias
            layer.W = np.array([neuron.weights for neuron in layer.neurons]).T
            layer.b = np.array([neuron.bias for neuron in layer.neurons])