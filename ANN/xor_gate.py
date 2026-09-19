import numpy as np
import matplotlib.pyplot as plt
import basic_neural_training 
import optimizers as optimizers

input_data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
target = np.array([
    [0],
    [1],
    [1],
    [0]
])

np.random.seed(42)
learning_rate = 0.1
optimizer = optimizers.Adam(learning_rate=learning_rate)
network = basic_neural_training.NeuralNetwork(optimizer)
network.add(basic_neural_training.Layer(input_size=2, neuron_count=4, activation="leaky_relu", derivative="leaky_relu_derivative"))
network.add(basic_neural_training.Layer(input_size=4, neuron_count=1, activation="sigmoid", derivative="sigmoid_derivative"))

#======================================#
prediction = network.forward(input_data)
print("prediction =", prediction)
#======================================#
"""
#====Analitic Gradient vs Numerical Gradient verification===
prediction = network.predict(input_data)
loss = basic_neural_training.mse_loss(target, prediction)
dA = 2 * (prediction - target) / target.size
network.backward(dA)


analytic_gradient = network.layers[0].dW[0, 0]
weights = network.layers[0].neurons[0].weights
original_weights = weights.copy()
epsilon = 1e-5
print("Analytic gradients:", analytic_gradient)

network.layers[0].neurons[0].weights[0] = original_weights[0] + epsilon
prediction_plus = network.forward(input_data)

network.layers[0].neurons[0].weights[0] = original_weights[0] - epsilon
prediction_minus = network.forward(input_data)

loss_plus = basic_neural_training.mse_loss(target, prediction_plus)
loss_minus = basic_neural_training.mse_loss(target, prediction_minus)
numerical_gradients = (loss_plus - loss_minus) / (2 * epsilon)
print("Numerical gradients:", numerical_gradients)
print("Difference:", np.abs(analytic_gradient - numerical_gradients))

network.layers[0].neurons[0].weights = original_weights
"""
loss_data = network.fit(
    input_data,
    target,
    epochs=1000,
    batch_size=4
)
print("epochs =", len(loss_data))
prediction = network.predict(input_data)
print(prediction)

epoch_numbers = range(1, len(loss_data) + 1)

plt.plot(epoch_numbers, loss_data)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("XOR Training Loss")
plt.show()

#RESULT:
#  epochs = 1000
# [[0.0036851 ]
#  [0.98938295]
#  [0.98936128]
#  [0.01289357]]
