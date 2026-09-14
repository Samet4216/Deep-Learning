import numpy as np
#.\.deep_venv\Scripts\Activate.ps1

"""
input = np.array([0.2, 0.1, 0.3])
output=neuron.forward(input)

target = 1
loss = mse_loss(target, output)

loss1 = mse_loss(target, output)

delta = 1e-5
original_weights = neuron.weights[0]

neuron.weights[0] = original_weights + delta
loss_plus = mse_loss(target, neuron.forward(input))

neuron.weights[0] = original_weights - delta
loss_minus = mse_loss(target, neuron.forward(input))

gradient_approx = (loss_plus - loss_minus) / (2 * delta)
neuron.weights[0] = original_weights

print(neuron.weights)
print(neuron.bias)
print(output)
print("===================")
print("Tahmin:", output)
print("Gerçek:", target)
print("Loss:", loss)
print("===================")
print("Loss:", loss1)
print("Loss (+delta):", loss_plus)
print("Loss (-delta):", loss_minus)
print("Numerical gradient (weights[0]):", gradient_approx)
print("===================")
gradient, bias_gradient = neuron.gradient(input, target)
print("Analytic gradient:", gradient)
print("Analytic gradient (weights[0]):", gradient[0])
print("Difference:", abs(gradient_approx - gradient[0]))
print("===================")
gradient, bias_gradient = neuron.gradient(input, target)
old_loss = mse_loss(target, neuron.forward(input))
neuron.update(gradient, bias_gradient, 0.1)
new_loss = mse_loss(target, neuron.forward(input))
print("Eski loss:", old_loss)
print("Yeni loss:", new_loss)
print("====================")
"""


"""
neuron = Neuron(3, "relu", "relu_derivative")
input = np.array([0.2, 0.1, 0.3])
target = 1

def train(neuron, x, target, epochs, learning_rate):
    for epoch in range(epochs):
        y_pred = neuron.forward(x) # Prediction from the neuron.
        loss = mse_loss(target, y_pred) # Training loss point.
        gradient, bias_gradient = neuron.gradient(x, target) # Calculate the required parameter updates.
        neuron.update(gradient, bias_gradient, learning_rate) # Apply the parameter updates.
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: prediction={y_pred:.6f}, loss={loss:.6f}")

train(
    neuron,
    input,
    target,
    epochs=100,
    learning_rate=0.01
)

print("==================")
"""

"""
print("Controlled Leaky ReLU training")

x = np.array([1.0, 1.0, 1.0])
target = 1.0

weights = np.array([-1.0, -1.0, -1.0])
bias = -1.0
alpha = 0.01
learning_rate = 0.01

for epoch in range(100):
    z = np.dot(x, weights) + bias
    prediction = np.where(z > 0, z, alpha * z) # alpha * z => 1 for normally ReLU
    loss = mse_loss(target, prediction)

    d_loss_d_prediction = -2 * (target - prediction)
    d_prediction_d_z = 1.0 if z > 0 else alpha
    common_derivative = d_loss_d_prediction * d_prediction_d_z

    gradient = common_derivative * x
    bias_gradient = common_derivative

    weights -= learning_rate * gradient
    bias -= learning_rate * bias_gradient

    if epoch % 10 == 0 or epoch == 99:
        print(
            f"Epoch {epoch}: prediction={prediction:.6f}, "
            f"loss={loss:.6f}, weights={weights}, bias={bias:.6f}"
        )
"""

from basic_neural_training import Layer, NeuralNetwork, mse_loss

# TEST
network = NeuralNetwork()

network.add(Layer(3, 4, "relu", "relu_derivative"))
network.add(Layer(4, 3, "relu", "relu_derivative"))

input = np.array([2.0, 1.0, 3.0])
target = np.array([1.0, 1.0, 1.0])
"""
loss = network.train_step(
    input, 
    target, 
    learning_rate=0.1
    )

print("Initial loss:", loss)
"""
prediction_before = network.forward(input)
loss_before = np.sum(mse_loss(target, prediction_before))

network.fit(
    input,
    target,
    epochs=5,
    learning_rate=0.1
)

prediction_after = network.forward(input)
loss_after = np.sum(mse_loss(target, prediction_after))

print("Before:", prediction_before)
print("Loss before:", loss_before)
print("After:", prediction_after)
print("Loss after:", loss_after)
