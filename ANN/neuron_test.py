import numpy as np
from basic_neural_training import neuron, mse_loss


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
print("Prediction:", output)
print("Target:", target)
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
print("Old loss:", old_loss)
print("New loss:", new_loss)
print("====================")
"""



neuron = neuron(3, "sigmoid", "sigmoid_derivative")
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