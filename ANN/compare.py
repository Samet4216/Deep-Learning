#note: ctrl + k => ctrl + c to comment the selected lines

    #===Adam vs SGD optimizer comparison===#
import numpy as np
import matplotlib.pyplot as plt
import basic_neural_training 
import optimizers as optimizers

input_data = np.array([
    [2.0, 1.0, 3.0],
    [4.0, 2.0, 1.0],
    [1.0, 3.0, 2.0]
])
target = np.array([
    [1.0],
    [0.0],
    [1.0]
])
learning_rate = 0.2
#===Adam optimizer===
np.random.seed(42)  # fixes NumPy's random number generator. randomly generated values are the same on every run.
optimizer = optimizers.Adam(learning_rate = learning_rate, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8)
network = basic_neural_training.NeuralNetwork(optimizer)
network.add(basic_neural_training.Layer(input_size=3, neuron_count=4, activation="relu", derivative="relu_derivative"))
network.add(basic_neural_training.Layer(input_size=4, neuron_count=1, activation="relu", derivative="relu_derivative"))

loss_data = network.fit(
    input_data,
    target,
    epochs=20,
    batch_size=3
)
adam_loss = loss_data
#===SGD optimizer===
np.random.seed(42)  # fixes NumPy's random number generator. randomly generated values are the same on every run.
optimizer = optimizers.SGD(learning_rate = learning_rate)
optimizer.time = 0 #The time attribute is already defined in the =Adam= class but it is missing in the =SGD= class
network = basic_neural_training.NeuralNetwork(optimizer)
network.add(basic_neural_training.Layer(input_size=3, neuron_count=4, activation="relu", derivative="relu_derivative"))
network.add(basic_neural_training.Layer(input_size=4, neuron_count=1, activation="relu", derivative="relu_derivative"))

loss_data = network.fit(
    input_data,
    target,
    epochs=20,
    batch_size=3
)
sgd_loss = loss_data
print("Learning Rate:", learning_rate)
print("Adam Loss:", adam_loss[-1])
print("SGD Loss:", sgd_loss[-1])

epochs = range(1, len(adam_loss) + 1)
plt.plot(epochs, adam_loss, label="Adam")
plt.plot(epochs, sgd_loss, label="SGD")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Adam vs SGD Loss")
plt.legend()
plt.show()

# RESULTS:
# The loss value in the first epoch is the same for both optimizers:
        # Adam: 0.532694
        # SGD : 0.532694
# For learning_rate = 0.1:
    # Adam Loss: 0.042951310707804634
    # SGD Loss: 0.17346963716093866
# For learning_rate = 0.01:
    # Adam Loss: 0.22831573649897616
    # SGD Loss: 0.36762524423765064
# For learning_rate = 0.2:
    # Adam Loss: 0.22224050373293583
    # SGD Loss: 0.06666606851876834

# SGD applies the learning rate directly to the gradient, so a learning rate of 0.2
# produces larger parameter updates. Adam scales each update using its own adaptive
# formula. Therefore, Adam reaches a higher loss than SGD with a learning rate of 0.2.
# A learning rate of 0.2 may be too aggressive for Adam in this example. With this
# small dataset and only a few epochs, SGD happens to perform better.
# Adam often performs better on larger datasets or with longer training because it
# combines ideas from momentum and RMSProp.
