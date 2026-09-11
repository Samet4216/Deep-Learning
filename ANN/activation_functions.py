#RELU FONKSİYONU DENEME
import matplotlib.pyplot as plt
import numpy as np

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):  # z > 0 [False, False, True]
    return float(z > 0) # [0.0, 0.0, 1.0]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    output = sigmoid(z)
    return output * (1 - output)

if __name__ == "__main__":
    z_values = np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10])
    for z in z_values:
        output = relu(z)
        derivative = relu_derivative(z)
        print(f"z={z:3} | ReLU={output:6} | derivative={derivative}")

    print("======RELU vs SİGMOİD=======")

    z_values = np.linspace(-10, 10, 400) #-10 ile 10 araasında 400 tane nokta oluşturur

    sigmoid_values = 1 / (1 + np.exp(-z_values))
    relu_values = np.maximum(0, z_values)

    plt.plot(z_values, sigmoid_values, label="Sigmoid")
    plt.plot(z_values, relu_values, label="ReLU")

    plt.xlabel("z")
    plt.ylabel("activation")
    plt.legend()
    plt.grid()
    plt.show()