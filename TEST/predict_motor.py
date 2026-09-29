import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ANN"))

from DATA.scaler import StandardScaler
from ANN.basic_neural_training import NeuralNetwork, Layer

try:
    data = np.load('../DATA/motor_model_weights.npz')
except FileNotFoundError:
    print("Model weights file not found. Please run the training script first.")
    exit()

scaler_X = StandardScaler() #initialize the input scaler 
scaler_X.mean, scaler_X.std = data['mean_X'], data['std_X'] #set the mean and std for the input scaler from the saved data
#scaler_X set the before traning value mean_X    [scaler_X.mean==mean_X]    

scaler_y = StandardScaler()
scaler_y.mean, scaler_y.std = data['mean_y'], data['std_y']

#SETUP THE OWN NEURAL NETWORK AND GİVE THE WEIGHTS FROM THE TRAINED MODEL
model =NeuralNetwork(optimizer=None) #initialize the neural network without an optimizer since we are only predicting
model.add(Layer(input_size=6, neuron_count=32, activation='relu', derivative='relu_derivative'))
model.add(Layer(input_size=32, neuron_count=16, activation='relu', derivative='relu_derivative'))
model.add(Layer(input_size=16, neuron_count=1, activation='linear', derivative='linear_derivative'))

""" MİSTAKE
model.layers[0].weights, model.layers[0].biases = data["W1"], data["b1"]
model.layers[1].weights, model.layers[1].biases = data["W2"], data["b2"]
model.layers[2].weights, model.layers[2].biases = data["W3"], data["b3"]
"""
for layer, (w_key, b_key) in zip(model.layers, [('W1', 'b1'), ('W2', 'b2'), ('W3', 'b3')]):
    W = data[w_key]
    b = data[b_key]
    for i, neuron in enumerate(layer.neurons):
        neuron.weights = W[:, i].copy()
        neuron.bias = b[i]


# -------------------------------------------------------------------------
# BUG FIX: Why we must update each neuron directly
# -------------------------------------------------------------------------
# 1. Problem: `Layer.forward()` always rebuilds its weight matrix from neurons.
#    Setting `layer.weights = W` did nothing, so the model used random weights!
#
# 2. Solution: We copy trained weights directly into each neuron (`W[:, i]`).
# -------------------------------------------------------------------------



print("=" * 50)
print("MODEL PREDİCT SYSTEM")
print("=" * 50)
print("Enter the following motor parameters:")

try:
    u_q = float(input("u_q (Voltage in q-axis): "))
    u_d = float(input("u_d (Voltage in d-axis): "))
    i_q = float(input("i_q (Current in q-axis): "))
    i_d = float(input("i_d (Current in d-axis): "))
    ambient_temp = float(input("ambient_temp (Ambient temperature): "))
    coolant_temp = float(input("coolant_temp (Coolant temperature): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    exit()

raw_input = np.array([[u_q, u_d, i_q, i_d, ambient_temp, coolant_temp]])
scaled_input = scaler_X.transform(raw_input)
scaled_prediction = model.predict(scaled_input)
rpm_prediction = scaler_y.inverse_transform(scaled_prediction)

print(f"Predicted motor speed (RPM): {rpm_prediction[0][0]:.2f}")