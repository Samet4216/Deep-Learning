"""
import sys
from pathlib import Path
# dataset.py lives in DATA, not in ANN.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DATA.dataset import create_dataset, split_dataset
import numpy as np
X=np.arange(300).reshape(100, 3) # Create a 100x3 array of integers from 0 to 299
y=np.arange(100).reshape(100, 1)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
    seed=42
)

from basic_neural_training import NeuralNetwork, Layer
from optimizers import Adam

optimizer = Adam(learning_rate=0.01)
network = NeuralNetwork(optimizer)

network.add(
    Layer(
        input_size=3,
        neuron_count=4,
        activation="leaky_relu",
        derivative="leaky_relu_derivative"
    )
)

network.add(
    Layer(
        input_size=4,
        neuron_count=1,
        activation="linear",
        derivative="linear_derivative"
    )
)

history = network.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=10,
    validation_data=(X_val, y_val)
)

print("Training loss history:", history["loss"])
print("Validation loss history:", history["val_loss"])

print("X train min:", X_train.min(axis=0))
print("X train max:", X_train.max(axis=0))
print("X train mean:", X_train.mean(axis=0))
print("y train min:", y_train.min())
print("y train max:", y_train.max())
print("y train mean:", y_train.mean())
""""""
print("\n" + "="*40)
print("FEATURE SCALING TESTLERİ")
print("="*40)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from DATA.scaler import MinMaxScaler, StandardScaler
from DATA.dataset import create_dataset, split_dataset
import numpy as np
X=np.arange(300).reshape(100, 3) # Create a 100x3 array of integers from 0 to 299
y=np.arange(100).reshape(100, 1)

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.70,
    validation_ratio=0.15,
    seed=42
)

scaler = MinMaxScaler()
scalled = scaler.fit_transform(X_train)
print("Scaled X train min:", scalled.min(axis=0))
print("="*50)
scalled_to_original = scaler.inverse_transform(scalled)
print("Scaled X train to original:", scalled_to_original[:3])
"""
