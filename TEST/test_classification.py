
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ANN"))

import numpy as np
import ANN.loss_functions as lf
from ANN.basic_neural_training import NeuralNetwork, Layer
from DATA.dataset import to_one_hot
from visualization import plot_decision_boundary
from ANN.optimizers import Adam

if __name__ == "__main__":
    X0 = np.random.randn(100, 2) + np.array([2, 2])  # Class 0
    X1 = np.random.randn(100, 2) + np.array([-2, -2])  # Class 1
    X2 = np.random.randn(100, 2) + np.array([2, 2])  # Class 2

    X = np.vstack((X0, X1, X2))
    y = np.array([0]*100 + [1]*100 + [2]*100) 
    """
    vstack => Stack arrays in sequence vertically (row wise).
    a = np.array([[1, 2, 3]])
    b = np.array([[4, 5, 6]])

    np.vstack((a, b))
    array([[1, 2, 3],
           [4, 5, 6]])
    """
    y_one_hot = to_one_hot(y, class_count=3)
    model = NeuralNetwork(Adam(learning_rate=0.01))
    model.add(Layer(input_size=2, neuron_count=16, activation="leaky_relu", derivative="leaky_relu_derivative")),
    model.add(Layer(input_size=16, neuron_count=3, activation="softmax", derivative="softmax_derivative"))
    model.compile(loss_function=lf.categorical_cross_entropy, loss_derivative=lf.categorical_cross_entropy_derivative)

    history = model.fit(
        X,
        y_one_hot,
        epochs=1000,
        batch_size=32,
        validation_data=(X, y_one_hot),
        patience=50
    )

    plot_decision_boundary(model, X, y, title="Entegration Test: Decision Boundary Visualization")