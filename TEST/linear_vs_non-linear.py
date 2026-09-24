
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ANN"))

import numpy as np
from DATA.dataset import split_dataset
from ANN.basic_neural_training import NeuralNetwork, Layer
from ANN.optimizers import Adam


# =============================================
# 1. PREPARATION: DATASET CREATION AND SPLITTING
# =============================================

np.random.seed(42)
# -3 and 3 between 400 rows and 1 column
X = np.random.uniform(-3, 3, (400,1))
# noise is a random normal distribution with 400 rows and 1 column, scaled by 0.2
noise = np.random.randn(400,1) * 0.2
y = (0.5 * (X ** 2) - X + 2) + noise
print("Data created -> X:", X.shape, "| y:", y.shape)

# dataset split
X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.7,
    validation_ratio=0.15,
    seed=42
    )

# ==========================================
# 2. MODEL A: ONE NEURON LİNEAR MODEL
# ==========================================
print("\n" + "="*50)
print("1. EXPERIMENT: LINEAR MODEL (No hidden layers)")
print("="*50)

linear_model = NeuralNetwork(Adam(learning_rate=0.01))
linear_model.add(
    Layer(1, 1, activation="linear", derivative="linear_derivative")
)
linear_model.fit(
    X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test)
)
linear_test_loss = linear_model.evaluate(X_test, y_test)
print(f"-->Linear model mse loss on test set: {linear_test_loss:.4f}")


# ====================================================
# 3. MODEL B: NONLINEAR ANN (Hidden Layer + Leaky ReLU)
# ====================================================
print("\n" + "="*50)
print("2. EXPERIMENT: NONLINEAR ANN (16 Neuron Hidden Layer)")
print("="*50)

nonlinear_model = NeuralNetwork(Adam(learning_rate=0.01))
nonlinear_model.add(
    Layer(1, 16, activation="leaky_relu", derivative="leaky_relu_derivative")
)
nonlinear_model.add(
    Layer(16, 1, activation="linear", derivative="linear_derivative")
)
nonlinear_model.fit(
    X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test)
)
nonlinear_test_loss = nonlinear_model.evaluate(X_test, y_test)
print(f"-->Nonlinear model mse loss on test set: {nonlinear_test_loss:.4f}")

# ==========================================
# 4. COMPARISON OF RESULTS
# ==========================================
print("\n" + "#"*50)
print("FİNAL COMPARİSON OF TEST LOSSES")
print("#"*50)
print(f"Linear Model Test loss : {linear_test_loss:.4f}")
print(f"Nonlinear ANN Test Loss: {nonlinear_test_loss:.4f}")
difference = ((linear_test_loss - nonlinear_test_loss) / linear_test_loss) * 100
print(f"Loss Reduction      : %{difference:.1f} BETTER")
print("#"*50)

#RESULT
# Hidden Layers: must have Nonlinear activation functions (ReLU, Leaky ReLU, GELU vb.) 
# → because the world's curves and complex relationships need to be learned.
# Output Layer (Output Layer): Must always have a Linear activation function (
# f(z) = z
# → because it must be able to freely generate numbers within any range.