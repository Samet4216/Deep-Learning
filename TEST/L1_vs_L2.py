import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ANN"))

import numpy as np
from DATA.dataset import split_dataset
from ANN.basic_neural_training import NeuralNetwork, Layer
from ANN.optimizers import Adam

# ==========================================
# 1. GENERATE DATASET
# ==========================================
np.random.seed(42)
X = np.random.uniform(-3, 3, (30, 1))
noise = np.random.randn(30, 1) * 0.8
y = np.cos(X) + noise

X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.25,
    validation_ratio=0.65,
    seed=42
)
EPOCH = 1000

#============================================
# NORMAL MODEL (NO REGULARİZATİON)
#============================================
np.random.seed(42)
model_normal = NeuralNetwork(Adam(learning_rate=0.01))
model_normal.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_normal.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_normal.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"))

history_normal = model_normal.fit(X_train, y_train, EPOCH, batch_size=len(X_train), validation_data=(X_validation, y_validation))

#============================================
# L1 REG. MODEL (RİDGE)
#============================================
np.random.seed(42)
model_L1 = NeuralNetwork(Adam(learning_rate=0.01), L1_lambda=0.0005)
model_L1.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_L1.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_L1.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"))

history_L1 = model_L1.fit(X_train, y_train, EPOCH, batch_size=len(X_train), validation_data=(X_validation, y_validation))

#============================================
# L2 REG. MODEL (LASSO)
#============================================
np.random.seed(42)
model_L2 = NeuralNetwork(Adam(learning_rate=0.01), L2_lambda=0.0005)
model_L2.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_L2.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_L2.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"))

history_L2 = model_L2.fit(X_train, y_train, EPOCH, batch_size=len(X_train), validation_data=(X_validation, y_validation))

#============================================
# RESULTS
#============================================
def report(name, history):
    train_loss = history["loss"]
    val_loss = history["val_loss"]
    best_val_loss = min(val_loss)
    print(f"\n{'='*55}")
    print(f"RAPOR: {name}")
    print(f"{'='*55}")
    print(f"Best Val Loss              : {best_val_loss:.6f} ")
    print(f"Last Train Loss            : {train_loss[-1]:.6f}")
    print(f"Last Val Loss              : {val_loss[-1]:.6f}")
    return best_val_loss, val_loss[-1]

best_normal, last_val_normal = report("NORMAL (No Reg)", history_normal)
best_L2, last_val_L2 = report("L2 (Ridge)", history_L2)
best_l1, last_val_L1 = report("L1 (Lasso)", history_L1)

for name, model in [("NORMAL", model_normal), ("L2", model_L2), ("L1", model_L1)]:
    total=0 #total weights in model
    near_zero_count=0 #how many weight is near zero in the model?
    for layer in model.layers:
        total += layer.W.size #add this layer weight count to total. for sparsity[deleted weight]
        near_zero_count += np.sum(np.abs(layer.W)<0.001) #outputs are true and false (1 or 0) [ex.1+1+1+0+0+1+1+0] 
    print(f"{name} -> Total: {total} | Near-Zero-Count: {near_zero_count} | Sparsity: %{(near_zero_count/total)*100:.1f}")

# =======================================================
# RAPOR: NORMAL (No Reg)
# =======================================================
# Best Val Loss              : 0.631553
# Last Train Loss            : 0.091168
# Last Val Loss              : 0.819140

# =======================================================
# RAPOR: L2 (Ridge)
# =======================================================
# Best Val Loss              : 0.628916
# Last Train Loss            : 0.096447
# Last Val Loss              : 0.726989

# =======================================================
# RAPOR: L1 (Lasso)
# =======================================================
# Best Val Loss              : 0.615246
# Last Train Loss            : 0.027431
# Last Val Loss              : 1.011069
# NORMAL -> Total: 2144 | Near-Zero-Count: 16 | Sparsity: %0.7
# L2 -> Total: 2144 | Near-Zero-Count: 118 | Sparsity: %5.5
# L1 -> Total: 2144 | Near-Zero-Count: 1705 | Sparsity: %79.5

# But then it exploded (+64.3%) 
# Why=> It zeroed out 80% of the weights but the remaining 20% ​​were completely free and they memorized the data