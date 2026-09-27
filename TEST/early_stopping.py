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
# 1. DATASET
# ==========================================
np.random.seed(42)
X = np.random.uniform(-3, 3, (30, 1))
noise = np.random.randn(30,1) * 0.8
y = np.cos(X) + noise 

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.3,
    validation_ratio=0.5,
    seed=42
)
MAX_EPOCH = 1000

# ==========================================
# 2. MODEL 1: NO EARLY STOPPİNG (1000 Epoch)
# ==========================================
np.random.seed(42)
model_normal = NeuralNetwork(Adam(learning_rate=0.01))
model_normal.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_normal.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_normal.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"))

history_normal = model_normal.fit(
    X_train,
    y_train,
    epochs=MAX_EPOCH,
    batch_size=len(X_train),
    validation_data=(X_val, y_val),
    patience=None
)
# ==========================================
# 3. MODEL 2: EARLY STOPPING VAR (patience=30)
# ==========================================
np.random.seed(42)
model_early_stop = NeuralNetwork(Adam(learning_rate=0.01))
model_early_stop.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_early_stop.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_early_stop.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"))

history_early_stop = model_early_stop.fit(
    X_train,
    y_train,
    epochs=MAX_EPOCH,
    batch_size=len(X_train),
    validation_data=(X_val, y_val),
    patience=30,              
    restore_best_weights=True  #back best weight at the stop
)
# ==========================================
# 4. RESULT
# ==========================================
final_val_normal = model_normal.evaluate(X_val, y_val)
final_val_es = model_early_stop.evaluate(X_val, y_val)
print("\n" + "="*60)
print("EARLY STOPPING KARSILASTIRMA RAPORU")
print("="*60)
print(f"Normal Model (NO ES)  -> total Epoch : {len(history_normal['loss'])} / {MAX_EPOCH}")
print(f"Normal Model (NO ES)  -> last Val Loss : {final_val_normal:.6f}")
print("-" * 60)
print(f"Early Stopping Model -> total Epoch : {len(history_early_stop['loss'])} / {MAX_EPOCH}")
print(f"Early Stopping Model -> last Val Loss : {final_val_es:.6f}")
print("="*60)

# [EARLY STOPPING] Stopped at epoch 53
# [EARLY STOPPING] Best score was at epoch 23 (Val Loss: 0.487191)
# [EARLY STOPPING] Restored best weights from epoch 23.

# ============================================================
# EARLY STOPPING KARSILASTIRMA RAPORU
# ============================================================
# Normal Model (NO ES)  -> total Epoch : 1000 / 1000
# Normal Model (NO ES)  -> last Val Loss : 0.694948
# ------------------------------------------------------------
# Early Stopping Model -> total Epoch : 53 / 1000
# Early Stopping Model -> last Val Loss : 0.487191
# ============================================================