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
noise = np.random.randn(30, 1) * 0.8
y = np.cos(X) + noise

X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.3,
    validation_ratio=0.5,
    seed=42,
)
EPOCH=1250
# ==========================================
# 2. NORMAL MODEL (NO DROPOUT)
# ==========================================
np.random.seed(42)
model_normal = NeuralNetwork(Adam(learning_rate=0.01))
model_normal.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_normal.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
model_normal.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"))

history_normal = model_normal.fit(
    X_train,
    y_train, 
    EPOCH,
    batch_size=len(X_train),
    validation_data=(X_validation, y_validation)
)

# ==========================================
# 3. DROPOUT MODEL (HİDDEN LAYER=> %20 DROPOUT)
# ==========================================
np.random.seed(42)
model_dropout = NeuralNetwork(Adam(learning_rate=0.01))
model_dropout.add(Layer(input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative", dropout_rate=0.2))
model_dropout.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative", dropout_rate=0.2))
model_dropout.add(Layer(input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative", dropout_rate=0.0))

history_dropout = model_dropout.fit(
    X_train,
    y_train, 
    EPOCH,
    batch_size=len(X_train),
    validation_data=(X_validation, y_validation)
)

# ==========================================
# 4. RESULT
# ==========================================
def report(name, history):
    train = history["loss"]
    validation = history["val_loss"]
    best_validation_loss = min(validation)
    print(f"\n{'='*55}")
    print(f"RESULT: {name}")
    print(f"{'='*55}")
    print(f"Best Validation Loss : {best_validation_loss:.6f})")
    print(f"Last Train Loss            : {train[-1]:.6f}")
    print(f"Last Val Loss              : {validation[-1]:.6f}")
    print(f"{'='*55}")
    return best_validation_loss, validation[-1]
report("NORMAL (No Dropout)", history_normal)
report("DROPOUT (p=0.2)", history_dropout)

#model with dropout learns much more slowly
# RESULT: NORMAL (No Dropout)
# =======================================================
# Best Validation Loss : 0.487191)
# Last Train Loss            : 0.047039
# Last Val Loss              : 0.611971
# =======================================================

# =======================================================
# RESULT: DROPOUT (p=0.2)
# =======================================================
# Best Validation Loss : 0.380001)
# Last Train Loss            : 0.194372
# Last Val Loss              : 0.736208
# =======================================================