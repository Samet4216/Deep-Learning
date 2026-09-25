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
# 1. SMALL DATASET (30 EXAMPLE)
# ==========================================
np.random.seed(42)

X = np.random.uniform(-3, 3, (30, 1))
noise = np.random.randn(30, 1) * 0.50 
y = np.cos(X) + noise

X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.50,
    validation_ratio=0.40,
    seed=42
    )

# ==========================================
# 2. BİG MODEL SETUP 
# ==========================================
model = NeuralNetwork(Adam(learning_rate=0.01))
model.add(
    Layer(
        input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"
    )
)
model.add(
    Layer(
        input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"
    )
)
model.add(
    Layer(
        input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"
    )
)

# ==========================================
# 3. MODEL TRAİNİNG(500 EPOCH)
# ==========================================
history = model.fit(
    X_train,
    y_train,
    500,
    batch_size=len(X_train), #full batch
    validation_data=(X_validation, y_validation)
)
train_loss = history["loss"]
val_loss = history["val_loss"]

min_val_loss = min(val_loss)
sweet_spot_epoch = val_loss.index(min_val_loss) + 1

print("\n" + "="*55)
print("RAPOR OF OVERFİTTİNG")
print("="*55)
print(f"Best Validation Loss (Sweet Spot) : {min_val_loss:.6f}  (Epoch {sweet_spot_epoch})")
print(f"Last Epoch Train Loss     : {train_loss[-1]:.6f}")
print(f"Last Epoch Val Loss       : {val_loss[-1]:.6f}")
difference = ((val_loss[-1] - min_val_loss) / min_val_loss) * 100
print(f"Val Loss Percent         : + %{difference:.1f} UP!")
if val_loss[-1] > min_val_loss * 1.5:
    print(f"The model learned up to epoch {sweet_spot_epoch}")
print("="*55)

# =======================================================
# RAPOR OF OVERFİTTİNG
# =======================================================
# Best Validation Loss (Sweet Spot) : 0.137882  (Epoch 384)
# Last Epoch Train Loss     : 0.066804
# Last Epoch Val Loss       : 0.195134
# Val Loss Percent         : + %41.5 UP!
# =======================================================
# ->train_loss down ||| val_loss up


# =======================================================
# =======================================================
# ADD L2 REGULARIZATION
# =======================================================
# =======================================================

# 1. DATASET

np.random.seed(42)
X = np.random.uniform(-3, 3, (30, 1))
noise = np.random.randn(30, 1) * 0.50
y = np.cos(X) + noise

X_train, y_train, X_validation, y_validation, X_test, y_test = split_dataset(
    X,
    y,
    train_ratio=0.50,
    validation_ratio=0.40,
    seed=42
    )

# BIG MODEL SETUP 

model = NeuralNetwork(Adam(learning_rate=0.01), L2_lambda=0.001)
model.add(
    Layer(
        input_size=1, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"
    )
)
model.add(
    Layer(
        input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"
    )
)
model.add(
    Layer(
        input_size=32, neuron_count=1, activation="linear", derivative="linear_derivative"
    )
)

# 3. MODEL TRAİNİNG(300 EPOCH)

history = model.fit(
    X_train,
    y_train,
    500,
    batch_size=len(X_train),
    validation_data=(X_validation, y_validation)
)
L2_train_loss = history["loss"]
L2_val_loss = history["val_loss"]

min_L2_val_loss = min(L2_val_loss)
sweet_spot_epoch = L2_val_loss.index(min_L2_val_loss) + 1

print("\n" + "="*55)
print("RAPOR OF OVERFİTTİNG")
print("="*55)
print(f"Best Validation Loss (Sweet Spot) : {min_L2_val_loss:.6f}  (Epoch {sweet_spot_epoch})")
print(f"Last Epoch Train Loss     : {L2_train_loss[-1]:.6f}")
print(f"Last Epoch Val Loss       : {L2_val_loss[-1]:.6f}")
difference = ((L2_val_loss[-1] - min_L2_val_loss) / min_L2_val_loss) * 100
print(f"Val Loss Percent         : + %{difference:.1f} UP!")
print("="*55)
difference_l2_and_normal = train_loss[-1] - L2_train_loss[-1]
print(f"difference between normal and L2: {difference_l2_and_normal:.6f}")

# =======================================================
# RAPOR OF OVERFİTTİNG
# =======================================================
# Best Validation Loss (Sweet Spot) : 0.125508  (Epoch 388)
# Last Epoch Train Loss     : 0.045575
# Last Epoch Val Loss       : 0.152636
# Val Loss Percent         : + %21.6 UP!
# L2 regularization => 0.001
# =======================================================