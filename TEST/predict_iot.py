
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ANN"))

import numpy as np
import ANN.loss_functions as lf
from ANN.basic_neural_training import NeuralNetwork, Layer
from DATA.dataset import  preprocess_iot, to_one_hot, split_dataset
from ANN.metrics import classification_report
from ANN.optimizers import Adam

if __name__ == "__main__":
    file_path = "../DATA/iot_data.csv"
    X, y, class_weights, scaler = preprocess_iot(file_path, sample_size=100000)
    y_one_hot = to_one_hot(y, class_count=5)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X,
        y_one_hot, 
        train_ratio=0.7, 
        validation_ratio=0.15,
        seed=42
        )

    input_features = X_train.shape[1] # 46 for this dataset
    model = NeuralNetwork(Adam(learning_rate=0.001))
    model.add(Layer(input_size=input_features, neuron_count=128, activation="leaky_relu", derivative="leaky_relu_derivative"))
    model.add(Layer(input_size=128, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative"))
    model.add(Layer(input_size=64, neuron_count=32, activation="leaky_relu", derivative="leaky_relu_derivative"))
    model.add(Layer(input_size=32, neuron_count=5, activation="softmax", derivative="softmax_derivative"))

    model.compile(loss_function=lf.categorical_cross_entropy, loss_derivative=lf.categorical_cross_entropy_derivative)

    print("Training the model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=250,
        batch_size=256,
        validation_data=(X_val, y_val),
        patience=25,
    )
    print("learning_rate:", model.optimizer.learning_rate, "sample_size:", len(X_train), "batch_size:", 256, "epochs:", 100, "patience:", 25)
    y_pred = model.predict(X_test)
    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    classification_report(y_test, y_pred, class_names)