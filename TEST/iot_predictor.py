
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
import pickle

if __name__ == "__main__":
    file_path = "../DATA/iot_data.csv"
    X, y, class_weights, scaler = preprocess_iot(file_path, sample_size=200000)
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
    model.add(Layer(input_size=input_features, neuron_count=128, activation="leaky_relu", derivative="leaky_relu_derivative", dropout_rate=0.2))
    model.add(Layer(input_size=128, neuron_count=64, activation="leaky_relu", derivative="leaky_relu_derivative", dropout_rate=0.1))
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
    print("===========0.2 0.1 DROPOUT=====================")
    print("learning_rate:", model.optimizer.learning_rate, "sample_size:", len(X_train), "batch_size:", 256, "epochs:", 250, "patience:", 25)
    y_pred = model.predict(X_test)
    class_names = ["Normal", "DoS", "Probe", "R2L", "U2R"]
    classification_report(y_test, y_pred, class_names)

    trained_weights = np.array(model.get_weights(), dtype=object)
    np.savez("../DATA/iot_model_weights.npz",
             weights=trained_weights,
             class_weights=class_weights,
             scaler_mean=scaler.mean,
             scaler_std=scaler.std
             )

    """
    [EARLY STOPPING] Stopped at epoch 51
    [EARLY STOPPING] Best score was at epoch 26 (Val Loss: 0.024704)
    [EARLY STOPPING] Restored best weights from epoch 26.
    ============NO DROPOUT=====================
    learning_rate: 0.001 sample_size: 140000 batch_size: 256 epochs: 250 patience: 25

    =======================================================
    CLASSIFICATION REPORT
    =======================================================
    Class Name      Precision    Recall       F1-Score    
    -------------------------------------------------------
    Normal          0.8116       0.9189       0.8619      
    DoS             0.9993       0.9996       0.9995      
    Probe           0.7092       0.6123       0.6572      
    R2L             0.7993       0.6540       0.7194      
    U2R             0.9982       0.9935       0.9959      
    -------------------------------------------------------
    Macro Avg       0.8635       0.8357       0.8468      
    =======================================================

    CONFUSION MATRIX:
    [[  646     0    21    36     0]
    [    0 27015     7     0     3]
    [   62     7   139    19     0]
    [   88     1    29   223     0]
    [    0    10     0     1  1693]]

    
    =======================================================
    =======================================================

    [EARLY STOPPING] Stopped at epoch 81
    [EARLY STOPPING] Best score was at epoch 56 (Val Loss: 0.022557)
    [EARLY STOPPING] Restored best weights from epoch 56.
    ===========0.2 0.1 DROPOUT=====================
    learning_rate: 0.001 sample_size: 140000 batch_size: 256 epochs: 250 patience: 25

    =======================================================
    CLASSIFICATION REPORT
    =======================================================
    Class Name      Precision    Recall       F1-Score    
    -------------------------------------------------------
    Normal          0.7957       0.9531       0.8673      
    DoS             0.9994       0.9997       0.9995      
    Probe           0.7857       0.5815       0.6684      
    R2L             0.8550       0.6569       0.7430      
    U2R             0.9982       0.9930       0.9956      
    -------------------------------------------------------
    Macro Avg       0.8868       0.8368       0.8547      
    =======================================================

    CONFUSION MATRIX:
    [[  670     0     8    25     0]
    [    0 27016     6     0     3]
    [   77     6   132    12     0]
    [   95     1    21   224     0]
    [    0    10     1     1  1692]]
    =======================================================
    """