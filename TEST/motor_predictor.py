# u_q, u_d: Voltage
# i_q, i_d: Current
# ambient, coolant: environmental and collected water temperature
# motor_speed: RPM (Revolutions Per Minute) (TARGET)
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ANN"))

import numpy as np
import pandas as pd 
from DATA.dataset import split_dataset
from DATA.scaler import StandardScaler
from ANN.basic_neural_training import NeuralNetwork, Layer
from ANN.optimizers import Adam
from ANN.metrics import regression_report
import matplotlib.pyplot as plt

#df=> data frame
df = pd.read_csv(PROJECT_ROOT / "DATA" / "PMSM.csv")
df = df.sample(n = 20000, random_state = 42)

features = ['u_q', 'u_d', 'i_q', 'i_d', 'ambient', 'coolant']
target = 'motor_speed'
X = df[features].values #convert to numpy array ==> [u_q, u_d, i_q, i_d, ambient, coolant]
y = df[target].values.reshape(-1, 1) #convert to numpy array ==> [motor_speed] and reshape column vector
# (-1, 1)=> -1 means "unspecified" and 1 means "one column".

X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, train_ratio=0.70, validation_ratio=0.15)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler() #because the target variable (motor_speed) is continuous, we can also scale it. this can help the neural network converge faster and improve performance.
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)
# 1. fit_transform(X_train): The scaler MUST ONLY look at the training data.
#    It learns the mean and standard deviation from X_train (fit) and applies it.
# 
# 2. transform(X_val & X_test): Validation and Test sets are unseen exams.
#    The scaler must NEVER learn from them (do NOT use fit here). We only 
#    apply (transform) the scaling metrics that were learned from X_train.

model = NeuralNetwork(Adam(learning_rate=0.001))
model.add(Layer(input_size=6, neuron_count=32, activation='relu', derivative='relu_derivative', dropout_rate=0.1))
model.add(Layer(input_size=32, neuron_count=16, activation='relu', derivative='relu_derivative', dropout_rate=0.05))
model.add(Layer(input_size=16, neuron_count=1, activation='linear', derivative='linear_derivative'))

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=100,
    batch_size=64,
    validation_data=(X_val_scaled, y_val_scaled),
    patience=10,
    min_delta=0.0001, #the minimum change in the monitored quantity to qualify as an improvement.
    restore_best_weights=True
)

y_pred_scaled = model.predict(X_test_scaled) #predict on the test set
y_pred = scaler_y.inverse_transform(y_pred_scaled) #inverse transform the predictions
regression_report(y_test, y_pred) #report the regression metrics

plt.figure(figsize=(14, 5)) #create a new figure with a specific size

plt.subplot(1, 2, 1) #1 row, 2 columns, 1st subplot
plt.plot(history["loss"], label="Training Loss", color='blue', linewidth=2)
plt.plot(history["val_loss"], label="Validation Loss", color='orange', linewidth=2)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True) #add grid lines

plt.subplot(1, 2, 2) #1 row, 2 columns, 2nd subplot
plt.plot(y_test[:100], label="ACTUAL RPM", color="green", linewidth=1.5) 
plt.plot(y_pred[:100], label="PREDICTED RPM", color="blue",linestyle='--', linewidth=1.5)
plt.title("Predicted vs Actual")
plt.xlabel("Sample Index")
plt.ylabel("RPM")
plt.legend()
plt.grid(True) 

plt.tight_layout()
plt.savefig('../DATA/motor_results.png', dpi=300)
print("Plot saved as DATA/motor_results.png")
plt.show()

np.savez('../DATA/motor_model_weights.npz',   
        W1=model.layers[0].W, #weights of the first layer
        b1=model.layers[0].b, #biases of the first layer
        W2=model.layers[1].W,
        b2=model.layers[1].b,
        W3=model.layers[2].W,
        b3=model.layers[2].b,
        mean_X=scaler_X.mean, #mean of the features in the training set
        std_X=scaler_X.std, #standard deviation of the features in the training set
        mean_y=scaler_y.mean, #mean of the target variable in the training set
        std_y=scaler_y.std #standard deviation of the target variable in the training set
)
print("Model weights and scaler parameters saved as DATA/motor_model_weights.npz")

# ==================================================
# REGRESSİON REPORTS
# ==================================================
#   MSE : 9190.0576 
#   RMSE : 95.8648
#   MAE : 39.6635
#   R^2 : %99.74 (Skor: 0.9974)
# ==================================================

# MAE=> calculate the average of all the errors the model made [this training: 2961-3039 RPM arranged]
# RMSE=> It generally has an error of 39 RPM
#but even during the motor's most aggressive moments the deviation does not exceed an average of 95 RPM.
# R^2=> trust score [between 0 and 1] 1 means perfect prediction
