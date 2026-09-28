# Deep Learning

This repository is a step-by-step deep learning study project built completely from scratch. Each chapter explores fundamental deep learning concepts by implementing the underlying mathematics and algorithms using **Python and NumPy** before moving to high-level frameworks.

---

## Chapter 1 — Artificial Neural Networks

**Status: Completed**

The goal of Chapter 1 was to master the basic mechanics of an artificial neural network by constructing the entire forward and backward propagation pipeline from scratch.

### What We Built
- Computational neurons with weights and bias
- ReLU, Leaky ReLU, and Sigmoid activation functions with derivatives
- Dense layers and vectorized forward propagation ($Z = XW + b$)
- Mean Squared Error (MSE) loss
- Analytical backpropagation using the chain rule
- Mini-batch training with shuffled indices
- Custom optimizers: SGD, Momentum, and Adam (with bias correction)
- Numerical vs. analytical gradient checking
- XOR gate non-linear learning experiment (`2-4-1` architecture)
- Adam vs. SGD optimizer convergence comparison

### Results
- The network successfully learned the non-linear XOR function.
- Diagnosed and resolved the dying ReLU problem by transitioning to Leaky ReLU.
- Gradient checking verified backpropagation implementation with minimal numerical difference.

---

## Chapter 2 — Data, Regression & Regularization

**Status: Completed**

Chapter 2 focuses on data preparation, feature engineering, non-linear regression, and a comprehensive arsenal of regularization techniques to prevent overfitting, culminating in a real-world motor performance prediction system.

### What We Built
- **Data Splitting & Leakage Prevention:** `split_dataset()` utility with train/val/test partitions and seed reproducibility.
- **Feature & Target Scaling:** `MinMaxScaler` and `StandardScaler` with robust `epsilon` handling and `inverse_transform` to convert model outputs back to physical units.
- **Universal Approximation Theorem:** Verified that a single hidden layer with non-linear activation reduces error by **96.6%** compared to a purely linear baseline.
- **Anti-Overfitting & Regularization Arsenal:**
  - **L2 Regularization (Ridge / Weight Decay):** Smooths weights via $dW += \lambda_2 W$.
  - **L1 Regularization (Lasso):** Induces sparsity via $dW += \lambda_1 \text{sign}(W)$, achieving **79.5% weight sparsity**.
  - **Inverted Dropout:** Applied during training with $1/(1-p)$ scaling to preserve signal expectation with zero test-time overhead.
  - **Early Stopping:** Monitored validation loss with configurable `patience` and automatic historical restoration of optimal weights (`restore_best_weights=True`).
- **Quantitative Evaluation Metrics:** Built a standalone evaluation module (`ANN/metrics.py`) providing MSE, RMSE, MAE, and $R^2$ Score.
- **Capstone Project (`TEST/motor_predictor.py`):** Real-world Permanent Magnet Synchronous Motor (PMSM) rotational speed predictor.

### Capstone Project: PMSM Motor Speed Predictor

Using 20,000 real-world observations from an industrial PMSM motor, we trained a multi-layer regularized neural network to predict rotational speed (`motor_speed`, up to $6000\text{ RPM}$) using 6 electrical and thermal features (`u_q`, `u_d`, `i_q`, `i_d`, `ambient`, `coolant`).

```text
==================================================
REGRESSION REPORTS (PMSM MOTOR SPEED PREDICTOR)
==================================================
  MSE   : 9190.0576
  RMSE  : 95.8648 RPM
  MAE   : 39.6635 RPM
  R^2   : %99.74 (Skor: 0.9974)
==================================================
```

- **Accuracy:** Achieved an **$R^2$ score of 99.74%** and an average error of only **$39.66\text{ RPM}$** across the test set (< 0.7% relative error).
- **Efficiency:** Early Stopping terminated training at epoch 43 and restored the peak weights from **epoch 33** (`Val Loss: 0.001942`), saving over 57% of unnecessary training cycles.
- **Model Persistence:** Trained weights, biases, and normalization metrics are exported to `DATA/motor_model_weights.npz` using `np.savez()` for standalone inference.

![PMSM Motor Speed Prediction Results](DATA/motor_results.png)

> [!WARNING]
> Results may different between runs due to random weight initialization and dropout.

---

## Project Structure

```text
Deep Learning/
├── ANN/                                # Chapter 1 & Core Engine
│   ├── activation_functions.py         # Sigmoid, ReLU, Leaky ReLU, Linear & derivatives
│   ├── basic_neural_training.py        # Neuron, Layer, NeuralNetwork (Dropout, L1/L2, Early Stopping)
│   ├── compare.py                      # Adam vs. SGD comparison experiment
│   ├── loss_functions.py               # MSE loss
│   ├── metrics.py                      # Regression metrics (MSE, RMSE, MAE, R² score)
│   ├── neuron_test.py                  # Unit and gradient tests
│   ├── optimizers.py                   # SGD, Momentum, Adam
│   ├── test_of_activation_func.py      # Activation function visualization
│   ├── xor_gate.py                     # Non-linear XOR benchmark
│   └── README.md                       # Chapter 1 documentation
├── DATA/                               # Chapter 2: Data & Preprocessing
│   ├── dataset.py                      # Dataset generation & split_dataset utility
│   ├── scaler.py                       # MinMaxScaler & StandardScaler with inverse_transform
│   ├── motor_results.png               # PMSM final evaluation loss & prediction plot
│   ├── motor_model_weights.npz         # Exported weights, biases & scaler parameters
│   └── README.md                       # Chapter 2 progress report & documentation
├── TEST/                               # Chapter 2 Experiments & Capstone
│   ├── linear_vs_non-linear.py         # Linear model vs. hidden layer ANN
│   ├── overfitting.py                  # Overfitting sweet spot & L2 mitigation
│   ├── L1_vs_L2.py                     # Ridge vs. Lasso weight sparsity comparison
│   ├── dropout.py                      # Inverted dropout benchmark
│   ├── early_stopping.py               # Early stopping & best weight restoration
│   └── motor_predictor.py              # Chapter 2 Capstone: PMSM speed prediction
├── pdf/                                # PDF notes and resource documents
│   ├── README.md
│   └── chapter-1.pdf
├── images_readme/
└── README.md
```

---

## Setup

Create and activate the project virtual environment:

```powershell
python -m venv .deep_venv
.\.deep_venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install numpy matplotlib pandas
```

---

## Running the Code

### Chapter 1 Examples
```powershell
# Run the XOR gate experiment
python ANN\xor_gate.py

# Run the Adam vs. SGD optimizer comparison
python ANN\compare.py
```

### Chapter 2 Experiments
```powershell
# Linear vs. Non-linear regression
python TEST\linear_vs_non-linear.py

# Overfitting and L2 regularization
python TEST\overfitting.py

# L1 vs. L2 regularization (sparsity analysis)
python TEST\L1_vs_L2.py

# Inverted Dropout experiment
python TEST\dropout.py

# Early stopping demonstration
python TEST\early_stopping.py

# Chapter 2 Capstone: PMSM Motor Speed Predictor
python TEST\motor_predictor.py
```