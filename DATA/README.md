ctrl+shift+V

# Chapter 2: Data & Regression — Progress Report

This module handles dataset preparation, splitting, feature scaling, and complete regularization techniques for the end-to-end regression pipeline.

---

## 1. Module Structure

```text
DATA/
├── dataset.py     # Dataset creation and train/validation/test splitting
├── scaler.py      # MinMaxScaler and StandardScaler implementations
└── README.md      # Chapter 2 progress log and documentation
```

---

## 2. Chapter 2 Comprehensive Roadmap & Progress

| Topic | Sub-topics | Status | Implementation / Notes |
| :--- | :--- | :---: | :--- |
| **2.1 Dataset Logic** | Sample, Feature, Target, Shapes ($X, y$) | **Completed** | Solidified matrix dimensions and data representations. |
| **2.2 Train / Val / Test Split** | Ratios, Data Leakage Prevention | **Completed** | `split_dataset()` with strict ratio assertions and seed support. |
| **2.3 Feature Scaling** | Min-Max Normalization vs Standardization | **Completed** | `MinMaxScaler` and `StandardScaler` with inverse transformation. |
| **2.4 Linear Regression** | Single-neuron $y = XW + b$, Loss & Gradients | **Completed** | Verified analytical parameter recovery ($w \approx [3, -2, 5], b \approx 10$). |
| **2.5 Nonlinear Regression** | Hidden Layer, Non-linear Activation, Universal Approximation | **Completed** | 16-neuron Hidden Layer + Leaky ReLU achieved **96.6% loss reduction** over linear baseline. |
| **2.6 Underfitting & Overfitting** | Bias-Variance Tradeoff, Sweet Spot Dynamics | **Completed** | Demonstrated overfitting U-curve in `TEST/overfitting_lab.py`. |
| **2.6.1 L2 Regularization (Ridge)** | Weight Decay, Shrinking Weight Magnitude | **Completed** | $dW += \lambda_2 W$, cut post-sweet-spot degradation in half (+29.7% $\to$ +15.6%). |
| **2.6.2 L1 Regularization (Lasso)** | Sparsity, Automatic Feature Selection | **Completed** | $dW += \lambda_1 \text{sign}(W)$, achieved **79.5% weight sparsity** by zeroing out connections. |
| **2.6.3 Inverted Dropout** | Co-adaptation Prevention, $1/(1-p)$ Scaling | **Completed** | Layer-level mask & scaling, improved best validation loss by **24%** (0.4871 $\to$ 0.3696). |
| **2.6.4 Early Stopping** | Patience, Automatic Training Halt, Weight Restoration | **Completed** | `get_weights()` and `set_weights()`, saved **94.7% compute time** (stopped at epoch 53/1000). |
| **2.7 Model Evaluation & Metrics** | MSE, RMSE, MAE, $R^2$ Score, Residual Analysis | **Upcoming** | Quantitative metrics to measure real regression performance. |
| **Chapter 2 Final Project** | `motor_performance_predictor` Pipeline | **Upcoming** | Raw Data $\to$ Split $\to$ Scale $\to$ Regularized ANN $\to$ Evaluate $\to$ Predict. |

---

## 3. Key Concepts Implemented

### Feature Scaling (`scaler.py`)
- **MinMaxScaler:** Squeezes features into $[0, 1]$ via $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min} + \epsilon}$.
- **StandardScaler:** Centers data around zero with unit variance via $x' = \frac{x - \mu}{\sigma + \epsilon}$.
- **Data Leakage Rule:** Parameters ($\mu, \sigma, \min, \max$) are computed exclusively on `X_train` via `.fit()`. Validation and test sets are transformed using train parameters.
- **Inverse Transformation:** Maps scaled predictions back to physical units (e.g., RPM).

### Non-Linear Representation (`TEST/linear_vs_non-linear.py`)
- Verified the **Universal Approximation Theorem**:
  - Linear Model ($y = wx + b$): Plateaued at MSE $\approx 1.8778$ (Underfitting).
  - Non-Linear ANN (Hidden Layer + Leaky ReLU): Dropped MSE to $\approx 0.0640$ (96.6% error reduction).

### Regularization & Anti-Overfitting Arsenal (`ANN/basic_neural_training.py`)
1. **L2 Regularization (Weight Decay):**
   - Loss Penalty: $\frac{\lambda_2}{2} \sum W^2$
   - Gradient Update: $dW += \lambda_2 W$
   - Effect: Smooths curves, shrinks weights toward zero without setting them to exact zero.
2. **L1 Regularization (Lasso):**
   - Loss Penalty: $\lambda_1 \sum |W|$
   - Gradient Update: $dW += \lambda_1 \text{sign}(W)$
   - Effect: Forces unimportant weights to exact $0.0$, creating sparse networks and performing automated feature selection.
3. **Inverted Dropout:**
   - Training: Scales active activations by $\frac{1}{1-p}$ via random Bernoulli masks to preserve signal expectation $\mathbb{E}[A] = A$.
   - Backward: Gradients masked with the exact same scaled binary mask ($dA \cdot M$).
   - Inference/Evaluation: Dropout is completely bypassed with zero test-time overhead.
4. **Early Stopping:**
   - Monitors validation loss with configurable `patience` and `min_delta`.
   - Takes deep copies of neuron parameters via `get_weights()`.
   - Restores peak historical weights via `set_weights()` upon halting, guaranteeing the optimal Sweet Spot model.
