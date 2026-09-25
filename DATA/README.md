# Chapter 2: Data & Regression — Progress Report

This module handles dataset preparation, splitting, feature scaling, and regularization techniques for the end-to-end regression pipeline.

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
| **2.6 Underfitting & Overfitting** | Bias-Variance Tradeoff, Loss Curves (Train vs Val) | **In Progress** | Deep theoretical and practical analysis of model complexity. |
| **2.6.1 Regularization & Anti-Overfitting** | L2 Regularization (Weight Decay), L1, Dropout, Early Stopping | **In Progress** | Mathematical formulation, backpropagation integration, and code design. |
| **2.7 Model Evaluation & Metrics** | MSE, RMSE, MAE, $R^2$ Score, Residual Plots | **Upcoming** | Quantitative and visual metrics to assess generalization. |
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
