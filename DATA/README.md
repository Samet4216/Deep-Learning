# Chapter 2: Data & Regression — Progress Report

This module handles dataset preparation, splitting, and feature scaling for the end-to-end regression pipeline.

---

## 1. Module Structure

```text
DATA/
├── dataset.py     # Dataset creation and train/validation/test splitting
├── scaler.py      # MinMaxScaler and StandardScaler implementations
└── README.md      # Chapter 2 progress log and documentation
```

---

## 2. Chapter 2 Roadmap & Progress

| Topic | Status | Implementation / Notes |
| :--- | :---: | :--- |
| **2.1 Dataset Logic** | **Completed** | Sample, Feature, Target definitions. Matrix representations ($X, y$). |
| **2.2 Train / Val / Test Split** | **Completed** | `split_dataset()` utility with strict ratio validation and no data leakage. |
| **2.3 Feature Scaling** | **Completed** | `MinMaxScaler` ($[0, 1]$) and `StandardScaler` (Z-score) with `fit`, `transform`, `inverse_transform`. |
| **2.4 Linear Regression** | **Completed** | Single-neuron linear network ($y = XW + b$) learning true parameters ($w, b$). |
| **2.5 Nonlinear Regression** | **Completed** | 16-neuron hidden layer + Leaky ReLU solving parabolic data with **96.6% loss reduction** over linear baseline (`TEST/linear_vs_non-linear.py`). |
| **2.6 Underfitting & Overfitting** | **In Progress** | Investigating model complexity, loss curves (Train vs Val), and generalization. |
| **2.7 Model Evaluation** | **Upcoming** | MSE, MAE, Prediction vs Actual plots, and generalization metrics. |
| **Final Project: Motor Predictor** | **Upcoming** | Full ML pipeline predicting motor RPM from Temperature, Voltage, Current, and Load. |

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
