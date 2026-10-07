# Chapter 3: Classification — Progress Report

This chapter transforms our from-scratch NumPy neural network framework from a regression-only engine into a **multi-purpose classification system** capable of detecting anomalies in UAV sensor data.

> **Goal:** Extend the existing architecture with Softmax, Cross-Entropy, advanced evaluation metrics, and deploy it on a real-time UAV anomaly detection pipeline.

---

## 1. Module Structure

```text
ANN/
├── activation_functions.py     # + Softmax activation (numerically stable)
├── loss_functions.py           # + Categorical, Binary & Weighted Cross-Entropy
├── basic_neural_training.py    # Multi-purpose training pipeline (planned integration)
├── metrics.py                  # + Confusion Matrix, Precision, Recall, F1-Score
DATA/
├── dataset.py                  # + One-Hot Encoding & Label Smoothing utilities
TEST/
├── visualization.py            # + Decision Boundary visualization tool
```

---

## 2. Chapter 3 Comprehensive Roadmap & Progress

| Step | Topic | Status | Implementation / Notes |
| :--- | :--- | :---: | :--- |
| **3.1** | Softmax Activation & Numerical Stability (Max-Shift) | **Completed** | `softmax()` in `activation_functions.py`. Subtracts row-max before exponentiation to prevent overflow. |
| **3.2** | Categorical Cross-Entropy (CCE) Loss & Information Theory | **Completed** | `categorical_cross_entropy()` in `loss_functions.py`. Uses `np.clip` to prevent $\log(0)$ singularity. |
| **3.3** | Backpropagation Proof: Softmax + CCE → $dZ = A - Y$ | **Completed** | `categorical_cross_entropy_derivative()` returns $(A - Y) / N$. Jacobian matrix cancellation proven analytically and verified numerically. |
| **3.4** | Binary (BCE) vs. Multi-Class (CCE) Architecture Comparison | **Completed** | `binary_cross_entropy()` and `binary_cross_entropy_derivative()` added. Sigmoid is the 2-class special case of Softmax. |
| **3.5** | One-Hot Encoding & Categorical Data Transformation | **Completed** | `to_one_hot()` in `dataset.py`. Vectorized NumPy indexing, no loops. |
| **3.6** | Class Imbalance & Weighted Cross-Entropy | **Completed** | `weighted_categorical_cross_entropy()` and derivative. Penalizes minority-class misses heavily. |
| **3.7** | Label Smoothing (Modern Regularization) | **Completed** | `apply_label_smoothing()`. Prevents overconfident predictions by softly distributing target probabilities. |
| **3.8** | Advanced Metrics: Confusion Matrix, Precision, Recall, F1-Score | **Completed** | `classification_report` and pure numpy `confusion_matrix` implemented. |
| **3.9** | Decision Boundary Visualization (2D Contour Plots) | **Completed** | `plot_decision_boundary()` with `meshgrid` for spatial mapping and scanning. |
| **3.10** | Synthetic UAV Sensor Data Generator | Upcoming | Physics-based flight simulator for Normal / Fault / Spoofing / Turbulence classes. |
| **3.11** | Multi-Purpose Model Integration (`basic_neural_training.py`) | Upcoming | Unified `loss_function` parameter for MSE / CCE / BCE selection. |
| **3.12** | **Final Project:** Real-Time UAV Anomaly & GPS Spoofing Detection | Upcoming | Live inference pipeline with alarm system. |
| **3.13** | Documentation: GitHub & LinkedIn Report | Upcoming | Comprehensive technical write-up. |

---

## 3. Key Concepts Implemented

### Softmax Activation (`activation_functions.py`)

Converts raw logits into a probability distribution where all outputs sum to 1:

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
$$

- **Max-Shift Trick:** Before computing $e^{z_i}$, the maximum value is subtracted from all logits ($z_i - z_{\max}$) to prevent numerical overflow.
- **Why not `a * (1 - a)` like Sigmoid?** Unlike Sigmoid where each neuron is independent, Softmax outputs are interconnected. Its derivative is a full $C \times C$ **Jacobian matrix**, not a simple element-wise vector. This is why we combine it with Cross-Entropy instead.

### Categorical Cross-Entropy Loss (`loss_functions.py`)

$$
L = -\frac{1}{N}\sum_{k=1}^{N} \sum_{i=1}^{C} y_{k,i} \log(\hat{y}_{k,i})
$$

- One-Hot encoded $y$ acts as a filter: zeros kill all wrong-class terms, leaving only $L = -\log(\hat{y}_{\text{correct}})$.

### Backpropagation: The $dZ = A - Y$ Derivation

The combined gradient of Softmax + Cross-Entropy simplifies via the chain rule:

$$
\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_i} = a_i - y_i
$$

### Weighted Cross-Entropy (Class Imbalance Handling)

In defense scenarios (e.g., UAVs), an attack class might represent $0.1\%$ of the data. Standard CCE causes the network to ignore these rare events. By introducing a class weight multiplier $w_c$, we severely penalize misclassifications on minority classes:

$$
L_{\text{weighted}} = - \sum_{i=1}^{C} w_i \cdot y_i \log(\hat{y}_i)
$$
The gradient scales proportionally: $dZ = w \odot (A - Y)$.

### Label Smoothing (Regularization)

Models trained with hard targets `[1.0, 0.0, 0.0]` tend to push weights to infinity, resulting in overconfidence and overfitting. Label Smoothing introduces a doubt factor $\alpha$ (e.g., $0.1$):

$$
y_{\text{smooth}} = y_{\text{true}} \times (1 - \alpha) + \frac{\alpha}{C}
$$
This transforms `[1.0, 0.0, 0.0]` into softer targets like `[0.933, 0.033, 0.033]`, preventing overconfidence on noisy sensor data.

### Decision Boundary Visualization

We use `np.meshgrid` to generate thousands of grid points across the 2D feature space. These points are flattened, passed through the model's `predict()` function, and refolded into an image matrix to map exactly how the network spatially separates distinct classes (regions).

---

## 4. Classification vs. Regression — Architecture Summary

| Component | Regression (Ch. 2) | Binary Classification | Multi-Class Classification |
| :--- | :---: | :---: | :---: |
| Output Neurons | 1 | 1 | $C$ (one per class) |
| Output Activation | Linear | Sigmoid | Softmax |
| Loss Function | MSE | Binary Cross-Entropy | Categorical Cross-Entropy |
| Gradient ($dZ$) | $\frac{2(A-Y)}{N}$ | $\frac{P-y}{N}$ | $\frac{A-Y}{N}$ |
| Target Format | Raw value | 0 or 1 | One-Hot vector |

---

*Chapter 3 is actively in progress. Next steps: Synthetic UAV Sensor Data Generator, Model Integration, and the final capstone project.*
