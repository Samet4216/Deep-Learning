# Chapter 3: Classification — Progress Report

This chapter transforms our from-scratch NumPy neural network framework from a regression-only engine into a **multi-purpose classification system** capable of detecting anomalies in UAV sensor data.

> **Goal:** Extend the existing architecture with Softmax, Cross-Entropy, advanced evaluation metrics, and deploy it on a real-time UAV anomaly detection pipeline.

---

## 1. Module Structure

```text
ANN/
├── activation_functions.py     # + Softmax activation (numerically stable)
├── loss_functions.py           # + Categorical & Binary Cross-Entropy (loss + derivative)
├── basic_neural_training.py    # Multi-purpose training pipeline (planned integration)
├── metrics.py                  # + Confusion Matrix, Precision, Recall, F1-Score (upcoming)
DATA/
├── dataset.py                  # + One-Hot Encoding utility (to_one_hot)
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
| **3.6** | Class Imbalance & Weighted Cross-Entropy | Upcoming | Penalizing minority-class misses in defense/UAV scenarios. |
| **3.7** | Label Smoothing (Modern Regularization) | Upcoming | Preventing overconfident Softmax outputs. |
| **3.8** | Advanced Metrics: Confusion Matrix, Precision, Recall, F1-Score | Upcoming | Macro vs. Weighted averaging, harmonic mean rationale. |
| **3.9** | Decision Boundary Visualization (2D Contour Plots) | Upcoming | Visualizing how the network separates classes in feature space. |
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

- **Max-Shift Trick:** Before computing $e^{z_i}$, the maximum value is subtracted from all logits ($z_i - z_{\max}$) to prevent numerical overflow. This does not change the result because:

$$
\frac{e^{z_i - C}}{\sum e^{z_j - C}} = \frac{e^{z_i}}{\sum e^{z_j}}
$$

- **Why not `a * (1 - a)` like Sigmoid?** Unlike Sigmoid where each neuron is independent, Softmax outputs are interconnected (the denominator contains all neurons). Its derivative is a full $C \times C$ **Jacobian matrix**, not a simple element-wise vector. This is why we combine it with Cross-Entropy instead.

### Categorical Cross-Entropy Loss (`loss_functions.py`)

Measures how "surprised" the model is when comparing its predicted probabilities against the true labels:

$$
L = -\frac{1}{N}\sum_{k=1}^{N} \sum_{i=1}^{C} y_{k,i} \log(\hat{y}_{k,i})
$$

- One-Hot encoded $y$ acts as a filter: zeros kill all wrong-class terms, leaving only $L = -\log(\hat{y}_{\text{correct}})$.
- **Clipping:** `np.clip(y_pred, 1e-15, 1 - 1e-15)` prevents $\log(0) = -\infty$ singularity.
- **Why not MSE for classification?** MSE combined with Softmax causes **Vanishing Gradients**: when the model is confidently wrong (output near 0 or 1), Sigmoid/Softmax derivatives approach zero, multiplying the large MSE error signal down to near-zero. Cross-Entropy bypasses this because the $\frac{1}{a}$ term from $\log$'s derivative cancels the $a$ term from Softmax's derivative.

### Backpropagation: The $dZ = A - Y$ Derivation

The combined gradient of Softmax + Cross-Entropy simplifies via the chain rule:

$$
\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial a_k} \cdot \frac{\partial a_k}{\partial z_i} = a_i - y_i
$$

**Three-stage cancellation:**
1. One-Hot zeros eliminate all terms except the correct class.
2. $\log$'s derivative ($\frac{1}{a}$) cancels Softmax's output ($a$) in numerator/denominator.
3. The remaining expression collapses to $a_i - y_i$ (Prediction minus Truth).

This is why `softmax_derivative()` returns `1` — the derivative is already folded into the loss derivative, and multiplying by 1 preserves the combined gradient.

### Binary Cross-Entropy (`loss_functions.py`)

For 2-class problems using a single Sigmoid output neuron:

$$
L = -\big[y \cdot \log(P) + (1-y) \cdot \log(1-P)\big]
$$

Its combined derivative also simplifies to $dZ = (P - y) / N$, identical in structure to the multi-class case. Sigmoid is mathematically a 2-class special case of Softmax.

### One-Hot Encoding (`dataset.py`)

Converts integer class labels into binary vectors:

```text
labels = [0, 2, 1, 0, 3]    (5 samples, 4 classes)

Output:
[[1, 0, 0, 0],    ← Class 0
 [0, 0, 1, 0],    ← Class 2
 [0, 1, 0, 0],    ← Class 1
 [1, 0, 0, 0],    ← Class 0
 [0, 0, 0, 1]]    ← Class 3
```

- **Why?** Raw integers imply false ordering ($3 > 2 > 1$). One-Hot vectors are equidistant, preventing the network from learning nonexistent hierarchies.
- **Implementation:** Vectorized NumPy advanced indexing (`one_hot[np.arange(N), labels] = 1`) — no Python loops.

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

*Chapter 3 is actively in progress. Next steps: Class Imbalance handling, Label Smoothing, advanced metrics, and the UAV anomaly detection capstone project.*
