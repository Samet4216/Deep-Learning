# Chapter 3: Classification Engine & Real-Time AI Firewall

This chapter advances our custom-built neural network framework from a regression engine into an industrial-grade **Multi-Class Classification System**. Implemented entirely from mathematical first principles using **pure NumPy** (zero PyTorch, TensorFlow, or Scikit-learn), the system culminates in a **Real-Time IoT Intrusion Detection System (AI Firewall)** capable of filtering high-throughput network packets with millisecond latency.

---

## 1. Chapter 3 Roadmap & Milestones

| Step | Milestone | Status | Technical Implementation Details |
| :---: | :--- | :---: | :--- |
| **3.1** | Softmax Activation & Max-Shift Stability | **Completed** | Vectorized $\text{Softmax}$ in `activation_functions.py` with $z - \max(z)$ shift to eliminate float64 overflows. |
| **3.2** | Categorical Cross-Entropy (CCE) Loss | **Completed** | Log-loss with dynamic clipping ($\epsilon = 10^{-15}$) to prevent $\log(0)$ mathematical singularities. |
| **3.3** | Analytical Backpropagation ($dZ = A - Y$) | **Completed** | Full chain-rule cancellation proof between Softmax Jacobian and CCE, yielding simplified linear gradient $(A - Y) / N$. |
| **3.4** | Binary (BCE) vs. Multi-Class (CCE) Formulation | **Completed** | Implemented BCE and unified Sigmoid as the binary special case of Softmax ($C=2$). |
| **3.5** | One-Hot Encoding Transformation | **Completed** | Vectorized NumPy array mapping in `dataset.py` without slow Python iterations. |
| **3.6** | Class Imbalance & Weighted Cross-Entropy | **Completed** | Dynamic class weighting $w_c = N / (C \cdot N_c)$ to heavily penalize misses on ultra-rare intrusion classes. |
| **3.7** | Label Smoothing Regularization | **Completed** | Soft target redistribution $y_{\text{smooth}} = y(1 - \alpha) + \frac{\alpha}{C}$ to curb overconfident overparameterization. |
| **3.8** | Advanced Metrics Engine | **Completed** | Pure NumPy Confusion Matrix, Precision, Recall, and F1-Score (Macro Average) in `metrics.py`. |
| **3.9** | 2D Decision Boundary Spatial Scanner | **Completed** | Meshgrid-based planar scanner in `visualization.py` mapping non-linear decision contours. |
| **3.10** | Dataset Realism Pivot | **Completed** | Pivoted away from toy synthetic datasets to a real-world 140,000-sample Kaggle IoT Intrusion benchmark. |
| **3.11** | Dependency Injection & Decoupled Engine | **Completed** | Refactored `NeuralNetwork.compile()` to dynamically accept arbitrary loss functions via SOLID design. |
| **3.12** | **Capstone Project: IoT AI Firewall** | **Completed** | Trained 4-layer network, eliminated severe overfitting via Dropout & $L_2$, exported weights, and deployed an interactive CLI. |

---

## 2. Mathematical Foundations

### 2.1 Numerically Stable Softmax
Softmax maps an unconstrained logit vector $z \in \mathbb{R}^C$ into a normalized probability distribution:

$$
\sigma(z)_i = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^{C} e^{z_j - \max(z)}}
$$

* **The Max-Shift Trick:** Direct calculation of $e^{z_i}$ triggers floating-point overflow (`inf`) when logits exceed $\approx 709$. Subtracting $\max(z)$ ensures the largest exponent is $e^0 = 1$, guaranteeing absolute numerical stability without changing the output distribution:

$$
\frac{e^{z_i - c}}{\sum_j e^{z_j - c}} = \frac{e^{-c} e^{z_i}}{e^{-c} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### 2.2 Categorical Cross-Entropy (CCE) & Loss Clipping
Measures divergence between true probability distribution $Y$ and predicted probabilities $\hat{Y}$:

$$
\mathcal{L} = -\frac{1}{N} \sum_{k=1}^{N} \sum_{i=1}^{C} y_{k,i} \log\left(\text{clip}(\hat{y}_{k,i}, 10^{-15}, 1 - 10^{-15})\right)
$$

### 2.3 The Analytical Backpropagation Miracle ($dZ = A - Y$)
While the derivative of Softmax alone is an intractable $C \times C$ Jacobian matrix:

$$
\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j)
$$

combining it with Categorical Cross-Entropy through the multi-variable chain rule yields a clean cancellation:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^C \frac{\partial \mathcal{L}}{\partial a_k} \frac{\partial a_k}{\partial z_i} = -\frac{y_i}{a_i} \cdot a_i(1 - a_i) + \sum_{k \neq i} \left( -\frac{y_k}{a_k} \right) \cdot (-a_k a_i)
$$

$$
\frac{\partial \mathcal{L}}{\partial z_i} = -y_i(1 - a_i) + \sum_{k \neq i} y_k a_i = -y_i + a_i \sum_{k=1}^C y_k
$$

Since true labels are One-Hot encoded ($\sum_{k=1}^C y_k = 1$):

$$
dZ = A - Y
$$

### 2.4 Dynamic Weighted Loss for Severe Imbalance
In asymmetric defense/cyber environments, attack classes can represent $<0.1\%$ of total traffic. Unweighted networks default to predicting the majority class. We introduce class-specific penalties:

$$
\mathcal{L}_{\text{weighted}} = -\frac{1}{N} \sum_{k=1}^N \sum_{i=1}^C w_i \cdot y_{k,i} \log(\hat{y}_{k,i}) \quad \implies \quad dZ = w \odot (A - Y)
$$

where weights are inversely proportional to class frequencies:

$$
w_c = \frac{N_{\text{total}}}{C \cdot N_c}
$$

---

## 3. Architecture & Project Structure

```text
Deep Learning/
├── ANN/
│   ├── activation_functions.py   # Vectorized Softmax with max-shift stability
│   ├── basic_neural_training.py  # Core NeuralNetwork & Layer classes (compile, fit, predict)
│   ├── loss_functions.py         # Categorical, Binary, and Weighted Cross-Entropy + derivatives
│   ├── metrics.py                # Pure NumPy confusion matrix & classification report
│   └── optimizers.py             # Adam, RMSProp, Momentum, and SGD optimizers
├── DATA/
│   ├── dataset.py                # Preprocessing, train/val/test split, one-hot, label smoothing
│   ├── scaler.py                 # Custom StandardScaler with numerical zero-protection
│   ├── iot_data.csv              # Full Kaggle IoT network traffic dataset (47 columns)
│   ├── IoT_sample.csv            # Extracted multi-class evaluation sample
│   └── iot_model_weights.npz     # Serialized model brain (weights + scaler mean/std)
├── TEST/
│   ├── iot_predictor.py          # Production training script (140k samples, Adam, early stopping)
│   ├── predict_iot.py            # Real-time interactive CLI packet tester
│   ├── test_classification.py    # Multi-class integration test suite
│   └── visualization.py          # 2D Decision boundary spatial scanner
└── pdf-chapter3/
    └── README-3.md               # Chapter 3 documentation report
```

### Dependency Injection Pattern
The core training engine in `basic_neural_training.py` is decoupled from loss mathematics. Loss functions and derivatives are dynamically injected at runtime:

```python
model = NeuralNetwork(Adam(learning_rate=0.001))
model.compile(
    loss_function=lambda yt, yp: lf.weighted_categorical_cross_entropy(yt, yp, class_weights),
    loss_derivative=lambda yt, yp: lf.weighted_categorical_cross_entropy_derivative(yt, yp, class_weights)
)
```

---

## 4. Capstone Project: Real-Time IoT AI Firewall

### 4.1 Problem Formulation & Dataset
* **Domain:** Cyber-Physical / IoT Network Intrusion Detection
* **Input Dimension:** 46 continuous & discrete network features (flow duration, header length, protocol flags, packet rates, variance, covariance, etc.)
* **Output Dimension:** 5 mutually exclusive traffic categories:
  1. `Normal`: Legitimate background network communications
  2. `DoS`: Denial-of-Service volumetric flood attacks
  3. `Probe`: Network reconnaissance and vulnerability port scanning
  4. `R2L`: Remote-to-Local unauthorized access attempts
  5. `U2R`: User-to-Root privilege escalation attacks (ultra-rare: $<0.1\%$)

### 4.2 Engineering Challenges & Solutions

```text
[Challenge 1: Severe Imbalance]  --->  Robin-Hood Inverse Class Weights  --->  U2R Recall: 99.3%
[Challenge 2: Severe Overfitting]  --->  Dropout (0.2/0.1) + L2 Decay      --->  Macro F1: 85.5%
[Challenge 3: Division by Zero]   --->  Epsilon (+ 1e-8) Scaler Safeguard  --->  Eliminated NaN Crashes
[Challenge 4: Multiline CLI I/O]  --->  Regex Stream Buffer ([eE][-+]?)     --->  Robust Real-Time Inference
```

1. **Extreme Class Imbalance:** DoS attacks accounted for $>90\%$ of packets, while U2R accounted for $<0.1\%$. The weighted loss successfully forced the network to prioritize rare intrusion vectors.
2. **Combating Epoch 52 Overfitting:** The baseline MLP memorized DoS patterns and failed on subtle R2L intrusions. Implementing **Inverted Dropout (0.2 on Layer 1, 0.1 on Layer 2)** combined with **$L_2$ weight decay** broke co-adaptations, boosting R2L recall from $16\%$ to $65.7\%$.
3. **The Silent NaN Bug (Zero-Variance Features):** Static network features (e.g., legacy IRC flags) produced $\sigma = 0$. Normalizing incoming test packets caused zero-division infinities that corrupted the forward pass. Resolved by injecting an epsilon safeguard: `(X - mean) / (std + 1e-8)`.
4. **Resilient Terminal Streaming:** Designed a buffer accumulator with scientific notation regex (`[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?`) enabling operators to copy-paste multiline arrays (including values like `5.51e-06`) directly into the CLI.

---

## 5. Benchmark Performance & Evaluation

### 5.1 Progression: Baseline vs. Regularized AI Firewall

| Metric | Baseline MLP (10k samples, No Dropout) | Optimized Firewall (140k samples, Dropout ) | Improvement |
| :--- | :---: | :---: | :---: |
| **Normal F1** | 0.5797 | **0.8673** | **+49.6%** |
| **DoS F1** | 0.9981 | **0.9995** | High Precision Maintained |
| **Probe F1** | 0.4000 | **0.6684** | **+67.1%** |
| **R2L F1** | 0.2667 | **0.7430** | **+178.6%** |
| **U2R F1** | 0.9953 | **0.9956** | Flawless Rare Detection |
| **Macro Average F1** | **0.6480** | **0.8547** | **+31.9% Overall Gain** |

### 5.2 Final Classification Report (140,000 Training Samples)

```text
=======================================================
CLASSIFICATION REPORT (IoT AI FIREWALL)
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
```

### 5.3 Confusion Matrix Breakdown

```text
                  Predicted
             Normal    DoS   Probe    R2L    U2R
  Normal   [[   670      0       8     25      0 ]
  DoS      [      0  27016       6      0      3 ]
Actual
  Probe    [     77      6     132     12      0 ]
  R2L      [     95      1      21    224      0 ]
  U2R      [      0     10       1      1   1692 ]]
```

* **Key Observation:** 224 R2L intrusions correctly blocked (vs. only 4 in the initial baseline). U2R intrusions achieved $99.3\%$ recall (1,692 caught out of 1,704).

---

## 6. How to Run & Verify

### 6.1 Interactive Packet Inspection CLI
Test individual 46-feature network packets in real-time with automatic normalization and Softmax probabilities:

```bash
python TEST/predict_iot.py
```

Paste any comma-separated or whitespace-separated packet string:
```text
copy-paste the 46 features (for exit 'q'):
> 48.27, 128041.4, 10.4, 97.8, 39966.18, 0.0, ..., 1.0, 244.6

=============================================
RESULTS:
=============================================
Normal    : %   8.12
DoS       : %   0.01
Probe     : %   1.45
R2L       : %  90.41
U2R       : %   0.01
=============================================
Best class: [R2L]
```

### 6.2 Retraining the Pipeline
To re-execute the training cycle with early stopping, dynamic class weights, and weight export:

```bash
python TEST/iot_predictor.py
```

---

## 7. Key Takeaways & Transition to Chapter 4

1. **Analytical vs. Framework Differentiation:** Writing analytical backpropagation manually reveals optimization nuances (such as Jacobian cancellation and division-by-zero stability) that high-level abstractions like PyTorch obscure.
2. **Architectural Parity:** A pure NumPy multi-layer perceptron with proper regularization (Dropout, $L_2$, Class Weights, Label Smoothing) can achieve industrial-grade metrics ($\approx 85.5\%$ Macro F1) on tabular data without GPU overhead.
3. **Hardware Ready:** The lightweight nature of NumPy arrays allows the exported `.npz` weights to be deployed directly on embedded systems, UAV onboard computers, and microcontrollers.

