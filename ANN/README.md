# Neural Network From Scratch

A neural network implementation built from scratch with **Python + NumPy**, developed step by step to understand how neural networks actually work instead of relying on high-level deep learning frameworks.

The project currently focuses on **ANN fundamentals**: neurons, activation functions, layers, forward propagation, loss, gradient descent, backpropagation, mini-batch training, and optimizers.

> **Goal:** Learn the mathematics and engineering of neural networks by implementing the core mechanisms ourselves.

---

## Current Progress

### Chapter 1 — ANN Fundamentals

- [x] Computational neuron
- [x] Weights and bias
- [x] Dot product
- [x] Activation functions
- [x] ReLU
- [x] Leaky ReLU
- [x] Sigmoid
- [x] Activation derivatives
- [x] Dense layer representation
- [x] Forward propagation
- [x] Mean Squared Error (MSE)
- [x] Gradient calculation
- [x] Gradient descent
- [x] Backpropagation
- [x] Mini-batch training
- [x] Batch shuffling
- [x] SGD
- [x] Momentum
- [x] Adam
- [x] Multi-layer neural network
- [x] XOR gate
- [x] Numerical vs. analytical gradient checking
- [x] Initial training-pipeline refactoring

Chapter 1 is now in the final cleanup stage.

---

## Why Build It From Scratch?

High-level libraries make neural networks easy to use, but they can hide the mechanics that matter when learning.

This project deliberately implements the important pieces ourselves:

```text
Input
  ↓
Weighted sum
  ↓
Activation
  ↓
Prediction
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Parameter update
  ↓
Repeat
```

The goal is not to compete with PyTorch or TensorFlow.
The goal is to understand what those frameworks are doing underneath.

---

## Neural Network Architecture

The implementation currently follows this separation of responsibilities:

```text
Neuron
 └── basic neuron calculations

Layer
 ├── forward()
 └── backward()

NeuralNetwork
 ├── forward()
 ├── predict()
 ├── backward()
 ├── update()
 ├── train_step()
 └── fit()

Optimizer
 └── update()
```

## 1. Computational Neuron

The basic neuron computes:

$$
z = x \cdot w + b
$$

where:

- `x` = input
- `w` = weights
- `b` = bias
- `z` = pre-activation value

The implementation uses NumPy's dot product:

```python
z = np.dot(input, self.weights) + self.bias
```

The output is then passed through an activation function.

---

## 2. Activation Functions

Activation functions introduce non-linearity into the network.
Implemented activation functions include:

### ReLU

$$
f(z)=\max(0,z)
$$

ReLU is simple and efficient, but a neuron can become stuck in the negative region where its derivative is zero.

### Leaky ReLU

Leaky ReLU keeps a small non-zero slope for negative values:

$$
f(z)=
\begin{cases}
z & z>0 \\
\alpha z & z\le0
\end{cases}
$$

This became especially important during the XOR experiment, where ReLU units became inactive and stopped providing useful gradients.

### Sigmoid

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

Sigmoid maps values into the interval `(0, 1)` and is currently used for the XOR output layer.

---

## 3. Dense Layers

A layer with multiple neurons is represented using a weight matrix.

For a layer with:

- `N` samples
- `input_size = 3`
- `neuron_count = 4`

the shapes are:

```text
X : (N, 3)
W : (3, 4)
b : (4,)
Z : (N, 4)
A : (N, 4)
```

Forward propagation is:

$$
Z = XW+b
$$

followed by:

$$
A=f(Z)
$$

Each column of `W` corresponds to one neuron.

---

## 4. Loss Function

The current loss is Mean Squared Error:

$$
L = \frac{1}{N}\sum_{i=1}^{N}\left(y_{\text{true},i}-y_{\text{pred},i}\right)^2
$$

Implemented separately in `loss_functions.py`.

The derivative used during training is:

$$
\frac{\partial L}{\partial A} = \frac{2(A-y)}{N}
$$

where `N` is the total number of elements being averaged.

---

## 5. Backpropagation

The layer computes:

$$
dZ = dA \odot f'(Z)
$$

Then the weight gradient:

$$
dW=X^TdZ
$$

Bias gradient:

$$
db=\sum dZ
$$

Gradient passed to the previous layer:

$$
dA_{\text{pre}}=dZW^T
$$

The important distinction is:

```text
dW
→ gradient for this layer's weights

db
→ gradient for this layer's biases

dA_pre
→ gradient sent to the previous layer
```

`dW` and `db` are stored on the layer:

```python
self.dW = dW
self.db = ...
```

while `dA_pre` is returned so the network can continue backpropagating.

---

## 6. Gradient Descent

The basic update rule is:

$$
W_{\text{new}}=W-\eta dW
$$

and:

$$
b_{\text{new}}=b-\eta db
$$

where `η` is the learning rate.

A positive gradient means the loss increases as the parameter increases, so subtracting the gradient moves the parameter in the opposite direction.

Learning rate controls the size of the step.

Experiments showed that:

- very small learning rates can make training extremely slow,
- larger learning rates can speed up learning,
- overly aggressive settings can make training unstable.

---

## 7. Mini-Batch Training

Training was extended from full-batch updates to mini-batches.

For each epoch:

```text
shuffle indices
    ↓
split indices into batches
    ↓
forward
    ↓
loss
    ↓
backward
    ↓
optimizer update
```

The dataset itself is kept unchanged during shuffling. Instead, permutation indices are generated and used to select each batch.

Example:

```text
original indices:
[0, 1, 2, 3]

permutation:
[2, 0, 3, 1]

batch_size = 2

batch 1 → [2, 0]
batch 2 → [3, 1]
```

This produces the same shuffled batches without replacing the original `X` and `y` arrays.

---

## 8. Optimizers

### SGD

SGD applies the current gradient directly:

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

### Momentum

Momentum keeps a running velocity:

$$
v_t=\beta v_{t-1}-\eta g_t
$$

$$
\theta_t=\theta_{t-1}+v_t
$$

This gives the optimizer memory of previous updates.

### Adam

Adam combines first- and second-moment estimates:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

Bias correction is applied:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Then the parameter update is:

$$
\theta_t=
\theta_{t-1}
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Adam keeps its optimizer state separately from the layers.

The optimizer interface also contains a `step()` method. For Adam, this advances the optimizer time used by bias correction.

---

## 9. XOR Gate

The first multi-layer learning problem is XOR:

```text
Input   Target

0 0       0
0 1       1
1 0       1
1 1       0
```

The network architecture is:

```text
2 inputs
   ↓
4 hidden neurons
Leaky ReLU
   ↓
1 output neuron
Sigmoid
   ↓
prediction
```

In shape form:

```text
(4, 2)
   ↓
(4, 4)
   ↓
(4, 1)
```

The network successfully learned XOR.

A representative final result was:

```text
epochs = 1000
[[0.0036851 ]
 [0.98938295]
 [0.98936128]
 [0.01289357]]
```

which corresponds closely to:

```text
0
1
1
0
```

The training loss reached approximately:

```text
0.000210
```

during the successful run.

---

## 10. ReLU and the XOR Failure

The XOR experiment also exposed the practical problem known as a dying ReLU.

At one point, the hidden layer produced:

```text
[0,0] → [1.517..., 0, 0, 0]
[0,1] → [0, 0, 0, 0]
[1,0] → [0, 0, 0, 0]
[1,1] → [0, 0, 0, 0]
```

The output layer therefore received the same hidden representation for the final three samples.

It could not distinguish them.

Their targets were:

```text
1
1
0
```

and their average is:

$$
\frac{1+1+0}{3}=\frac23
$$

which explains why the output became approximately `0.6667`.

Switching the hidden activation to Leaky ReLU allowed gradients to continue flowing through negative pre-activation values, and XOR training succeeded.

---

## 11. Gradient Checking

Backpropagation was independently checked using numerical differentiation.

For a selected weight:

$$
\frac{\partial L}{\partial w}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}{2\epsilon}
$$

The analytical gradient and numerical gradient were:

```text
Analytic:
-5.805755821445017e-05

Numerical:
-5.805756153609564e-05

Difference:
3.3216454692580437e-12
```

The extremely small difference strongly confirms that the implemented multi-layer backpropagation is computing the expected gradient for the tested parameter.

---

## Project Structure

The current project is intentionally still simple:

```text
ANN/
├── activation_functions.py
├── basic_neural_training.py
├── compare.py
├── loss_functions.py
├── neuron_test.py
├── optimizers.py
├── test_of_activation_func.py
├── xor_gate.py
```

The project started as a collection of small experiments and is gradually being cleaned into a more structured neural-network implementation.

---

## Refactoring Principles

During the cleanup, the following responsibilities were separated:

### Layer

Responsible for:

- forward propagation
- backward propagation
- storing `dW` and `db`

### NeuralNetwork

Responsible for:

- passing data through layers
- coordinating backpropagation
- coordinating optimizer steps
- training over epochs and batches
- exposing prediction functionality

### Optimizer

Responsible for:

- maintaining optimizer-specific state
- updating weights and biases

### Losses

Responsible for loss calculations.

### Example scripts

Responsible for experiments such as XOR, comparisons, and visualization.

---

## Current API

A network can be built by combining layers and an optimizer:

```python
optimizer = Adam(learning_rate=0.1)

network = NeuralNetwork(optimizer)

network.add(
    Layer(
        input_size=2,
        neuron_count=4,
        activation="leaky_relu",
        derivative="leaky_relu_derivative"
    )
)

network.add(
    Layer(
        input_size=4,
        neuron_count=1,
        activation="sigmoid",
        derivative="sigmoid_derivative"
    )
)
```

Training:

```python
loss_history = network.fit(
    X,
    y,
    epochs=1000,
    batch_size=4
)
```

Prediction:

```python
prediction = network.predict(X)
```

---

## Dependencies

The current implementation uses:

- Python
- NumPy
- Matplotlib

Install the dependencies with:

```bash
pip install numpy matplotlib
```
---

The objective is to understand what happens inside a neural network well enough to implement the core ideas independently.

---

<div align="center">

<a href="https://www.linkedin.com/in/abdussamet-türkoğlu-493544324?utm_source=share_via&utm_content=profile&utm_medium=member_android">LinkedIn</a>

### Samet Türkoğlu

Questions, suggestions, and contributions are welcome.

</div>
