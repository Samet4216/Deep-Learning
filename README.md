# Deep Learning

This repository is a step-by-step deep learning study project. Each chapter explores a different concept by implementing the underlying ideas with Python and NumPy before moving to high-level frameworks.

## Chapter 1 — Artificial Neural Networks

**Status: Completed**

The goal of Chapter 1 was to understand the basic mechanics of an artificial neural network by building the training pipeline from scratch.

### What We Built

- Computational neurons with weights and bias
- ReLU, Leaky ReLU, and Sigmoid activation functions with derivatives
- Dense layers and vectorized forward propagation
- Mean Squared Error loss
- Gradient calculation and gradient descent
- Backpropagation using the chain rule
- Mini-batch training with shuffled batch indices
- SGD, Momentum, and Adam optimizers
- A multi-layer neural network API with `fit()` and `predict()` methods
- XOR gate training with a `2-4-1` network
- Numerical versus analytical gradient checking
- A comparison of Adam and SGD optimizer behavior

### Results

The network successfully learned the XOR relationship. The experiment also demonstrated the dying ReLU problem: when hidden units became inactive, the model could not distinguish some samples. Replacing ReLU with Leaky ReLU allowed gradients to continue flowing and produced correct XOR predictions.

The analytical and numerical gradients were also compared, with a very small difference, confirming that the implemented backpropagation was working as expected.

## Project Structure

```text
Deep Learning/
├── ANN/                       # Completed Chapter 1: ANN fundamentals
│   ├── activation_functions.py
│   ├── basic_neural_training.py
│   ├── compare.py             # Adam vs. SGD comparison
│   ├── loss_functions.py
│   ├── neuron_test.py
│   ├── optimizers.py
│   ├── test_of_activation_func.py
│   ├── xor_gate.py
│   └── README.md              # Detailed Chapter 1 documentation
├── pdf/                       # Chapter 1 PDF notes and images
│   ├── README.md
│   └── chapter-1.pdf
├── images_readme/
└── README.md
```

## Chapter 1 PDF

The complete Chapter 1 notes are also available as a downloadable PDF, together with the images used to document the work.

- [Open the Chapter 1 PDF](pdf/chapter-1.pdf)
- [View the PDF notes and images](pdf/README.md)

## Setup

Create and activate the project virtual environment:

```powershell
python -m venv .deep_venv
.\.deep_venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install numpy matplotlib
```

## Run Chapter 1 Examples

Run the XOR gate experiment:

```powershell
python ANN\xor_gate.py
```

Run the optimizer comparison:

```powershell
python ANN\compare.py
```

Run the neural-network test script:

```powershell
python ANN\neuron_test.py
```