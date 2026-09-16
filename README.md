# Deep Learning

This repository contains a simple artificial neural network implementation built with Python and NumPy.

## Project Structure

- `ANN/activation_functions.py`: Activation functions and their derivatives.
- `ANN/basic_neural_training.py`: A layered neural network with batch-based forward and backward propagation, gradient updates, and a configurable training loop.
- `ANN/neuron_test.py`: Trains a neuron with the sigmoid activation function.
- `ANN/test_of_activation_func.py`: Visualizes activation functions and their derivatives with Matplotlib.

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

## Run

Run the neuron training example from the project root:

```powershell
python ANN\neuron_test.py
```

Run the layered batch-training implementation from the project root:

```powershell
python ANN\basic_neural_training.py
```

The network processes multiple samples at once using matrix operations. Its `fit()` method combines forward propagation, loss calculation, backward propagation, parameter updates, and training-loss visualization. The number of training epochs can be provided by the caller.

To view the activation function plots, open `ANN/test_of_activation_func.py` in VS Code and run the cells in order, starting with the import cell.

