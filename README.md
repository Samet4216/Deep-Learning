# Deep Learning

This repository contains a simple artificial neural network implementation built with Python and NumPy.

## Project Structure

- `ANN/activation_functions.py`: Activation functions and their derivatives.
- `ANN/basic_neural_training.py`: A single-neuron model with forward propagation, gradients, and parameter updates.
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

To view the activation function plots, open `ANN/test_of_activation_func.py` in VS Code and run the cells in order, starting with the import cell.

