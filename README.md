# Quantum Data Parallelism in Quantum Neural Networks

## Description
This repository contains the Python code for our research on quantum data parallelism in quantum neural networks. We explore two encoding methods, orthogonal and non-orthogonal, across different problems: XOR, Parity, Iris, and MNIST. The code includes implementations for training quantum circuits and analyzing their decision boundaries and performance.

## Requirements
- Python 3
- PennyLane==0.30.0
- TensorFlow==2.12.0
- TensorFlow Quantum==0.5.1
- Jupyter Notebook

## Usage
Navigate to the respective problem directory and run the Jupyter notebooks or Python scripts.

## Structure
- `Xor_ortho_loss.ipynb`: Train the quantum circuit with orthogonal inputs for the XOR problem.
- `Xor_nonortho_loss.ipynb`: Train the quantum circuit with non-orthogonal inputs for the XOR problem.
- `Xor_ortho_boundary.py`: Generate the decision boundary for the XOR problem with orthogonal encoding.
- `Xor_nonortho_boundary.py`: Generate the decision boundary for the XOR problem with non-orthogonal encoding.
- `Parity.ipynb`: Notebook for the Parity problem analysis.
- `Iris.ipynb`: Notebook for the Iris dataset analysis.
- `MNIST_ortho.ipynb`: Train and evaluate the MNIST dataset with orthogonal encoding.
- `MNIST_nonortho.ipynb`: Train and evaluate the MNIST dataset with non-orthogonal encoding.
