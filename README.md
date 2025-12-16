# SimpleSVM

This project implements a linear Support Vector Machine (SVM) from scratch using Python. It uses a brute-force approach to find the optimal hyperplane that separates data points into two classes. The project also provides tools to visualize the decision boundary and compare the results with Scikit-learn's SVM implementation.

## Features

*   **Custom SVM Implementation**: A ground-up implementation of a linear SVM trained via a cost function optimization.
*   **Soft Margin**: Supports imperfectly separable data by incorporating a penalty parameter `C`.
*   **Visualization**: Built-in functions effectively map the decision boundary, margins, and support vectors using Matplotlib.
*   **Scikit-learn Comparison**: Includes a direct comparison script to validate the custom implementation against the industry-standard `svm.SVC`.
*   **Data Generation**: Utilities to generate synthetic datasets for testing and experimentation.

## How It Works

The core of the project is the `svm_train_brute` function. It iterates through possible support vector candidates (pairs and triplets of points) to calculate the optimal weight vector `w` and bias `b`.

The cost function used is:

`Cost = 0.5 * (1 / margin)^2 + C * TotalError`

Where `TotalError` helps account for misclassified points, allowing the model to handle non-linearly separable data to some extent (soft margin).

## Installation

You can install the package and its dependencies using pip. It is recommended to use a virtual environment.

```bash
pip install -e .
```

## Usage

To see the SVM in action and compare it with Scikit-learn, you can run the comparison script directly from your terminal:

```bash
python src/svm_project/compare.py
```

This command will:
1.  Generate a random dataset.
2.  Train the custom SVM.
3.  Train a Scikit-learn SVM on the same data.
4.  Display a side-by-side plot showing the decision boundaries of both models.

## Structure

*   `src/svm_project/data.py`: Functions for generating synthetic training data.
*   `src/svm_project/svm.py`: The main SVM implementation, including training, distance calculations, and testing logic.
*   `src/svm_project/compare.py`: Scripts for running the comparison against Scikit-learn.

## Requirements

*   Python 3.8+
*   NumPy
*   Matplotlib
*   Scikit-learn
