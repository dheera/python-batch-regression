# Pytorch Batch Regression Library

A Python library for performing various regression analyses including linear, quadratic, cubic, and nonlinear regression models. This library is designed for batch processing of regression models on multiple datasets, making it easy to switch between different types of regressions as needed.

This can solve ~1 million linear regression problems with 500-1000 points in a couple seconds on a V100 GPU. Much faster than scipy!

## Features

- **Linear Regression:** Implements simple and multiple linear regression models.
- **Quadratic Regression:** Fits quadratic relationships to data.
- **Cubic Regression:** Handles cubic regression modeling.
- **Nonlinear Regression:** Supports customizable nonlinear regression models.

## Installation

To install the library, clone the repository and install it using pip:

```bash
git clone <repository-url>
cd <repository-directory>
pip install .
```

## Usage

```
B = 1048576   # number of regression problems (batches)
N = 1000  # number of samples per regression

print("Creating data")
torch.manual_seed(42)

# Create random x values.
x = torch.randn(B, N)

# Define true regression parameters.
true_slope = 2.0
true_intercept = 3.0

# Generate y values with added noise.
noise = torch.randn(B, N) * 0.1
y = true_slope * x + true_intercept + noise

print("Running regression")

# Instantiate the regression class (default precision: float32)
regressor = BatchLinearRegression(precision="float32")

# Perform batch regression.
slope, intercept, r2 = regressor.fit(x, y)

print("Mean slope:", slope.mean().item())
print("Mean intercept:", intercept.mean().item())
print("Mean R²:", r2.mean().item())
```
