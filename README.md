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
from batch_regression.linear import LinearRegression
from batch_regression.quadratic import QuadraticRegression
from batch_regression.cubic import CubicRegression
from batch_regression.nonlinear import NonlinearRegression

# Example usage for linear regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```
