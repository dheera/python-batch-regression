# PyTorch Batch Regression Library

A Python library for performing PyTorch-accelerated regression analyses and time series statistical testing in batch. This library is designed for efficiently processing millions of regression problems simultaneously on a GPU – for example, solving ~1 million parallel linear regression problems with 500-1000 points each in a couple of seconds on a V100 GPU. Much faster than conventional CPU-based libraries like SciPy!

It also includes time series statistical tests that can be done in batches on a GPU.

## Features

### Regression Models
- **Linear Regression:** Simple and multiple linear regression models.
- **Quadratic Regression:** Fits quadratic relationships to data.
- **Cubic Regression:** Handles cubic regression modeling.
- **Nonlinear Regression:** Supports customizable nonlinear regression models.
- **Multilinear Regression:** Efficient estimation of models with multiple predictors.
- **Quadratic Regression:** For modeling second-degree polynomial relationships.

### Time Series Statistical Testing
- **KPSS Test:** Tests for stationarity around a constant or a deterministic trend. Returns the KPSS statistic (and trend coefficients if applicable).
- **ADF Test:** Performs the Augmented Dickey-Fuller test to check for unit roots; supports both constant-only and constant-plus-trend models.
- **Johansen Test:** Assesses cointegration among multiple time series by computing eigenvalues and trace statistics.
- **Chow Test:** Tests for a structural break at a known break date by comparing RSS from pooled and segmented regressions.
- **QLR Test (Sup-Wald):** Searches over a range of candidate break dates to detect an unknown structural break.
- **CUSUM Test:** Computes the cumulative sum (CUSUM) of residuals to detect parameter instability over time.
- **CUSUM of Squares Test:** Uses the cumulative sum of squared residuals to detect abrupt changes in variance.
- **ICSS Algorithm:** Iterative Cumulative Sum of Squares method for detecting multiple variance breakpoints.
- **Bai–Perron Test:** A dynamic programming approach to detect multiple structural breaks (with model selection via BIC).
- **Hurst Exponent Estimation:** Estimates the Hurst exponent using rescaled range (R/S) analysis to gauge long-term memory in time series.

## Installation

To install the library, clone the repository and install it using pip:

```bash
git clone <repository-url>
cd <repository-directory>
pip install .
```

## Usage

See the individual .py files for examples of how to use all the above tests.

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
