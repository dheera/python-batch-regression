#!/usr/bin/env python3

import torch

class BatchLinearRegression:
    def __init__(self, device: torch.device = None, precision="float32"):
        """
        Initialize the BatchLinearRegression instance.

        Args:
            device (torch.device, optional): The device on which to perform computations.
                                               Defaults to CUDA if available.
            precision (str or torch.dtype, optional): The numerical precision to use. Options:
                                                        "float32" (default), "float64", or "float16".
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Convert precision from string to torch.dtype if necessary.
        if isinstance(precision, str):
            precision = precision.lower()
            if precision == "float32":
                self.precision = torch.float32
            elif precision == "float64":
                self.precision = torch.float64
            elif precision == "float16":
                self.precision = torch.float16
            else:
                raise ValueError("Unsupported precision value: " + precision)
        else:
            self.precision = precision

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform batch linear regression (y = A*x + B) on the provided x, y pairs.
        Each row in x and y is treated as an independent regression problem.

        Args:
            x (torch.Tensor): Tensor of shape [B, N] for the independent variable.
            y (torch.Tensor): Tensor of shape [B, N] for the dependent variable.

        Returns:
            slope (torch.Tensor): Tensor of shape [B] containing slopes for each regression.
            intercept (torch.Tensor): Tensor of shape [B] containing intercepts for each regression.
            r2 (torch.Tensor): Tensor of shape [B] containing the R² values.
        """
        # Move x and y to the desired device and precision.
        x = x.to(self.device).to(self.precision)
        y = y.to(self.device).to(self.precision)

        # Compute means for each batch (each row).
        mean_x = x.mean(dim=1, keepdim=True)
        mean_y = y.mean(dim=1, keepdim=True)

        # Center the data.
        x_centered = x - mean_x
        y_centered = y - mean_y

        # Compute the numerator and denominator for the slope.
        numerator = (x_centered * y_centered).sum(dim=1)       # Covariance between x and y.
        denominator = (x_centered ** 2).sum(dim=1)               # Variance of x.

        # Compute slope: avoid division by zero.
        slope = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))

        # Compute intercept.
        intercept = mean_y.squeeze(1) - slope * mean_x.squeeze(1)

        # Compute standard deviations for correlation.
        std_x = torch.sqrt(denominator)
        std_y = torch.sqrt((y_centered ** 2).sum(dim=1))
        denom_corr = std_x * std_y

        # Compute correlation coefficient and then R².
        r = torch.where(denom_corr != 0, numerator / denom_corr, torch.zeros_like(numerator))
        r2 = r ** 2

        return slope, intercept, r2


# Example usage:
if __name__ == '__main__':
    import time

    # Parameters for simulation: 1000 independent regressions, each with 10000 samples.
    B = 1048576   # number of regression problems (batches)
    N = 1000  # number of samples per regression

    torch.manual_seed(42)

    print("Creating data")

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

    t = time.time()

    # Perform batch regression.
    slope, intercept, r2 = regressor.fit(x, y)

    print(f"Run time: {time.time() - t:.02f} sec")

    print("Mean slope:", slope.mean().item())
    print("Mean intercept:", intercept.mean().item())
    print("Mean R²:", r2.mean().item())

