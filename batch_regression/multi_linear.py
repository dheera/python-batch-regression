#!/usr/bin/env python3
import torch

class BatchMultiLinearRegression:
    def __init__(self, device: torch.device = None, precision="float32"):
        """
        Initialize the BatchMultiLinearRegression instance.

        Args:
            device (torch.device, optional): The device on which to perform computations.
                                               Defaults to CUDA if available.
            precision (str or torch.dtype, optional): The numerical precision to use.
                                                        Options: "float32" (default), "float64", or "float16".
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Convert precision from string to torch.dtype if needed.
        if isinstance(precision, str):
            precision = precision.lower()
            if precision == "float32":
                self.precision = torch.float32
            elif precision == "float64":
                self.precision = torch.float64
            elif precision == "float16":
                self.precision = torch.float16
            else:
                raise ValueError("Unsupported precision: " + precision)
        else:
            self.precision = precision

    def fit(self, x: torch.Tensor, y: torch.Tensor):
        """
        Perform batch multivariate linear regression (y = A*x + B) on the provided x, y pairs.
        Each batch is treated as an independent regression problem.

        Args:
            x (torch.Tensor): Tensor of shape [B, N, d] where d is the number of features.
            y (torch.Tensor): Tensor of shape [B, N] containing the target values.

        Returns:
            A (torch.Tensor): Tensor of shape [B, d] containing the regression coefficients.
            B (torch.Tensor): Tensor of shape [B] containing the intercepts.
            r2 (torch.Tensor): Tensor of shape [B] containing the R² values.
        """
        # Move data to the desired device and precision.
        x = x.to(self.device).to(self.precision)
        y = y.to(self.device).to(self.precision)

        B_batch, N, d = x.shape

        # Compute means for each batch.
        mean_x = x.mean(dim=1)  # Shape: [B, d]
        mean_y = y.mean(dim=1)  # Shape: [B]

        # Center the data.
        X_centered = x - mean_x.unsqueeze(1)  # Shape: [B, N, d]
        y_centered = y - mean_y.unsqueeze(1)    # Shape: [B, N]

        # Compute the covariance matrix of X for each batch. 
        # This gives us a [B, d, d] matrix per batch.
        cov_x = torch.bmm(X_centered.transpose(1, 2), X_centered)
        
        # Compute the cross-covariance between X and y.
        # Shape: [B, d, 1]
        cov_xy = torch.bmm(X_centered.transpose(1, 2), y_centered.unsqueeze(2))

        # Solve for coefficients A using the batched normal equation.
        # A is computed as (cov_x)^{-1} * cov_xy.
        # Note: In practice you might add a small regularization term to cov_x if it is close to singular.
        A = torch.linalg.solve(cov_x, cov_xy)  # Shape: [B, d, 1]
        A = A.squeeze(2)  # Shape: [B, d]

        # Compute intercept B for each batch: B = mean_y - mean_x dot A.
        B_param = mean_y - (mean_x * A).sum(dim=1)  # Shape: [B]

        # Compute predictions: y_pred = x dot A + B.
        y_pred = torch.bmm(x, A.unsqueeze(2)).squeeze(2) + B_param.unsqueeze(1)

        # Compute R² for each batch.
        # Total sum of squares.
        total_sum_sq = ((y - mean_y.unsqueeze(1)) ** 2).sum(dim=1)
        # Residual sum of squares.
        residual_sum_sq = ((y - y_pred) ** 2).sum(dim=1)
        # R² = 1 - (RSS / TSS). Use torch.where to avoid division by zero.
        r2 = 1 - torch.where(total_sum_sq != 0, residual_sum_sq / total_sum_sq, torch.zeros_like(total_sum_sq))

        return A, B_param, r2


# Example usage:
if __name__ == '__main__':
    import time

    # Simulation parameters:
    #  - B: number of independent regressions (batches)
    #  - N: number of samples per regression
    #  - d: number of features (i.e. A1, A2, ..., Ad)
    B = 100000   # number of regression problems
    N = 1000   # samples per regression
    d = 3      # number of features

    torch.manual_seed(42)

    print("Creating data")

    # Generate random feature data. Shape: [B, N, d]
    x = torch.randn(B, N, d)

    # Define true regression parameters (same for all batches for simplicity).
    true_A = torch.tensor([2.0, -1.5, 0.5]).expand(B, d)  # Shape: [B, d]
    true_B = 3.0

    # Generate target values using the model: y = x @ true_A^T + true_B.
    y = torch.bmm(x, true_A.unsqueeze(2)).squeeze(2) + true_B

    # Add noise to the target.
    noise = torch.randn(B, N) * 0.1
    y = y + noise

    print("Running regression")

    # Instantiate the regression class (default precision: float32)
    regressor = BatchMultiLinearRegression(precision="float32")

    t = time.time()

    # Perform batch regression.
    A_est, B_est, r2 = regressor.fit(x, y)

    print(f"Run time: {time.time() - t:.02f} sec")
    print("Mean estimated coefficients:", A_est.mean(dim=0).tolist())
    print("Mean estimated intercept:", B_est.mean().item())
    print("Mean R²:", r2.mean().item())

