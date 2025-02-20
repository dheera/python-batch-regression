import torch

class BatchCubicRegression:
    def __init__(self, device: torch.device = None, precision="float32"):
        """
        Initialize the BatchCubicRegression instance.

        Args:
            device (torch.device, optional): The device on which to perform computations.
                                               Defaults to CUDA if available.
            precision (str or torch.dtype, optional): The numerical precision to use.
                                                        Options: "float32" (default), "float64", or "float16".
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
        Perform batch cubic regression (y = A*x^3 + B*x^2 + C*x + D) on the provided x, y pairs.
        Each row in x and y is treated as an independent regression problem.

        Args:
            x (torch.Tensor): Tensor of shape [B, N] for the independent variable.
            y (torch.Tensor): Tensor of shape [B, N] for the dependent variable.

        Returns:
            coeffs (torch.Tensor): Tensor of shape [B, 4] containing the coefficients [A, B, C, D] for each regression.
            r2 (torch.Tensor): Tensor of shape [B] containing the R² values for each regression.
        """
        # Move x and y to the desired device and precision.
        x = x.to(self.device).to(self.precision)
        y = y.to(self.device).to(self.precision)
        
        B, N = x.shape
        ones = torch.ones_like(x)
        
        # Build the design matrix X with columns: [x^3, x^2, x, 1]
        X = torch.stack([x**3, x**2, x, ones], dim=2)  # shape: [B, N, 4]
        
        # Compute X^T X for each batch: shape [B, 4, 4]
        X_transpose = X.transpose(1, 2)  # shape: [B, 4, N]
        XTX = torch.bmm(X_transpose, X)   # shape: [B, 4, 4]
        
        # Compute X^T y for each batch: shape [B, 4, 1]
        XTy = torch.bmm(X_transpose, y.unsqueeze(2))  # shape: [B, 4, 1]
        
        # Solve for coefficients using the pseudo-inverse for robustness.
        # beta = (X^T X)^(-1) X^T y
        XTX_pinv = torch.linalg.pinv(XTX)  # shape: [B, 4, 4]
        beta = torch.bmm(XTX_pinv, XTy).squeeze(2)  # shape: [B, 4]
        
        # Compute predicted y values: y_pred = A*x^3 + B*x^2 + C*x + D
        y_pred = beta[:, 0:1] * (x**3) + beta[:, 1:2] * (x**2) + beta[:, 2:3] * x + beta[:, 3:4]
        
        # Calculate R²: r2 = 1 - (SS_res / SS_tot)
        ss_res = ((y - y_pred) ** 2).sum(dim=1)  # Sum of squared residuals per batch
        mean_y = y.mean(dim=1, keepdim=True)
        ss_tot = ((y - mean_y) ** 2).sum(dim=1)    # Total sum of squares per batch
        
        # Avoid division by zero for cases with zero variance.
        r2 = torch.where(ss_tot != 0, 1 - ss_res / ss_tot, torch.zeros_like(ss_res))
        
        return beta, r2


# Example usage:
if __name__ == '__main__':
    # Simulate 300 independent cubic regressions, each with 500 data points.
    B = 300   # Number of regression problems (batches)
    N = 500   # Number of samples per regression

    torch.manual_seed(42)
    
    # Create random x values.
    x = torch.randn(B, N)
    
    # Define true cubic parameters.
    true_A = 1.0
    true_B = -0.5
    true_C = 2.0
    true_D = 0.3
    
    # Generate y values with noise.
    noise = torch.randn(B, N) * 0.1
    y = true_A * (x**3) + true_B * (x**2) + true_C * x + true_D + noise
    
    # Instantiate the cubic regressor with default precision (float32)
    regressor = BatchCubicRegression(precision="float32")
    
    # Fit the model.
    coeffs, r2 = regressor.fit(x, y)
    
    print("Mean estimated coefficients [A, B, C, D]:", coeffs.mean(dim=0))
    print("Mean R²:", r2.mean().item())

