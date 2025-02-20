import torch
from tqdm import tqdm

class BatchNonlinearRegression:
    def __init__(self, model_fn, device: torch.device = None, precision="float32"):
        """
        Initialize the BatchNonlinearRegression instance.

        Args:
            model_fn (callable): A function with signature model_fn(x, params) -> y_pred.
                                 x should be a tensor of shape [B, N] (B regressions, N data points)
                                 and params a tensor of shape [B, P] (P parameters per regression).
            device (torch.device, optional): The device for computation. Defaults to CUDA if available.
            precision (str or torch.dtype, optional): Numerical precision to use.
                                                        Options: "float32" (default), "float64", or "float16".
        """
        self.model_fn = model_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

    def fit(self, x: torch.Tensor, y: torch.Tensor, initial_params: torch.Tensor,
            lr=1e-3, n_iter=1000, verbose=True):
        """
        Perform batch nonlinear regression using gradient-based optimization.
        
        Args:
            x (torch.Tensor): Independent variable tensor of shape [B, N].
            y (torch.Tensor): Dependent variable tensor of shape [B, N].
            initial_params (torch.Tensor): Initial guess for parameters, shape [B, P].
            lr (float): Learning rate for the optimizer.
            n_iter (int): Number of optimization iterations.
            verbose (bool): If True, prints progress.
            
        Returns:
            params (torch.Tensor): Fitted parameters of shape [B, P].
            r2 (torch.Tensor): Coefficient of determination (R²) for each regression, shape [B].
        """
        # Move inputs to the desired device and precision.
        x = x.to(self.device).to(self.precision)
        y = y.to(self.device).to(self.precision)
        initial_params = initial_params.to(self.device).to(self.precision)
        
        # Create a learnable parameter tensor.
        params = torch.nn.Parameter(initial_params.clone())
        
        # Setup an optimizer (Adam) for the parameters.
        optimizer = torch.optim.Adam([params], lr=lr)
        
        # Optimization loop
        iterator = range(n_iter)
        if verbose:
            iterator = tqdm(iterator, desc="Optimizing")
            
        for i in iterator:
            optimizer.zero_grad()
            y_pred = self.model_fn(x, params)  # predicted y, shape: [B, N]
            loss = ((y - y_pred) ** 2).mean()
            loss.backward()
            optimizer.step()
            if verbose and (i % max(1, n_iter // 10) == 0):
                iterator.set_postfix({"loss": loss.item()})
        
        # Compute final predictions and R² values for each regression.
        with torch.no_grad():
            y_pred = self.model_fn(x, params)
            mean_y = y.mean(dim=1, keepdim=True)
            ss_tot = ((y - mean_y) ** 2).sum(dim=1)
            ss_res = ((y - y_pred) ** 2).sum(dim=1)
            r2 = torch.where(ss_tot != 0, 1 - ss_res / ss_tot, torch.zeros_like(ss_tot))
        
        return params.detach(), r2


# Example usage:
if __name__ == '__main__':
    # Here we consider a generic nonlinear model: y = a * sin(b * x + c) + d.
    # This model has 4 parameters per regression.
    def model_fn(x, params):
        # params is of shape [B, 4] containing [a, b, c, d]
        a = params[:, 0:1]
        b = params[:, 1:2]
        c = params[:, 2:3]
        d = params[:, 3:4]
        return a * torch.sin(b * x + c) + d

    # Number of independent regressions and data points per regression.
    B = 100  # number of regression problems
    N = 500  # data points per regression

    torch.manual_seed(42)
    
    # Generate x values in an interval (e.g., [-pi, pi]).
    x = torch.linspace(-3.14, 3.14, steps=N).unsqueeze(0).repeat(B, 1)
    
    # Create true parameters for each regression (varying slightly per batch).
    true_a = 2.0 + 0.1 * torch.randn(B, 1)
    true_b = 1.5 + 0.1 * torch.randn(B, 1)
    true_c = 0.5 + 0.1 * torch.randn(B, 1)
    true_d = 0.3 + 0.1 * torch.randn(B, 1)
    true_params = torch.cat([true_a, true_b, true_c, true_d], dim=1)
    
    # Generate y values from the model and add noise.
    y_clean = model_fn(x, true_params)
    noise = 0.1 * torch.randn(B, N)
    y = y_clean + noise
    
    # Provide an initial guess for the parameters.
    initial_params = torch.tensor([[1.0, 1.0, 0.0, 0.0]]).repeat(B, 1)
    
    # Instantiate and fit the model.
    regressor = BatchNonlinearRegression(model_fn, precision="float32")
    fitted_params, r2 = regressor.fit(x, y, initial_params, lr=1e-2, n_iter=5000, verbose=True)
    
    print("Mean fitted parameters:", fitted_params.mean(dim=0))
    print("Mean R²:", r2.mean().item())

