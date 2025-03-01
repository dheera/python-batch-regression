#!/usr/bin/env python3
import torch

def adf_test(time_series: torch.Tensor, regression: str = "c", lags: int = 0) -> torch.Tensor:
    """
    Compute the ADF (Augmented Dickey-Fuller) test statistic for a batch of time series.
    
    The ADF regression is:
        Δy_t = α + [β t] + γ y_{t-1} + ∑_{i=1}^{lags} δ_i Δy_{t-i} + ε_t
    where the term in square brackets is included only for regression='ct'.
    
    The test statistic is the t-statistic for γ (the coefficient on y_{t-1}).
    Under the null hypothesis (unit root), γ = 0.
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] where B is the number of series
                                      and T is the number of time points.
        regression (str): 'c' for constant, 'ct' for constant and trend.
        lags (int): Number of lagged difference terms to include.
    
    Returns:
        torch.Tensor: A tensor of shape [B] containing the ADF test statistic for each series.
    """
    B, T = time_series.shape
    # We need at least (lags + 2) observations (one for y_{t-1} and one for Δy_t)
    if T <= lags + 1:
        raise ValueError("Time series is too short for the specified number of lags.")
    
    # Compute first differences: Δy_t = y_t - y_{t-1}, shape [B, T-1]
    diff_y = time_series[:, 1:] - time_series[:, :-1]
    # We'll regress for t = (lags) ... (T-1) [in terms of the diff index]
    M = T - lags - 1  # number of usable observations in the regression
    
    # Dependent variable: Δy from index lags to end, shape [B, M]
    Y = diff_y[:, lags:]
    
    # Build the regressor matrix X of shape [B, M, k]
    X_parts = []
    
    # Constant term (if regression is 'c' or 'ct')
    ones = torch.ones(B, M, 1, dtype=time_series.dtype, device=time_series.device)
    X_parts.append(ones)
    
    # Trend term if regression is 'ct'
    if regression.lower() == "ct":
        # Create a trend variable starting at 1 to M
        trend = torch.linspace(1, M, M, device=time_series.device, dtype=time_series.dtype)
        trend = trend.unsqueeze(0).expand(B, M).unsqueeze(2)  # shape [B, M, 1]
        X_parts.append(trend)
    elif regression.lower() != "c":
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")
    
    # Lagged level term: y_{t-1} corresponding to t = lags+1,...,T-1, shape [B, M, 1]
    y_lag = time_series[:, lags:T-1].unsqueeze(2)
    X_parts.append(y_lag)
    
    # Add lagged differences Δy_{t-i} for i = 1 to lags.
    # For each i, the regressor is Δy_{t-i} for t = lags+1,...,T-1.
    # That is, for each series, column for lag i is diff_y[:, (lags - i):(T - 1 - i)]
    for i in range(1, lags + 1):
        lagged_diff = diff_y[:, (lags - i):(T - 1 - i)].unsqueeze(2)
        X_parts.append(lagged_diff)
    
    # Concatenate all parts along the last dimension. k = 1 (constant) + (1 if 'ct') + 1 (lagged level) + lags.
    X = torch.cat(X_parts, dim=2)  # shape [B, M, k]
    
    # Number of regressors.
    k = X.shape[2]
    
    # OLS: For each batch, beta = (X'X)^{-1} X'Y.
    # Compute X'X for each batch: shape [B, k, k]
    Xt = X.transpose(1, 2)  # shape [B, k, M]
    XtX = torch.bmm(Xt, X)
    # Compute X'Y for each batch: shape [B, k, 1]
    XtY = torch.bmm(Xt, Y.unsqueeze(2))
    # Solve for beta.
    beta = torch.linalg.solve(XtX, XtY).squeeze(2)  # shape [B, k]
    
    # Get fitted values and residuals.
    Y_hat = (X * beta.unsqueeze(1)).sum(dim=2)  # shape [B, M]
    residuals = Y - Y_hat  # shape [B, M]
    
    # Estimate variance of residuals for each batch: sigma^2 = (residuals' * residuals) / (M - k)
    dof = M - k
    sigma2 = (residuals ** 2).sum(dim=1) / dof  # shape [B]
    
    # The coefficient on the lagged level term is at index 1 if regression=='c', or 2 if regression=='ct'.
    if regression.lower() == "c":
        idx_gamma = 1  # [constant, gamma, ...]
    else:  # "ct"
        idx_gamma = 2  # [constant, trend, gamma, ...]
    
    gamma_coeff = beta[:, idx_gamma]  # shape [B]
    
    # Compute the standard error of gamma coefficient.
    # Standard error: sqrt(sigma2 * [ (X'X)^{-1} ]_{gamma,gamma} )
    inv_XtX = torch.linalg.inv(XtX)  # shape [B, k, k]
    se_gamma = torch.sqrt(sigma2 * inv_XtX[:, idx_gamma, idx_gamma])  # shape [B]
    
    # The ADF test statistic is the t-statistic for gamma: gamma_coeff / se_gamma.
    adf_stat = gamma_coeff / se_gamma
    
    return adf_stat

if __name__ == '__main__':
    # Example usage:
    import time

    # Create a batch of time series.
    B = 5      # number of series
    T = 200    # number of time points
    torch.manual_seed(42)
    
    # Simulate non-stationary series (random walks).
    random_walk = torch.zeros(B, T)
    random_walk[:, 0] = torch.randn(B)
    for t in range(1, T):
        random_walk[:, t] = random_walk[:, t-1] + torch.randn(B) * 0.2

    # Simulate stationary series (mean reverting around zero).
    stationary_series = torch.randn(B, T) * 0.5
    
    # Compute ADF test statistics.
    # For a random walk, we expect the ADF statistic to be closer to 0 (fail to reject unit root).
    print("ADF statistic for random walk series (regression='c', lags=5):")
    adf_stat_rw = adf_test(random_walk, regression="c", lags=5)
    print(adf_stat_rw)

    # For a stationary series, the ADF statistic should be more negative.
    print("\nADF statistic for stationary series (regression='c', lags=5):")
    adf_stat_stat = adf_test(stationary_series, regression="c", lags=5)
    print(adf_stat_stat)

