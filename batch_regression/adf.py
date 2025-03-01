#!/usr/bin/env python3

#!/usr/bin/env python3
import torch

def adf_test(time_series: torch.Tensor, regression: str = "c", lags: int = 0):
    """
    Compute the ADF (Augmented Dickey-Fuller) test statistic for a batch of time series,
    and also return the estimated trend slope when regression='ct'.

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
        adf_stat (torch.Tensor): A tensor of shape [B] containing the ADF test statistic for each series.
        trend_coef (torch.Tensor or None): If regression is 'ct', returns a tensor of shape [B] with the
                                             estimated trend slopes. Otherwise, returns None.
    """
    B, T = time_series.shape
    if T <= lags + 1:
        raise ValueError("Time series is too short for the specified number of lags.")

    # Compute first differences: Δy_t = y_t - y_{t-1}, shape [B, T-1]
    diff_y = time_series[:, 1:] - time_series[:, :-1]
    M = T - lags - 1  # number of usable observations

    # Dependent variable: Δy for observations from index lags to end, shape [B, M]
    Y = diff_y[:, lags:]

    # Build regressor matrix X for t = lags+1,...,T-1.
    X_parts = []

    # Constant term (always included)
    ones = torch.ones(B, M, 1, dtype=time_series.dtype, device=time_series.device)
    X_parts.append(ones)

    # Trend term if regression is 'ct'
    trend_coef = None
    if regression.lower() == "ct":
        # Create a trend variable from 1 to M for each series.
        trend = torch.linspace(1, M, M, device=time_series.device, dtype=time_series.dtype)
        trend = trend.unsqueeze(0).expand(B, M).unsqueeze(2)  # shape [B, M, 1]
        X_parts.append(trend)
    elif regression.lower() != "c":
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")

    # Lagged level term: y_{t-1} for t = lags+1,...,T-1, shape [B, M, 1]
    y_lag = time_series[:, lags:T-1].unsqueeze(2)
    X_parts.append(y_lag)

    # Add lagged differences Δy_{t-i} for i = 1 to lags.
    for i in range(1, lags + 1):
        lagged_diff = diff_y[:, (lags - i):(T - 1 - i)].unsqueeze(2)
        X_parts.append(lagged_diff)

    # Concatenate all regressors; let k denote the number of regressors.
    X = torch.cat(X_parts, dim=2)  # shape [B, M, k]
    k = X.shape[2]

    # OLS: Solve for beta in each batch: beta = (X'X)^{-1} X'Y.
    Xt = X.transpose(1, 2)  # shape [B, k, M]
    XtX = torch.bmm(Xt, X)  # shape [B, k, k]
    XtY = torch.bmm(Xt, Y.unsqueeze(2))  # shape [B, k, 1]
    beta = torch.linalg.solve(XtX, XtY).squeeze(2)  # shape [B, k]

    # Compute fitted values and residuals.
    Y_hat = (X * beta.unsqueeze(1)).sum(dim=2)
    residuals = Y - Y_hat  # shape [B, M]

    # Estimate variance of residuals: sigma^2 = (residuals' * residuals) / (M - k)
    dof = M - k
    sigma2 = (residuals ** 2).sum(dim=1) / dof  # shape [B]

    # Identify index for the coefficient on the lagged level term.
    if regression.lower() == "c":
        idx_gamma = 1  # beta = [constant, gamma, ...]
    else:  # regression 'ct'
        idx_gamma = 2  # beta = [constant, trend, gamma, ...]
        # Also, return the trend coefficient (second element).
        trend_coef = beta[:, 1]

    gamma_coeff = beta[:, idx_gamma]

    # Compute standard error for gamma coefficient.
    inv_XtX = torch.linalg.inv(XtX)
    se_gamma = torch.sqrt(sigma2 * inv_XtX[:, idx_gamma, idx_gamma])

    # ADF test statistic is the t-statistic for gamma.
    adf_stat = gamma_coeff / se_gamma

    return adf_stat, trend_coef

if __name__ == '__main__':
    # Example usage:
    import time

    B = 5   # number of series
    T = 200 # number of time points
    torch.manual_seed(42)

    # Simulate non-stationary series (random walks).
    random_walk = torch.zeros(B, T)
    random_walk[:, 0] = torch.randn(B)
    for t in range(1, T):
        random_walk[:, t] = random_walk[:, t-1] + torch.randn(B) * 0.2

    # Simulate stationary series (mean-reverting noise).
    stationary_series = torch.randn(B, T) * 0.5

    print("ADF statistic for random walk series (regression='ct', lags=5):")
    adf_stat_rw, trend_rw = adf_test(random_walk, regression="ct", lags=5)
    print("ADF Statistic:", adf_stat_rw)
    print("Trend Coefficient:", trend_rw)

    print("\nADF statistic for stationary series (regression='ct', lags=5):")
    adf_stat_stat, trend_stat = adf_test(stationary_series, regression="ct", lags=5)
    print("ADF Statistic:", adf_stat_stat)
    print("Trend Coefficient:", trend_stat)

