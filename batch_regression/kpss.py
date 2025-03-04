#!/usr/bin/env python3

import torch

def kpss_test(time_series: torch.Tensor, regression: str = "c", lags: int = 0):
    """
    Compute the KPSS test statistic for a batch of time series and,
    if using a constant-plus-trend model, also return the trend slope.

    The KPSS test examines the null hypothesis that a time series is stationary.
    For regression:
      - "c": tests stationarity around a constant (i.e., the series is level-stationary).
      - "ct": tests stationarity around a deterministic trend.

    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] where B is the number of series
                                    and T is the number of time points.
        regression (str): 'c' for constant or 'ct' for constant plus trend.
        lags (int): Number of lags to use in the long-run variance estimator.
                    lags=0 uses only the residual variance.

    Returns:
        kpss_stat (torch.Tensor): Tensor of shape [B] with the KPSS statistic for each series.
        trend_coef (torch.Tensor or None): If regression == 'ct', tensor of shape [B] with the estimated
                                             trend slopes; otherwise, None.
    """
    B, T = time_series.shape
    device = time_series.device
    t = torch.arange(1, T + 1, device=device, dtype=time_series.dtype)  # time index 1,...,T
    trend_slope = None
    trend_intercept = None

    # Compute residuals based on the type of regression.
    if regression.lower() == "c":
        # For constant-only, subtract the mean.
        mean_ts = time_series.mean(dim=1, keepdim=True)
        residuals = time_series - mean_ts
    elif regression.lower() == "ct":
        # For constant-plus-trend, regress each series on [1, t].
        # Compute the mean of time_series for each batch.
        mean_y = time_series.mean(dim=1, keepdim=True)  # shape [B, 1]
        # Compute mean of time index.
        mean_t = t.mean()
        # Center time variable.
        t_centered = t - mean_t  # shape [T]
        # Variance of centered time.
        var_t = (t_centered ** 2).mean()  # scalar
        # Center the series.
        y_centered = time_series - mean_y  # shape [B, T]
        # Compute covariance between t and y for each series.
        cov_ty = (y_centered * t_centered).mean(dim=1, keepdim=True)  # shape [B, 1]
        # Estimate the trend coefficient (slope) for each series.
        beta = cov_ty / var_t  # shape [B, 1]
        trend_slope = beta.squeeze(1)  # shape [B]
        # Compute the intercept.
        alpha = mean_y - beta * mean_t  # shape [B, 1]
        # Compute the fitted trend.
        trend = alpha + beta * t.unsqueeze(0)  # shape [B, T]
        # This is for convenience so that the user of this can compute the continuing trend
        # starting with beta * t + alpha of their continuing series
        # without having to know the length of t in this series
        trend_intercept = (alpha + beta * mean_t).squeeze(1)
        # Residuals: deviation from the fitted trend.
        residuals = time_series - trend
    else:
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")

    # Compute the cumulative sum of the residuals along time.
    # S_t = sum_{i=1}^{t} residual_i for each series.
    S = torch.cumsum(residuals, dim=1)  # shape [B, T]

    # Estimate the long-run variance.
    # Gamma(0): variance of residuals.
    gamma0 = (residuals ** 2).mean(dim=1)  # shape [B]
    sigma2 = gamma0.clone()

    # If lags > 0, include weighted autocovariances using the Bartlett kernel.
    if lags > 0:
        for h in range(1, lags + 1):
            # Compute gamma(h): for each series, average of residual[t]*residual[t-h].
            gamma_h = (residuals[:, h:] * residuals[:, :-h]).mean(dim=1)  # shape [B]
            weight = 1 - h / (lags + 1)
            sigma2 += 2 * weight * gamma_h

    # KPSS statistic for each series:
    # KPSS = sum_{t=1}^{T} S_t^2 / (T^2 * sigma2)
    sum_S2 = (S ** 2).sum(dim=1)  # shape [B]
    kpss_stat = sum_S2 / (T ** 2 * sigma2)

    return kpss_stat, trend_slope, trend_intercept

if __name__ == '__main__':
    # Example usage:
    import time

    B = 5     # number of series
    T = 500   # number of time points
    torch.manual_seed(42)

    # Simulate stationary series: white noise around zero.
    stationary_series = torch.randn(B, T)

    # Simulate non-stationary series: random walk.
    random_walk = torch.zeros(B, T)
    random_walk[:, 0] = torch.randn(B)
    for t in range(1, T):
        random_walk[:, t] = random_walk[:, t-1] + torch.randn(B) * 0.1

    print("KPSS statistic for stationary series (regression='ct', lags=5):")
    stat_stationary, trend_slope, trend_intercept = kpss_test(stationary_series, regression="ct", lags=5)
    print("KPSS Statistic:", stat_stationary)
    print("Trend Coefficient:", trend_slope)

    print("\nKPSS statistic for random walk series (regression='ct', lags=5):")
    stat_rw, trend_slope, trend_intercept = kpss_test(random_walk, regression="ct", lags=5)
    print("KPSS Statistic:", stat_rw)
    print("Trend Coefficient:", trend_slope)

