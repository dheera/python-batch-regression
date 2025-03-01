#!/usr/bin/env python3
import torch

def kpss_test(time_series: torch.Tensor, regression: str = "c", lags: int = 0) -> torch.Tensor:
    """
    Compute the KPSS test statistic for a batch of time series.
    
    KPSS (Kwiatkowski–Phillips–Schmidt–Shin) tests the null hypothesis that an observed time series
    is stationary around a deterministic trend (constant or constant plus trend).
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] where B is the number of series
                                    and T is the number of time points.
        regression (str): 'c' for testing stationarity around a constant, or 'ct' for a constant plus trend.
        lags (int): Number of lags to use in the long-run variance estimator.
                    Setting lags=0 uses only the variance of the residuals.
    
    Returns:
        torch.Tensor: KPSS test statistic for each series in the batch, shape [B].
    """
    B, T = time_series.shape
    device = time_series.device
    t = torch.arange(1, T + 1, device=device, dtype=time_series.dtype)  # time index: 1,...,T

    # Compute the residuals from the regression of the series on the deterministic component.
    if regression.lower() == "c":
        # Subtract the mean
        mean_ts = time_series.mean(dim=1, keepdim=True)
        eps = time_series - mean_ts
    elif regression.lower() == "ct":
        # Regress y on [1, t]. For each series compute:
        #   beta = Cov(t, y) / Var(t)
        #   alpha = mean(y) - beta * mean(t)
        mean_t = t.mean()
        t_centered = t - mean_t  # shape [T]
        var_t = (t_centered ** 2).mean()  # scalar
        mean_y = time_series.mean(dim=1, keepdim=True)  # shape [B, 1]
        # Compute covariance between t and y for each series.
        # First center y:
        y_centered = time_series - mean_y  # shape [B, T]
        # Covariance: dot product divided by T.
        cov_ty = (y_centered * t_centered).mean(dim=1, keepdim=True)  # shape [B, 1]
        beta = cov_ty / var_t  # shape [B, 1]
        alpha = mean_y - beta * mean_t  # shape [B, 1]
        # Fitted trend: alpha + beta * t
        trend = alpha + beta * t.unsqueeze(0)  # shape [B, T]
        eps = time_series - trend
    else:
        raise ValueError("regression must be 'c' or 'ct'")

    # Compute the cumulative sum of residuals along the time axis.
    # S_t = sum_{i=1}^{t} eps_i for each series.
    S = torch.cumsum(eps, dim=1)  # shape [B, T]

    # Estimate the long-run variance.
    # Compute gamma(0) (the variance of residuals).
    gamma0 = (eps ** 2).mean(dim=1)  # shape [B]
    sigma2 = gamma0.clone()
    
    # If lags > 0, compute weighted autocovariances using a Bartlett kernel.
    if lags > 0:
        for h in range(1, lags + 1):
            # Compute gamma(h): for each series, average over products eps[t]*eps[t-h]
            gamma_h = (eps[:, h:] * eps[:, :-h]).mean(dim=1)  # shape [B]
            weight = 1 - h / (lags + 1)
            sigma2 += 2 * weight * gamma_h

    # Compute KPSS statistic for each series:
    # KPSS = sum_{t=1}^{T} S_t^2 / (T^2 * sigma2)
    sum_S2 = (S ** 2).sum(dim=1)  # shape [B]
    kpss_stat = sum_S2 / (T ** 2 * sigma2)

    return kpss_stat


if __name__ == '__main__':
    # Example usage.
    import time

    # Create a batch of time series data.
    B = 5     # number of series
    T = 500   # number of time points
    torch.manual_seed(42)
    
    # Simulate stationary series (mean-reverting noise around zero).
    stationary_series = torch.randn(B, T)
    
    # Simulate non-stationary series (random walk).
    random_walk = torch.zeros(B, T)
    random_walk[:, 0] = torch.randn(B)
    for t in range(1, T):
        random_walk[:, t] = random_walk[:, t-1] + torch.randn(B) * 0.1

    print("KPSS statistic for stationary series (regression='c', lags=5):")
    stat_stationary = kpss_test(stationary_series, regression="c", lags=5)
    print(stat_stationary)

    print("\nKPSS statistic for random walk series (regression='c', lags=5):")
    stat_rw = kpss_test(random_walk, regression="c", lags=5)
    print(stat_rw)

