#!/usr/bin/env python3

import torch

def cusum_test(time_series: torch.Tensor, regression: str = "c"):
    """
    Compute the CUSUM test statistic for a batch of time series to assess parameter stability.
    The null hypothesis is that the underlying process is stable over time.
    
    This implementation mimics the classic CUSUM test (based on the Brown-Durbin-Evans approach)
    using the full-sample residuals rather than recursive residuals. It computes the cumulative sum 
    of standardized residuals, and the test statistic is the maximum absolute value of this cumulative 
    sum. Additionally, time-varying boundary functions (based on the asymptotic distribution of a 
    Brownian bridge) are provided for reference.
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] where B is the number of series
                                    and T is the number of time points.
        regression (str): 'c' for constant or 'ct' for constant plus trend.
        
    Returns:
        cusum_stat (torch.Tensor): Tensor of shape [B] with the maximum absolute value of the standardized
                                   cumulative sum (CUSUM) for each series.
        cusum_series (torch.Tensor): Tensor of shape [B, T] containing the standardized cumulative sum series.
        boundaries (tuple of torch.Tensor): A tuple (lower_bound, upper_bound) each of shape [T],
                                   representing the time-varying boundaries at the 5% significance level.
                                   Under the null, the CUSUM series should remain within these bounds.
    """
    B, T = time_series.shape
    device = time_series.device
    t = torch.arange(1, T + 1, device=device, dtype=time_series.dtype)  # time index 1,...,T
    
    # Compute residuals based on the type of regression.
    if regression.lower() == "c":
        # For constant-only: remove the mean.
        mean_ts = time_series.mean(dim=1, keepdim=True)
        residuals = time_series - mean_ts
    elif regression.lower() == "ct":
        # For constant-plus-trend: regress each series on [1, t].
        mean_y = time_series.mean(dim=1, keepdim=True)  # shape [B, 1]
        mean_t = t.mean()
        t_centered = t - mean_t  # shape [T]
        var_t = (t_centered ** 2).mean()  # scalar variance of time index
        y_centered = time_series - mean_y
        cov_ty = (y_centered * t_centered).mean(dim=1, keepdim=True)  # shape [B, 1]
        beta = cov_ty / var_t  # estimated slope, shape [B, 1]
        # Compute the intercept.
        alpha = mean_y - beta * mean_t  # shape [B, 1]
        # Compute fitted trend.
        trend = alpha + beta * t.unsqueeze(0)  # shape [B, T]
        residuals = time_series - trend
    else:
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")
    
    # Estimate the standard deviation of residuals for each series.
    sigma = torch.sqrt((residuals ** 2).mean(dim=1))  # shape [B]
    
    # Compute the cumulative sum of residuals for each series.
    S = torch.cumsum(residuals, dim=1)  # shape [B, T]
    
    # Standardize the cumulative sum.
    # We divide by sigma (per series) and by sqrt(T) (a common scaling factor).
    cusum_series = S / (sigma.unsqueeze(1) * torch.sqrt(torch.tensor(T, dtype=time_series.dtype, device=device)))
    
    # The CUSUM test statistic for each series is the maximum absolute value of the standardized cumulative sum.
    cusum_stat, _ = torch.max(torch.abs(cusum_series), dim=1)  # shape [B]
    
    # Compute time-varying boundaries.
    # Under the null, the standardized cumulative sum approximates a Brownian bridge.
    # A common 5% significance level boundary is given by:
    #   ± 1.96 * sqrt( t/T * (1 - t/T) )
    # for t=1,...,T.
    t_ratio = t / T  # shape [T]
    boundaries_upper = 1.96 * torch.sqrt(t_ratio * (1 - t_ratio))
    boundaries_lower = -boundaries_upper
    
    return cusum_stat, cusum_series, (boundaries_lower, boundaries_upper)


if __name__ == '__main__':
    # Example usage:
    import time

    B = 5    # number of series
    T = 500  # number of time points
    torch.manual_seed(42)

    # Simulate stable series: white noise around zero.
    stable_series = torch.randn(B, T)

    # Simulate series with a structural change:
    # We'll simulate a series that is stationary for the first half
    # and then has a shift in mean for the second half.
    structural_break_series = torch.randn(B, T)
    structural_break_series[:, T//2:] += 2.0  # introduce a break by shifting the mean

    print("CUSUM test for stable series (regression='ct'):")
    stat_stable, cusum_stable, (lb, ub) = cusum_test(stable_series, regression="ct")
    print("CUSUM Statistic:", stat_stable)

    print("\nCUSUM test for series with a structural break (regression='ct'):")
    stat_break, cusum_break, (lb, ub) = cusum_test(structural_break_series, regression="ct")
    print("CUSUM Statistic:", stat_break)

    # Optionally, one might visualize the standardized CUSUM series along with the boundaries
    # to inspect the stability over time.

