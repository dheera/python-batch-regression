#!/usr/bin/env python3

import torch

def cusum_squares_test(time_series: torch.Tensor, regression: str = "c"):
    """
    Compute the CUSUM of squares test statistic for a batch of time series to assess changes in variance.
    The null hypothesis is that the variance is constant over time.
    
    The procedure is as follows:
      1. Remove the mean (or detrend for a constant-plus-trend model) to obtain residuals.
      2. Compute the squared residuals.
      3. Compute the cumulative sum of these squared residuals, S_t, for t = 1,...,T.
      4. Normalize by the total sum of squares, S_T, so that under constant variance,
         S_t/S_T should be approximately equal to t/T.
      5. The test statistic is the maximum absolute deviation:
             sup_{t in [1, T]} | (S_t/S_T) - (t/T) |.
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] where B is the number of series
                                    and T is the number of time points.
        regression (str): 'c' for constant or 'ct' for constant plus trend.
        
    Returns:
        cusum_sq_stat (torch.Tensor): Tensor of shape [B] with the maximum absolute deviation for each series.
        cusum_sq_series (torch.Tensor): Tensor of shape [B, T] containing the deviation series:
                                         D_t = (S_t/S_T) - (t/T).
        boundaries (tuple of torch.Tensor): A tuple (lower_bound, upper_bound) each of shape [T],
                                   representing approximate time-varying boundaries.
                                   (These are provided as a rough guide; in practice, critical values
                                    are often obtained via simulation.)
    """
    B, T = time_series.shape
    device = time_series.device
    t = torch.arange(1, T + 1, device=device, dtype=time_series.dtype)  # time index: 1, 2, ..., T
    
    # Compute residuals based on the chosen regression model.
    if regression.lower() == "c":
        # For a constant-only model, remove the mean.
        mean_ts = time_series.mean(dim=1, keepdim=True)
        residuals = time_series - mean_ts
    elif regression.lower() == "ct":
        # For constant-plus-trend, regress each series on [1, t].
        mean_y = time_series.mean(dim=1, keepdim=True)  # [B, 1]
        mean_t = t.mean()
        t_centered = t - mean_t  # centered time index, [T]
        var_t = (t_centered ** 2).mean()  # scalar variance of t
        y_centered = time_series - mean_y
        cov_ty = (y_centered * t_centered).mean(dim=1, keepdim=True)  # [B, 1]
        beta = cov_ty / var_t  # slope, [B, 1]
        alpha = mean_y - beta * mean_t  # intercept, [B, 1]
        trend = alpha + beta * t.unsqueeze(0)  # fitted trend, [B, T]
        residuals = time_series - trend
    else:
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")
        
    # Compute squared residuals.
    squared_residuals = residuals ** 2  # [B, T]
    
    # Compute the cumulative sum of squared residuals for each series.
    S = torch.cumsum(squared_residuals, dim=1)  # [B, T]
    
    # Total sum of squares for each series.
    S_total = S[:, -1].unsqueeze(1)  # [B, 1]
    
    # Normalize the cumulative sum: S_t/S_total.
    normalized_S = S / S_total  # [B, T]
    
    # The expected cumulative proportion under constant variance is t/T.
    expected = t / T  # [T]
    
    # Deviation series: the difference between the normalized cumulative sum and expected proportion.
    deviation = normalized_S - expected.unsqueeze(0)  # [B, T]
    
    # The test statistic is the maximum absolute deviation.
    cusum_sq_stat, _ = torch.max(torch.abs(deviation), dim=1)  # [B]
    
    # Compute approximate time-varying boundaries.
    # A common approximation (under a Brownian bridge) is:
    #   ± 1.96 * sqrt( (t/T) * (1 - t/T) )
    # Note: For the CUSUM of squares test, critical values are often derived via simulation.
    boundaries_upper = 1.96 * torch.sqrt((t / T) * (1 - t / T))
    boundaries_lower = -boundaries_upper
    
    return cusum_sq_stat, deviation, (boundaries_lower, boundaries_upper)


if __name__ == '__main__':
    # Example usage:
    import time

    B = 5    # number of series
    T = 500  # number of time points
    torch.manual_seed(42)

    # Simulate series with stable variance: white noise.
    stable_series = torch.randn(B, T)

    # Simulate series with a variance change:
    # Here, the first half has lower variance than the second half.
    variance_change_series = torch.randn(B, T)
    variance_change_series[:, T//2:] *= 3.0  # increase variance in the second half

    print("CUSUM of squares test for stable variance series (regression='ct'):")
    stat_stable, cusum_sq_stable, (lb, ub) = cusum_squares_test(stable_series, regression="ct")
    print("CUSUM of Squares Statistic:", stat_stable)

    print("\nCUSUM of squares test for series with a variance change (regression='ct'):")
    stat_change, cusum_sq_change, (lb, ub) = cusum_squares_test(variance_change_series, regression="ct")
    print("CUSUM of Squares Statistic:", stat_change)

    # One may also visualize the deviation series along with the boundaries to assess variance stability.

