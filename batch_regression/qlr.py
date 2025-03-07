#!/usr/bin/env python3
import torch

def compute_rss(y: torch.Tensor, regression: str):
    """
    Compute the residual sum of squares (RSS) for a single time series y given the regression model.
    
    Args:
        y (torch.Tensor): 1D tensor of observations.
        regression (str): 'c' for constant or 'ct' for constant-plus-trend.
    
    Returns:
        rss (torch.Tensor): Residual sum of squares (scalar tensor).
        k (int): Number of estimated parameters (1 for 'c', 2 for 'ct').
    """
    n = y.shape[0]
    if regression.lower() == "c":
        y_bar = y.mean()
        rss = ((y - y_bar) ** 2).sum()
        k = 1
    elif regression.lower() == "ct":
        t = torch.arange(1, n + 1, device=y.device, dtype=y.dtype)
        # Design matrix with constant and trend.
        X = torch.stack([torch.ones(n, device=y.device, dtype=y.dtype), t], dim=1)
        beta = torch.linalg.lstsq(X, y.unsqueeze(1), rcond=None).solution.squeeze()
        fitted = X @ beta
        rss = ((y - fitted) ** 2).sum()
        k = 2
    else:
        raise ValueError("regression must be either 'c' or 'ct'")
    return rss, k

def qlr_test(time_series: torch.Tensor, regression: str = "c", trim: float = 0.15):
    """
    Perform the QLR (sup-Wald) test for an unknown structural break in a batch of time series.
    
    For each series, the procedure is:
      1. Compute the pooled RSS for the full series.
      2. For candidate break dates in the range [trim*T, (1-trim)*T], compute the Chow test F statistic:
             F = ((RSS_pooled - (RSS_1 + RSS_2)) / k) / ((RSS_1 + RSS_2) / (n1+n2-2*k))
      3. The QLR statistic is the maximum F value over all candidate break dates.
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] representing the series.
        regression (str): 'c' for constant or 'ct' for constant-plus-trend model.
        trim (float): Proportion of observations to trim from both ends when searching for the break date (default: 0.15).
    
    Returns:
        qlr_stats (torch.Tensor): Tensor of shape [B] with the QLR test statistic for each series.
        break_indices (list): A list of length B containing the candidate break index that maximizes the F statistic.
    """
    B, T = time_series.shape
    qlr_stats = torch.empty(B, dtype=time_series.dtype, device=time_series.device)
    break_indices = []
    
    # Define candidate break index range based on trimming.
    start_candidate = int(trim * T)
    end_candidate = int((1 - trim) * T)
    
    for i in range(B):
        y = time_series[i]
        rss_p, k = compute_rss(y, regression)
        max_F = -float("inf")
        best_break = None
        
        # Evaluate Chow test over candidate break dates.
        for candidate in range(start_candidate, end_candidate):
            n1 = candidate
            n2 = T - candidate
            if n1 < k or n2 < k:
                continue
            rss1, _ = compute_rss(y[:candidate], regression)
            rss2, _ = compute_rss(y[candidate:], regression)
            numerator = (rss_p - (rss1 + rss2)) / k
            denominator = (rss1 + rss2) / (n1 + n2 - 2 * k)
            F_candidate = numerator / denominator
            if F_candidate > max_F:
                max_F = F_candidate
                best_break = candidate
        
        qlr_stats[i] = max_F
        break_indices.append(best_break)
    
    return qlr_stats, break_indices

if __name__ == '__main__':
    # Example usage:
    torch.manual_seed(42)
    B = 5    # number of series
    T = 500  # number of time points
    true_break = 250  # true break point for simulation
    
    # Simulate a batch of series with a structural break in the mean.
    series_batch = torch.empty(B, T)
    for i in range(B):
        seg1 = torch.randn(true_break)           # first segment with mean 0
        seg2 = torch.randn(T - true_break) + 2.0   # second segment with shifted mean
        series_batch[i] = torch.cat([seg1, seg2])
    
    qlr_stats, break_indices = qlr_test(series_batch, regression="c", trim=0.15)
    for i in range(B):
        print(f"Series {i}: QLR test statistic = {qlr_stats[i].item():.4f}, estimated break index = {break_indices[i]}")

