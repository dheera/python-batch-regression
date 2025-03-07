#!/usr/bin/env python3
import torch

def compute_rss(y: torch.Tensor, regression: str):
    """
    Compute the residual sum of squares (RSS) for a single series y given the regression model.
    
    Args:
        y (torch.Tensor): 1D tensor of observations.
        regression (str): 'c' for constant or 'ct' for constant plus trend.
    
    Returns:
        rss (torch.Tensor): Residual sum of squares (scalar tensor).
        k (int): Number of estimated parameters (1 for 'c', 2 for 'ct').
    """
    n = y.shape[0]
    if regression.lower() == "c":
        # Constant-only regression: fitted value is the mean.
        y_bar = y.mean()
        rss = ((y - y_bar) ** 2).sum()
        k = 1
    elif regression.lower() == "ct":
        # Constant-plus-trend regression.
        t = torch.arange(1, n + 1, device=y.device, dtype=y.dtype)  # time index 1,...,n
        # Build design matrix X with a constant and a trend.
        X = torch.stack([torch.ones(n, device=y.device, dtype=y.dtype), t], dim=1)  # shape [n, 2]
        # Solve for beta in the least squares sense.
        # We reshape y to [n, 1] to use lstsq.
        beta = torch.linalg.lstsq(X, y.unsqueeze(1), rcond=None).solution.squeeze()
        fitted = X @ beta
        rss = ((y - fitted) ** 2).sum()
        k = 2
    else:
        raise ValueError("regression must be either 'c' or 'ct'")
    return rss, k

def chow_test(time_series: torch.Tensor, break_idx: int, regression: str = "c"):
    """
    Perform the Chow test for a structural break at the specified break index
    for a batch of time series.
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] representing the dependent variable for each series.
        break_idx (int): Index at which the structural break is hypothesized.
                         The first sub-sample is time_series[:, :break_idx] and the second is time_series[:, break_idx:].
        regression (str): 'c' for constant-only or 'ct' for constant plus trend.
    
    Returns:
        F_stat (torch.Tensor): Tensor of shape [B] with the Chow test statistic for each series.
    
    Note:
        The test statistic follows an F-distribution with degrees of freedom k and (n1+n2-2k),
        where k is the number of parameters (1 or 2) and n1 and n2 are the lengths of the two sub-samples.
    """
    B, T = time_series.shape
    F_stats = torch.empty(B, dtype=time_series.dtype, device=time_series.device)
    
    # Loop over each series in the batch.
    for i in range(B):
        y = time_series[i]
        n = T
        
        # Split into two subsamples.
        # First segment: observations 0,1,...,break_idx-1
        # Second segment: observations break_idx, break_idx+1,...,T-1
        if break_idx <= 0 or break_idx >= n:
            raise ValueError("break_idx must be between 1 and T-1.")
        y1 = y[:break_idx]
        y2 = y[break_idx:]
        n1, n2 = y1.shape[0], y2.shape[0]
        
        # Compute pooled RSS.
        rss_p, k = compute_rss(y, regression)
        # Compute RSS for each sub-sample.
        rss_1, _ = compute_rss(y1, regression)
        rss_2, _ = compute_rss(y2, regression)
        
        # Compute Chow test F statistic.
        numerator = (rss_p - (rss_1 + rss_2)) / k
        denominator = (rss_1 + rss_2) / (n1 + n2 - 2 * k)
        F_stat = numerator / denominator
        F_stats[i] = F_stat
    
    return F_stats

if __name__ == '__main__':
    # Example usage:
    torch.manual_seed(42)
    
    # Create a batch of series with a structural break.
    # For example, series 0...B-1 with a break at index 250.
    B = 5
    T = 500
    break_idx = 250
    
    # Simulate a batch where the first half is generated with one mean (and/or trend)
    # and the second half has a shifted mean.
    series_batch = torch.empty(B, T)
    for i in range(B):
        # First segment: white noise with mean 0.
        seg1 = torch.randn(break_idx)  
        # Second segment: white noise with a shift in mean (e.g., +2.0).
        seg2 = torch.randn(T - break_idx) + 2.0  
        series_batch[i] = torch.cat([seg1, seg2])
    
    # Run the Chow test using a constant-only model.
    F_stats = chow_test(series_batch, break_idx, regression="c")
    for i, F in enumerate(F_stats):
        print(f"Series {i}: Chow test F-statistic = {F.item():.4f}")

