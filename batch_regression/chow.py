#!/usr/bin/env python3
import torch

def compute_rss_vec(y: torch.Tensor, regression: str):
    """
    Compute the residual sum of squares (RSS) for a batch of series y in a vectorized way.

    Args:
        y (torch.Tensor): Tensor of shape [B, n] where B is the batch size and n is the number of observations.
        regression (str): 'c' for constant-only or 'ct' for constant plus trend.

    Returns:
        rss (torch.Tensor): Tensor of shape [B] containing the RSS for each series.
        k (int): Number of parameters estimated (1 for 'c', 2 for 'ct').
    """
    B, n = y.shape
    if regression.lower() == "c":
        mean_y = y.mean(dim=1, keepdim=True)
        rss = ((y - mean_y)**2).sum(dim=1)
        k = 1
    elif regression.lower() == "ct":
        # Create design matrix X of shape [n, 2]: constant and time trend.
        t = torch.arange(1, n+1, device=y.device, dtype=y.dtype)
        X = torch.stack([torch.ones(n, device=y.device, dtype=y.dtype), t], dim=1)  # [n, 2]
        # Expand X to batch: [B, n, 2]
        X = X.unsqueeze(0).expand(B, n, 2)
        # Reshape y for regression: [B, n, 1]
        y_ = y.unsqueeze(-1)
        # Solve OLS in batch
        sol = torch.linalg.lstsq(X, y_, rcond=None)
        beta = sol.solution  # shape [B, 2, 1]
        # Compute fitted values: [B, n, 1] = X @ beta, then squeeze to [B, n]
        fitted = torch.bmm(X, beta).squeeze(-1)
        rss = ((y - fitted)**2).sum(dim=1)
        k = 2
    else:
        raise ValueError("regression must be either 'c' or 'ct'")
    return rss, k

def chow_test(time_series: torch.Tensor, break_idx: int, regression: str = "c"):
    """
    Perform the Chow test for a structural break at the specified break index
    in a vectorized manner (batch dimension is handled without an explicit Python loop).

    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] representing the dependent variable for each series.
        break_idx (int): Index at which the structural break is hypothesized.
                         The first sub-sample is time_series[:, :break_idx] and the second is time_series[:, break_idx:].
        regression (str): 'c' for constant-only or 'ct' for constant plus trend.

    Returns:
        F_stats (torch.Tensor): Tensor of shape [B] with the Chow test statistic for each series.
    """
    B, T = time_series.shape
    if break_idx <= 0 or break_idx >= T:
        raise ValueError("break_idx must be between 1 and T-1.")

    # Compute pooled RSS for the full series.
    rss_p, k = compute_rss_vec(time_series, regression)

    # Compute RSS for the first sub-sample (observations 0 to break_idx-1).
    y1 = time_series[:, :break_idx]
    rss_1, _ = compute_rss_vec(y1, regression)

    # Compute RSS for the second sub-sample (observations break_idx to T-1).
    y2 = time_series[:, break_idx:]
    rss_2, _ = compute_rss_vec(y2, regression)

    n1 = break_idx
    n2 = T - break_idx
    numerator = (rss_p - (rss_1 + rss_2)) / k
    denominator = (rss_1 + rss_2) / (n1 + n2 - 2 * k)
    F_stats = numerator / denominator
    return F_stats

if __name__ == '__main__':
    # Example usage:
    torch.manual_seed(42)

    # Create a batch of series with a structural break at index 250.
    B = 5
    T = 500
    break_idx = 250

    # Simulate series: first half from one distribution, second half with a shifted mean.
    # Here we use a constant-only model.
    seg1 = torch.randn(B, break_idx)
    seg2 = torch.randn(B, T - break_idx) + 2.0
    series_batch = torch.cat([seg1, seg2], dim=1)

    F_stats = chow_test(series_batch, break_idx, regression="c")
    for i, F in enumerate(F_stats):
        print(f"Series {i}: Chow test F-statistic = {F.item():.4f}")

