#!/usr/bin/env python3
import torch

def compute_segment_rss(y: torch.Tensor, start: int, end: int, regression: str):
    """
    Compute the residual sum of squares (RSS) for a segment of the series y from index start (inclusive)
    to end (exclusive) using either a constant ("c") or constant-plus-trend ("ct") regression.
    """
    segment = y[start:end]
    n_seg = segment.shape[0]
    if n_seg <= 0:
        return 0.0
    if regression.lower() == "c":
        mean_seg = segment.mean()
        rss = ((segment - mean_seg) ** 2).sum()
        return rss.item()
    elif regression.lower() == "ct":
        # Use a simple linear regression: y = alpha + beta * t.
        t = torch.arange(1, n_seg + 1, dtype=segment.dtype, device=segment.device)
        X = torch.stack([torch.ones(n_seg, dtype=segment.dtype, device=segment.device), t], dim=1)
        beta = torch.linalg.lstsq(X, segment.unsqueeze(1), rcond=None).solution.squeeze()
        fitted = X @ beta
        rss = ((segment - fitted) ** 2).sum()
        return rss.item()
    else:
        raise ValueError("regression must be either 'c' or 'ct'")

def dynamic_programming_segmentation(y: torch.Tensor, regression: str, min_size: int, max_breaks: int):
    """
    Uses dynamic programming to segment a 1D series into 1 to max_breaks+1 segments.
    
    Precompute a cost matrix for all valid segments (each segment must have at least min_size observations).
    Then, for m segments (m = 1 corresponds to no break), find the segmentation that minimizes total RSS.
    Finally, select the segmentation that minimizes a Bayesian Information Criterion (BIC) criterion.
    
    Returns:
        segments: Sorted list of break indices (each index is the starting index of a segment, excluding 0)
        best_m: The optimal number of segments (i.e. best segmentation uses best_m segments)
        best_cost: The total RSS for the optimal segmentation.
        best_bic: The BIC value for the optimal segmentation.
    """
    n = y.shape[0]
    # Precompute cost[i][j]: cost for segment y[i:j], for all i and j with j - i >= min_size.
    cost = [[float('inf')] * (n + 1) for _ in range(n)]
    for i in range(n):
        for j in range(i + min_size, n + 1):
            cost[i][j] = compute_segment_rss(y, i, j, regression)
    
    # Let dp[m][j] be the minimum cost to segment the first j observations into m segments.
    # We allow m = 1, 2, ..., max_breaks+1 (where m=1 corresponds to no break).
    dp = [[float('inf')] * (n + 1) for _ in range(max_breaks + 2)]
    # backpointer: bp[m][j] stores the index at which the last segment starts for the optimal segmentation.
    bp = [[None] * (n + 1) for _ in range(max_breaks + 2)]
    
    # Base case: with 1 segment, the cost is just cost[0][j].
    for j in range(min_size, n + 1):
        dp[1][j] = cost[0][j]
        bp[1][j] = 0
    
    # For m segments (m >= 2), fill in dp[m][j] for j from m*min_size to n.
    for m in range(2, max_breaks + 2):
        for j in range(m * min_size, n + 1):
            best = float('inf')
            best_s = None
            # The last segment must have at least min_size observations,
            # so the candidate split point s runs from (m-1)*min_size to j - min_size.
            for s in range((m - 1) * min_size, j - min_size + 1):
                candidate = dp[m - 1][s] + cost[s][j]
                if candidate < best:
                    best = candidate
                    best_s = s
            dp[m][j] = best
            bp[m][j] = best_s
    
    # Now, for each possible number of segments m (1 to max_breaks+1), compute a BIC criterion.
    # Let k be the number of regression parameters per segment: k=1 for "c", k=2 for "ct".
    k = 1 if regression.lower() == "c" else 2
    best_bic = float('inf')
    best_m = None
    best_cost = None
    n_val = n  # total number of observations
    for m in range(1, max_breaks + 2):
        total_rss = dp[m][n]
        num_params = m * k
        # A simple BIC: ln(total_rss/n) + (num_params * ln(n)) / n.
        bic = torch.log(torch.tensor(total_rss / n_val, dtype=torch.float64)) + (num_params * torch.log(torch.tensor(n_val, dtype=torch.float64))) / n_val
        bic_val = bic.item()
        if bic_val < best_bic:
            best_bic = bic_val
            best_m = m
            best_cost = total_rss
    
    # Reconstruct the segmentation for the chosen number of segments (best_m).
    segments = []
    m = best_m
    j = n
    while m > 1:
        s = bp[m][j]
        segments.append(s)
        j = s
        m -= 1
    segments = sorted(segments)
    # 'segments' now holds the indices at which breaks occur (excluding index 0).
    return segments, best_m, best_cost, best_bic

def bai_perron_test(time_series: torch.Tensor, regression: str = "c", min_size: int = 30, max_breaks: int = 3):
    """
    Apply a simplified Bai–Perron procedure to detect multiple structural breaks in a batch of time series.
    
    For each series (each row of time_series), we use dynamic programming to search for the optimal segmentation
    (i.e. locations of breaks) given a minimum segment size and an upper limit on the number of breaks.
    The optimal segmentation is chosen as the one that minimizes a BIC-type criterion.
    
    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] for B time series of length T.
        regression (str): 'c' for constant or 'ct' for constant-plus-trend.
        min_size (int): Minimum number of observations per segment.
        max_breaks (int): Maximum number of breaks to consider.
        
    Returns:
        results (list): A list of length B, where each element is a dictionary with keys:
                        'breakpoints': list of break indices (int) where breaks are detected.
                        'num_segments': optimal number of segments (int).
                        'total_rss': total RSS for the optimal segmentation.
                        'bic': BIC value for the optimal segmentation.
    """
    B, T = time_series.shape
    results = []
    for i in range(B):
        y = time_series[i]
        segments, num_segments, total_rss, bic = dynamic_programming_segmentation(y, regression, min_size, max_breaks)
        results.append({
            'breakpoints': segments,
            'num_segments': num_segments,
            'total_rss': total_rss,
            'bic': bic
        })
    return results

if __name__ == '__main__':
    torch.manual_seed(42)
    B = 3
    T = 300
    # Simulate a batch of series with multiple breaks in the mean.
    series_batch = torch.empty(B, T)
    for i in range(B):
        # Each series is composed of three segments with different means.
        seg1 = torch.randn(100)        # mean ~0
        seg2 = torch.randn(100) + 2.0    # mean shifted upward
        seg3 = torch.randn(100) - 1.0    # mean shifted downward
        series_batch[i] = torch.cat([seg1, seg2, seg3])
    
    results = bai_perron_test(series_batch, regression="c", min_size=30, max_breaks=3)
    for i, res in enumerate(results):
        print(f"Series {i}:")
        print("  Breakpoints:", res['breakpoints'])
        print("  Number of segments:", res['num_segments'])
        print("  Total RSS:", res['total_rss'])
        print("  BIC:", res['bic'])

