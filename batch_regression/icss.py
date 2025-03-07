#!/usr/bin/env python3
#!/usr/bin/env python3
import torch

def _icss_single(series: torch.Tensor, start_idx: int, end_idx: int, regression: str, threshold: float, min_size: int):
    """
    Recursively apply the ICSS algorithm on a segment of a time series.

    Args:
        series (torch.Tensor): 1D tensor for the time series.
        start_idx (int): Starting index of the segment.
        end_idx (int): Ending index (exclusive) of the segment.
        regression (str): 'c' for constant or 'ct' for constant plus trend.
        threshold (float): Critical threshold for the scaled test statistic.
        min_size (int): Minimum segment length to consider for break detection.

    Returns:
        List[int]: Sorted list of breakpoints (absolute indices) detected in the segment.
    """
    # If the segment is too short, do not test further.
    if (end_idx - start_idx) < min_size:
        return []

    segment = series[start_idx:end_idx]
    T_segment = segment.shape[0]

    # Compute residuals (detrend if needed)
    if regression.lower() == "c":
        seg_mean = segment.mean()
        residuals = segment - seg_mean
    elif regression.lower() == "ct":
        t = torch.arange(1, T_segment + 1, device=segment.device, dtype=segment.dtype)
        seg_mean = segment.mean()
        mean_t = t.mean()
        t_centered = t - mean_t
        var_t = (t_centered ** 2).mean()
        cov_ty = ((segment - seg_mean) * t_centered).mean()
        beta = cov_ty / var_t
        alpha = seg_mean - beta * mean_t
        trend = alpha + beta * t
        residuals = segment - trend
    else:
        raise ValueError("regression must be either 'c' (constant) or 'ct' (constant and trend)")

    # Work with squared residuals
    squared = residuals ** 2
    S = torch.cumsum(squared, dim=0)
    S_total = S[-1]

    # If the total is zero (no variability), return no breakpoints.
    if S_total == 0:
        return []

    normalized_S = S / S_total  # Should follow roughly t/T under constant variance.
    t_ratio = torch.arange(1, T_segment + 1, device=segment.device, dtype=segment.dtype) / T_segment
    deviation = normalized_S - t_ratio
    abs_deviation = torch.abs(deviation)

    # Scale the maximum deviation by sqrt(T_segment)
    scaled_deviation = torch.sqrt(torch.tensor(T_segment, dtype=segment.dtype, device=segment.device)) * abs_deviation
    max_dev, max_idx = torch.max(scaled_deviation, dim=0)

    breakpoints = []
    if max_dev > threshold:
        # A significant break is detected. Record the breakpoint (convert to absolute index).
        bp = start_idx + int(max_idx.item())
        breakpoints.append(bp)
        # Recursively check the left segment [start_idx, bp) and the right segment [bp+1, end_idx).
        left_breaks = _icss_single(series, start_idx, bp, regression, threshold, min_size)
        right_breaks = _icss_single(series, bp + 1, end_idx, regression, threshold, min_size)
        breakpoints.extend(left_breaks)
        breakpoints.extend(right_breaks)

    return sorted(breakpoints)

def icss_test(time_series: torch.Tensor, regression: str = "c", threshold: float = 1.358, min_size: int = 30):
    """
    Apply the ICSS algorithm to detect multiple variance breakpoints in a batch of time series.

    The algorithm works as follows:
      1. For each series, compute the squared residuals (using either a constant or constant-plus-trend model).
      2. Compute the cumulative sum of squared residuals and normalize by the total sum.
      3. Compute the scaled deviation: sqrt(segment_length) * |(S_t/S_total) - (t/T)|.
      4. If the maximum scaled deviation exceeds the threshold, record the index as a breakpoint
         and recursively apply the procedure on the subsegments.

    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] where B is the number of series and T is the number of time points.
        regression (str): 'c' for constant or 'ct' for constant plus trend.
        threshold (float): Threshold for the scaled test statistic (default 1.358 for ~5% significance level).
        min_size (int): Minimum segment length for detecting breaks.

    Returns:
        List[List[int]]: A list of length B, where each element is a sorted list of indices (int)
                         indicating the detected breakpoints in the corresponding series.
    """
    B, T = time_series.shape
    all_breakpoints = []
    for i in range(B):
        series = time_series[i]
        breaks = _icss_single(series, 0, T, regression, threshold, min_size)
        all_breakpoints.append(breaks)
    return all_breakpoints

if __name__ == '__main__':
    # Example usage:
    torch.manual_seed(42)
    B = 5    # Number of series
    T = 500  # Number of time points
    # Simulate a batch of series with a variance break:
    # For each series, the first half has a lower variance than the second half.
    series_batch = torch.randn(B, T)
    series_batch[:, T // 2:] *= 3.0  # Introduce a variance change

    breakpoints = icss_test(series_batch, regression="c", threshold=1.358, min_size=30)
    for i, bp in enumerate(breakpoints):
        print(f"Series {i}: Breakpoints detected at indices: {bp}")

