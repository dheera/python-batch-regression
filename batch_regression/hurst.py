#!/usr/bin/env python3
#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt

def estimate_hurst_exponent(time_series: torch.Tensor, min_window: int = 8, max_window: int = None, num_scales: int = 20) -> torch.Tensor:
    """
    Estimate the Hurst exponent for a batch of time series using Rescaled Range Analysis (R/S analysis).

    Args:
        time_series (torch.Tensor): Tensor of shape [B, T], where B is the number of series and T is the number of time points.
        min_window (int): Minimum window size to use for analysis.
        max_window (int): Maximum window size. If None, defaults to T//2.
        num_scales (int): Number of window sizes (scales) to evaluate.

    Returns:
        torch.Tensor: Tensor of shape [B] containing the estimated Hurst exponent for each series.
    """

    B, T = time_series.shape
    if max_window is None:
        max_window = T // 2

    # Generate logarithmically spaced window sizes (scales)
    scales = torch.logspace(torch.log10(torch.tensor(min_window, device=time_series.device, dtype=torch.float32)),
                            torch.log10(torch.tensor(max_window, device=time_series.device, dtype=torch.float32)),
                            steps=num_scales).round().long().to(time_series.device)
    scales = torch.unique(scales)  # ensure unique scales
    num_scales = scales.numel()

    # Preallocate tensor to store average R/S for each series and scale
    rs_values = torch.zeros(B, num_scales, device=time_series.device, dtype=torch.float32)

    # Loop over scales
    for i, s in enumerate(scales):
        s = s.item()  # window size as integer
        if s < 2:
            continue
        # Number of non-overlapping segments of length s
        num_segments = T // s
        if num_segments < 1:
            continue

        # Truncate the series to have an integer number of segments and reshape
        truncated = time_series[:, :num_segments * s]  # [B, num_segments*s]
        segments = truncated.view(B, num_segments, s)    # [B, num_segments, s]

        # For each segment, compute deviations from the segment mean.
        seg_mean = segments.mean(dim=2, keepdim=True)
        deviations = segments - seg_mean  # [B, num_segments, s]

        # Compute cumulative deviation for each segment.
        cum_dev = torch.cumsum(deviations, dim=2)  # [B, num_segments, s]

        # Range: maximum minus minimum of the cumulative deviations.
        R = cum_dev.max(dim=2)[0] - cum_dev.min(dim=2)[0]  # [B, num_segments]

        # Standard deviation of the original segment.
        S = segments.std(dim=2, unbiased=True)  # [B, num_segments]
        # Avoid division by zero.
        S = torch.where(S == 0, torch.tensor(1e-8, device=S.device, dtype=S.dtype), S)

        RS = R / S  # Rescaled range, [B, num_segments]

        # Average R/S for the current scale s.
        avg_RS = RS.mean(dim=1)  # [B]
        rs_values[:, i] = avg_RS

    # Perform a log-log regression: log(R/S) = H * log(scale) + c
    log_scales = torch.log(scales.to(torch.float32))  # [num_scales]
    log_rs = torch.log(rs_values)  # [B, num_scales]

    # Compute mean and variance of log_scales (same for all series)
    mean_log_scales = log_scales.mean()
    var_log_scales = ((log_scales - mean_log_scales) ** 2).mean()

    # For each series, compute the covariance between log_scales and log_rs
    cov = ((log_rs - log_rs.mean(dim=1, keepdim=True)) * (log_scales - mean_log_scales)).mean(dim=1)
    # The slope of the regression is the Hurst exponent
    H = cov / var_log_scales  # [B]

    return H

if __name__ == '__main__':
    torch.manual_seed(42)
    B = 3    # number of series per process type
    T = 1024 # length of each series

    # -------------------------------
    # Brownian Motion (Random Walk)
    # -------------------------------
    brownian_motion = torch.zeros(B, T)
    brownian_motion[:, 0] = torch.randn(B)
    for t in range(1, T):
        brownian_motion[:, t] = brownian_motion[:, t-1] + torch.randn(B)

    # -------------------------------------
    # Mean-Reverting Process (Ornstein–Uhlenbeck)
    # -------------------------------------
    theta = 0.1   # speed of reversion
    mu = 0.0      # long-term mean
    sigma = 1.0   # volatility
    mean_reverting = torch.zeros(B, T)
    mean_reverting[:, 0] = torch.randn(B)
    for t in range(1, T):
        dt = 1.0
        noise = sigma * torch.randn(B)
        mean_reverting[:, t] = mean_reverting[:, t-1] + theta * (mu - mean_reverting[:, t-1]) * dt + noise

    # ---------------------------
    # Trending Process
    # ---------------------------
    slope = 0.05  # constant drift per time step
    trending = torch.zeros(B, T)
    for t in range(T):
        trending[:, t] = slope * t + torch.randn(B) * 1.0  # linear trend plus noise

    # Estimate Hurst exponents
    H_brownian = estimate_hurst_exponent(brownian_motion)
    H_mean_reverting = estimate_hurst_exponent(mean_reverting)
    H_trending = estimate_hurst_exponent(trending)

    # Print the estimated Hurst exponents for each series
    print("Estimated Hurst exponents for Brownian Motion:")
    for i in range(B):
        print(f" Series {i}: {H_brownian[i].item():.3f}")

    print("\nEstimated Hurst exponents for Mean-Reverting Process:")
    for i in range(B):
        print(f" Series {i}: {H_mean_reverting[i].item():.3f}")

    print("\nEstimated Hurst exponents for Trending Process:")
    for i in range(B):
        print(f" Series {i}: {H_trending[i].item():.3f}")

    # Optionally, plot one series from each type to visually compare their behavior.
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(brownian_motion[0].numpy())
    plt.title("Brownian Motion (Random Walk)")

    plt.subplot(3, 1, 2)
    plt.plot(mean_reverting[0].numpy())
    plt.title("Mean-Reverting Process (Ornstein–Uhlenbeck)")

    plt.subplot(3, 1, 3)
    plt.plot(trending[0].numpy())
    plt.title("Trending Process")

    plt.tight_layout()
    plt.show()

