#!/usr/bin/env python3
import torch

def compute_rss_const(y: torch.Tensor):
    """
    Vectorized computation of the residual sum of squares (RSS) for a constant-only model.
    For each series y (of shape [B, n]), returns the RSS using the formula:

        RSS = sum(y^2) - (sum(y)^2) / n

    Args:
        y (torch.Tensor): Tensor of shape [B, n].

    Returns:
        rss (torch.Tensor): Tensor of shape [B] containing the RSS for each series.
    """
    n = y.shape[1]
    sum_y = y.sum(dim=1)
    sum_y_sq = (y**2).sum(dim=1)
    rss = sum_y_sq - (sum_y**2) / n
    return rss

def qlr_test(time_series: torch.Tensor, trim: float = 0.15):
    """
    Vectorized implementation of the QLR (sup-Wald) test for an unknown structural break
    for a batch of time series under the constant-only model.

    For each candidate break date, we compute:
        rss1 = RSS for the first segment (observations 0...candidate-1)
        rss2 = RSS for the second segment (observations candidate...T-1)
    and the pooled RSS (rss_p) is computed over the entire series.

    The Chow test statistic for a candidate break is given by:

        F_candidate = ((rss_p - (rss1+rss2)) / k) / ((rss1+rss2) / (T - 2*k))

    where for the constant model, k=1 and T is the full series length.
    The QLR statistic is the maximum F_candidate over candidate break dates.

    Args:
        time_series (torch.Tensor): Tensor of shape [B, T] representing B time series.
        trim (float): Proportion of observations to trim from both ends when searching for the break date.

    Returns:
        qlr_stats (torch.Tensor): Tensor of shape [B] with the QLR test statistic for each series.
        break_indices (torch.Tensor): Tensor of shape [B] with the candidate break index (int) that maximizes the F statistic.
    """
    B, T = time_series.shape
    k = 1  # number of parameters for constant-only model

    # Define candidate break index range.
    start_candidate = int(trim * T)
    end_candidate = int((1 - trim) * T)
    candidates = torch.arange(start_candidate, end_candidate, device=time_series.device, dtype=torch.float32)
    n_candidates = candidates.numel()  # number of candidate break points

    # Precompute cumulative sums and cumulative sum of squares for each series.
    S = torch.cumsum(time_series, dim=1)        # shape [B, T]
    S_sq = torch.cumsum(time_series**2, dim=1)    # shape [B, T]

    # Total sums (last column) for each series.
    S_total = S[:, -1].unsqueeze(1)       # [B, 1]
    S_sq_total = S_sq[:, -1].unsqueeze(1) # [B, 1]

    # Compute pooled RSS for each series.
    rss_p = S_sq_total.squeeze(1) - (S_total.squeeze(1)**2) / T  # shape [B]

    # For each candidate break, we need the cumulative sums at index candidate-1.
    idx = (candidates - 1).long()           # shape [n_candidates]
    idx_exp = idx.unsqueeze(0).expand(B, -1)  # shape [B, n_candidates]

    # Gather S and S_sq at these indices.
    S_candidate = torch.gather(S, dim=1, index=idx_exp)       # shape [B, n_candidates]
    S_sq_candidate = torch.gather(S_sq, dim=1, index=idx_exp)   # shape [B, n_candidates]

    # Expand candidates to shape [B, n_candidates].
    candidates_exp = candidates.unsqueeze(0).expand(B, -1)      # shape [B, n_candidates]
    # Number of observations in the first segment is candidates_exp and in the second segment T - candidate.
    n2 = T - candidates_exp                                   # shape [B, n_candidates]

    # Compute RSS for first segment.
    rss1 = S_sq_candidate - (S_candidate**2) / candidates_exp   # shape [B, n_candidates]

    # Compute sums for second segment.
    S2 = S_total - S_candidate                                  # shape [B, n_candidates]
    S_sq2 = S_sq_total - S_sq_candidate                         # shape [B, n_candidates]
    rss2 = S_sq2 - (S2**2) / n2                                  # shape [B, n_candidates]

    # Compute Chow test F statistic for each candidate.
    df = T - 2 * k  # degrees of freedom in denominator.
    rss_p_exp = rss_p.unsqueeze(1).expand(-1, n_candidates)      # shape [B, n_candidates]
    numerator = (rss_p_exp - (rss1 + rss2)) / k                  # shape [B, n_candidates]
    denominator = (rss1 + rss2) / df                             # shape [B, n_candidates]
    F_candidates = numerator / denominator                     # shape [B, n_candidates]

    # For each series, select the candidate with the maximum F statistic.
    max_F, argmax = F_candidates.max(dim=1)                      # shapes: [B], [B]
    best_breaks = candidates_exp.gather(dim=1, index=argmax.unsqueeze(1)).squeeze(1).long()  # shape [B]

    return max_F, best_breaks

if __name__ == '__main__':
    torch.manual_seed(42)
    B = 5    # number of series
    T = 500  # number of time points

    # For each series, choose a random true break index within the candidate region.
    true_breaks = []
    series_list = []
    for i in range(B):
        # Choose a random true break index between 20% and 80% of T.
        true_break = torch.randint(int(0.2 * T), int(0.8 * T), (1,)).item()
        true_breaks.append(true_break)
        # Simulate series: first segment with mean 0; second segment with mean shift +2.
        seg1 = torch.randn(1, true_break)
        seg2 = torch.randn(1, T - true_break) + 2.0
        series = torch.cat([seg1, seg2], dim=1)
        series_list.append(series)

    series_batch = torch.cat(series_list, dim=0)  # shape [B, T]

    # Run the vectorized QLR test for the constant-only model.
    qlr_stats, break_indices = qlr_test(series_batch, trim=0.15)
    for i in range(B):
        print(f"Series {i}: True break index = {true_breaks[i]}, Estimated break index = {break_indices[i].item()}, QLR test statistic = {qlr_stats[i].item():.4f}")

