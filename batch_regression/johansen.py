#!/usr/bin/env python3

import torch

def johansen_test(Y: torch.Tensor, deterministic: str = "none"):
    """
    Perform a vectorized Johansen cointegration test on a batch of time series.

    Args:
        Y (torch.Tensor): Tensor of shape [B, T, n] where B is the number of batches,
                          T is the number of time points, and n is the number of series.
        deterministic (str): Type of deterministic component.
                             "none" means no deterministic terms,
                             "constant" means detrend by subtracting the time mean.

    Returns:
        eigenvalues (torch.Tensor): Tensor of shape [B, n] containing the sorted eigenvalues
                                    (in descending order) for each batch.
        trace_stats (torch.Tensor): Tensor of shape [B, n] where the i-th column is the trace
                                    statistic for testing the null of at most i cointegrating relations.
    """
    B, T, n = Y.shape
    device = Y.device
    dtype = Y.dtype

    # We lose one observation when differencing.
    T_prime = T - 1

    # Compute first differences: ΔY_t = Y_t - Y_{t-1} for t=2,...,T.
    dY = Y[:, 1:, :] - Y[:, :-1, :]  # Shape: [B, T-1, n]
    # Set R0 = ΔY and R1 = Y_{t-1} (levels, excluding the last observation)
    R0 = dY  # [B, T-1, n]
    R1 = Y[:, :T_prime, :]  # [B, T-1, n]

    # If a constant is included, detrend R0 and R1 by removing the time mean.
    if deterministic.lower() == "constant":
        R0 = R0 - R0.mean(dim=1, keepdim=True)
        R1 = R1 - R1.mean(dim=1, keepdim=True)
    elif deterministic.lower() != "none":
        raise ValueError("deterministic must be either 'none' or 'constant'")

    # Compute covariance matrices in batch.
    # S00 = (R0^T R0)/T_prime, S11 = (R1^T R1)/T_prime, S01 = (R0^T R1)/T_prime.
    S00 = torch.bmm(R0.transpose(1, 2), R0) / T_prime  # [B, n, n]
    S11 = torch.bmm(R1.transpose(1, 2), R1) / T_prime  # [B, n, n]
    S01 = torch.bmm(R0.transpose(1, 2), R1) / T_prime  # [B, n, n]

    # S10 = S01^T.
    S10 = S01.transpose(1, 2)

    # Compute M = inv(S11) @ S10 @ inv(S00) @ S01 for each batch.
    inv_S00 = torch.linalg.inv(S00)
    inv_S11 = torch.linalg.inv(S11)
    M = torch.bmm(torch.bmm(inv_S11, S10), torch.bmm(inv_S00, S01))  # [B, n, n]

    # Compute eigenvalues in batch.
    eigvals = torch.linalg.eig(M)[0].real  # [B, n]
    # Sort eigenvalues in descending order along the last dimension.
    eigenvalues, _ = torch.sort(eigvals, descending=True, dim=1)  # [B, n]

    # Compute trace statistics.
    # For each batch b and for each cointegration rank r = 0,...,n-1:
    #   Trace(r) = -T_prime * sum_{i=r}^{n-1} ln(1 - eigenvalue_i)
    log_term = torch.log(1 - eigenvalues)  # [B, n]
    # Reverse along the eigenvalue dimension, compute cumulative sum, then flip back.
    trace_stats = -T_prime * torch.flip(torch.cumsum(torch.flip(log_term, dims=[1]), dim=1), dims=[1])

    return eigenvalues, trace_stats

if __name__ == '__main__':
    # Example usage:
    B = 10   # Number of batches
    T = 200  # Number of time points
    n = 3    # Number of series per batch
    torch.manual_seed(42)

    # Simulate cointegrated series.
    # Generate a common stochastic trend Z_t (random walk) and add stationary noise.
    Z = torch.zeros(B, T)
    Z[:, 0] = torch.randn(B)
    for t in range(1, T):
        Z[:, t] = Z[:, t-1] + torch.randn(B) * 0.2

    # Generate cointegrated series: each series is Z_t plus different stationary noise.
    Y = torch.zeros(B, T, n)
    for i in range(n):
        noise = torch.randn(B, T) * 0.5
        Y[:, :, i] = Z + noise

    eigenvals, trace_stats = johansen_test(Y, deterministic="none")
    print("Eigenvalues (each row sorted descending):")
    print(eigenvals)
    print("\nTrace Statistics (each row, columns corresponding to testing ≤ r cointegrating vectors):")
    print(trace_stats)

