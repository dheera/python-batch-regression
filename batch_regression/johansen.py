#!/usr/bin/env python3
import torch

def johansen_test(Y: torch.Tensor, deterministic: str = "none"):
    """
    Perform a simplified Johansen cointegration test on a batch of time series.
    
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
                                    statistic for testing at most i-1 cointegrating relations.
                                    (For i=1, the statistic tests the null of no cointegration.)
    """
    B, T, n = Y.shape
    device = Y.device
    dtype = Y.dtype
    
    # Number of usable observations (we lose one due to differencing)
    T_prime = T - 1
    
    # Compute first differences: ΔY_t = Y_t - Y_{t-1} for t=2,...,T.
    dY = Y[:, 1:, :] - Y[:, :-1, :]  # Shape: [B, T-1, n]
    # Set R0 = ΔY_t and R1 = Y_{t-1} (i.e. levels, excluding the last observation)
    R0 = dY  # Shape: [B, T-1, n]
    R1 = Y[:, :T_prime, :]  # Shape: [B, T-1, n]
    
    # If a constant is included, detrend R0 and R1 by removing their time means.
    if deterministic.lower() == "constant":
        mean_R0 = R0.mean(dim=1, keepdim=True)  # [B, 1, n]
        mean_R1 = R1.mean(dim=1, keepdim=True)  # [B, 1, n]
        R0 = R0 - mean_R0
        R1 = R1 - mean_R1
    elif deterministic.lower() != "none":
        raise ValueError("deterministic must be either 'none' or 'constant'")
    
    # Compute covariance matrices for each batch.
    # We will compute S00, S11, and S01 (and S10 = S01^T) for each batch.
    # Note: for each batch b, we treat R0[b] and R1[b] as matrices of shape [T_prime, n].
    # We compute S00[b] = (R0[b]^T R0[b]) / T_prime, etc.
    S00 = torch.zeros(B, n, n, device=device, dtype=dtype)
    S11 = torch.zeros(B, n, n, device=device, dtype=dtype)
    S01 = torch.zeros(B, n, n, device=device, dtype=dtype)
    
    for b in range(B):
        R0_b = R0[b]  # [T_prime, n]
        R1_b = R1[b]  # [T_prime, n]
        S00[b] = (R0_b.t() @ R0_b) / T_prime
        S11[b] = (R1_b.t() @ R1_b) / T_prime
        S01[b] = (R0_b.t() @ R1_b) / T_prime
    
    # For each batch, compute the matrix M = S11^{-1} S10 S00^{-1} S01.
    # Here, S10 = S01^T.
    M = torch.zeros(B, n, n, device=device, dtype=dtype)
    for b in range(B):
        S10_b = S01[b].t()
        inv_S00 = torch.linalg.inv(S00[b])
        inv_S11 = torch.linalg.inv(S11[b])
        M[b] = inv_S11 @ S10_b @ inv_S00 @ S01[b]
    
    # Compute eigenvalues for each batch.
    # The eigenvalues are computed from M and should be real numbers.
    eigenvalues = []
    for b in range(B):
        # torch.linalg.eig returns complex numbers even if the imaginary parts are 0.
        eigvals = torch.linalg.eig(M[b])[0]
        # Take the real part and sort in descending order.
        eigvals = eigvals.real
        eigvals, _ = torch.sort(eigvals, descending=True)
        eigenvalues.append(eigvals)
    eigenvalues = torch.stack(eigenvalues, dim=0)  # shape: [B, n]
    
    # Compute the trace statistic for each possible cointegration rank.
    # For each batch b and for r = 0,1,...,n-1, the trace statistic is defined as:
    #   Trace(r) = -T_prime * sum_{i=r+1}^{n} ln(1 - eigenvalue_i)
    trace_stats = torch.zeros(B, n, device=device, dtype=dtype)
    for b in range(B):
        for r in range(n):
            if r < n:
                # Sum from index r to n-1 (i.e. testing null of at most r cointegrating relations)
                lam_sum = torch.log(1 - eigenvalues[b, r:]).sum()
                trace_stats[b, r] = -T_prime * lam_sum
    # Now, trace_stats[b, 0] tests the null of zero cointegrating vectors,
    # trace_stats[b, 1] tests the null of at most one cointegrating vector, and so on.
    
    return eigenvalues, trace_stats

if __name__ == '__main__':
    # Example usage:
    import time

    # Simulation parameters.
    B = 3    # number of batches/regression problems
    T = 200  # number of time points
    n = 3    # number of series per batch

    torch.manual_seed(42)

    # For a simple simulation, we create cointegrated series as follows.
    # Generate a common stochastic trend Z_t (random walk) and add stationary noise.
    Z = torch.zeros(B, T)
    Z[:, 0] = torch.randn(B)
    for t in range(1, T):
        Z[:, t] = Z[:, t-1] + torch.randn(B) * 0.2

    # Generate cointegrated series by adding different stationary components.
    Y = torch.zeros(B, T, n)
    for i in range(n):
        noise = torch.randn(B, T) * 0.5
        Y[:, :, i] = Z + noise

    # Run Johansen test without deterministic adjustment.
    eigenvals, trace_stats = johansen_test(Y, deterministic="none")
    print("Eigenvalues (each row sorted descending):")
    print(eigenvals)
    print("\nTrace Statistics (each column: null hypothesis of ≤ r cointegrating vectors):")
    print(trace_stats)

