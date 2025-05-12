#@TODO IMPLEMENT:

import pandas as pd
import numpy as np
from scipy.linalg import cholesky, solve_triangular, inv
from sklearn.decomposition import PCA
from scipy.optimize import minimize

def ekf_loglik(params, y, tau):
    """
    Extended Kalman Filter for the Dynamic Nelson-Siegel model with time-varying lambda.

    Arguments:
        params: flattened vector of model parameters
        y: observed yields [T x N]
        tau: maturities [N]

    Returns:
        Negative log-likelihood
    """

    T, N = y.shape

    # Unpack parameters
    mu = params[0:4]
    phi = params[4:8]
    q_flat = params[8:8 + 16]     # q is 4x4, flattened row-wise
    sigma_diag = params[24:24 + N]

    A = np.diag(phi)
    q = q_flat.reshape(4, 4)
    Q = q @ q.T
    Sigma = np.diag(sigma_diag)

    # Initialize storage
    log_likelihood = 0.0
    x_tt = mu.copy()        # x_{0|0}
    P_tt = np.eye(4) * 0.1  # Initial covariance

    for t in range(T):
        ### Prediction ###
        x_pred = mu + A @ (x_tt - mu)
        P_pred = A @ P_tt @ A.T + Q

        L, S, C, lam = x_pred
        lam = np.clip(lam, 1e-4, 10)  # Keep lambda in safe range


        # Build B matrix
        B = np.zeros((N, 3))
        dL = np.ones(N)
        dS = np.zeros(N)
        dC = np.zeros(N)
        dLam = np.zeros(N)

        for i in range(N):
            tau_i = tau[i]
            e = np.exp(-tau_i * lam)
            B[i, 0] = 1
            denom = tau_i * lam if tau_i * lam > 1e-6 else 1e-6
            B[i, 1] = (1 - e) / denom
            B[i, 2] = B[i, 1] - e

            # Derivatives w.r.t. lambda
            dS[i] = (tau_i * lam * e - (1 - e)) / (tau_i * lam**2)
            dC[i] = dS[i] + tau_i * e

            dLam[i] = S * dS[i] + C * dC[i]  # Chain rule on B(x)

        # Observation prediction
        h = B @ np.array([L, S, C])

        # Jacobian H: Nx4 (L, S, C, lambda)
        H = np.zeros((N, 4))
        H[:, 0] = dL
        H[:, 1] = B[:, 1]
        H[:, 2] = B[:, 2]
        H[:, 3] = dLam

        # Innovation
        v = y[t] - h
        S_t = H @ P_pred @ H.T + Sigma

        try:
            # Cholesky for numerical stability
            L_s = cholesky(S_t, lower=True)
            alpha = solve_triangular(L_s, v, lower=True)
            log_det = 2.0 * np.sum(np.log(np.diag(L_s)))
            log_likelihood += 0.5 * (alpha.T @ alpha + log_det + N * np.log(2 * np.pi))
        except np.linalg.LinAlgError:
            return 1e10  # Penalize invalid covariance

        # Kalman gain
        K = P_pred @ H.T @ inv(S_t)

        # Update
        x_tt = x_pred + K @ v
        P_tt = (np.eye(4) - K @ H) @ P_pred

    return log_likelihood


def initialize_dns_params(y, tau, lambda_init=0.5, q_scale=0.01):
    """
    Generate a reasonable initial guess for EKF parameter estimation.

    Args:
        y (T x N ndarray): yield curve data
        tau (N ndarray): maturities
        lambda_init (float): initial guess for lambda
        q_scale (float): scale of state process noise

    Returns:
        params0: initial parameter vector
        bounds: list of bounds (tuples) for each parameter
    """
    T, N = y.shape

    # Estimate factors using PCA
    y_centered = y - y.mean(axis=0)
    pca = PCA(n_components=3)
    factors = pca.fit_transform(y_centered)

    mu_L = np.mean(factors[:, 0])
    mu_S = np.mean(factors[:, 1])
    mu_C = np.mean(factors[:, 2])
    mu_lambda = lambda_init

    # AR(1) coefficients
    phi = [0.95, 0.9, 0.8, 0.9]

    # q: small process noise, lower-triangular
    q = np.tril(np.random.randn(4, 4) * q_scale)
    q_flat = q.flatten()

    # Measurement noise std devs (diagonal of Sigma)
    sigma = np.std(y, axis=0) * 0.5

    # Final parameter vector
    params0 = np.concatenate([
        [mu_L, mu_S, mu_C, mu_lambda],  # 4 means
        phi,                            # 4 AR coefficients
        q_flat,                         # 16 q entries
        sigma                           # N observation noise
    ])

    return params0

def main():
    # Dummy dimensions
    T, N = 100, 8
    tau = np.linspace(0.25, 10, N)
    y = np.random.randn(T, N) * 0.01  # Replace with real data
    
    print("Shape of y:", y.shape)
    print("Type of y:", type(y))
    print(y)

    # Initial guess
    params0 = initialize_dns_params(y, tau)
    print("Initial parameters:", params0)

    # Bounds (optional, helpful for stability)
    bounds = [(None, None)]*4 + [(0.001, 0.999)]*4 + [(-1, 1)]*16 + [(1e-6, None)]*N

    result = minimize(ekf_loglik, params0, args=(y, tau), bounds=bounds, method='L-BFGS-B')

    print("Optimized log-likelihood:", -result.fun)
    print("Estimated parameters:", result.x)
    

# main()

def TEMP_KALMAN(dates, maturities, data):
    """
    Temporary function to run Kalman filter on data.
    """
    
    T, N = data.shape
    tau = maturities
    y = np.array(data.values)
    

    
    
    params0 = initialize_dns_params(y, tau)
    logger.debug("Initial parameters:", params0)
    bounds = [(None, None)]*4 + [(0.001, 0.999)]*4 + [(-1, 1)]*16 + [(1e-6, None)]*N
    result = minimize(ekf_loglik, params0, args=(y, tau), bounds=bounds, method='L-BFGS-B')
    print("Optimized log-likelihood:", -result.fun)
    print("Estimated parameters:", result.x)
    return result.x