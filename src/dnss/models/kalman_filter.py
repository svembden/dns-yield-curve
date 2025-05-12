#@TODO IMPLEMENT:

import pandas as pd
import numpy as np
from scipy.linalg import cholesky, solve_triangular, inv
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize
from dnss.utils.logging import setup_logger
from dnss.utils.helpers import input_checks


class KALMAN:
    """Extended Kalman Filter for Dynamic Nelson-Siegel yield curve modeling."""

    def __init__(self, custom_logger=None):
        """
        Initialize KALMAN model.

        Parameters:
        custom_logger (Logger, optional): Custom logger instance. If None, a default logger will be created.
        """
        if custom_logger is None:
            self.logger = setup_logger(__name__)
        else:
            self.logger = custom_logger
        self.params = None        # Estimated parameter vector
        self.filtered_states = None  # Kalman filtered state estimates
        self.covariances = None     # Filter covariance matrices
        self.maturities = None
        self.dates = None

    @staticmethod
    def _ekf_loglik(params: np.ndarray, y: np.ndarray, tau: np.ndarray) -> float:
        """
        Extended Kalman Filter negative log-likelihood.
        """
        T, N = y.shape
        # Unpack parameters
        mu = params[0:4]
        phi = params[4:8]
        q_flat = params[8:24]
        sigma_diag = params[24:24+N]

        A = np.diag(phi)
        q = q_flat.reshape(4, 4)
        Q = q @ q.T
        Sigma = np.diag(sigma_diag)

        log_lik = 0.0
        x_tt = mu.copy()
        P_tt = np.eye(4) * 0.1
        # P_tt = np.diag(np.var(fac, axis=0).tolist() + [0.01]) # based on empirical variance of factors

        for t in range(T):
            # Prediction
            x_pred = mu + A @ (x_tt - mu)
            P_pred = A @ P_tt @ A.T + Q

            L, S, C, lam = x_pred
            lam = np.clip(lam, 1e-4, 10) # ensure positive
            # Build B and H
            B = np.zeros((N, 3))
            H = np.zeros((N, 4))
            for i in range(N):
                tau_i = tau[i]
                e = np.exp(-tau_i * lam)
                denom = tau_i * lam if tau_i*lam>1e-6 else 1e-6
                B[i,0] = 1
                B[i,1] = (1 - e) / denom
                B[i,2] = B[i,1] - e
                # Derivatives
                dS = (tau_i*lam*e - (1 - e)) / (tau_i * lam**2)
                dC = dS + tau_i*e
                dLam = S * dS + C * dC
                H[i,0] = 1
                H[i,1] = B[i,1]
                H[i,2] = B[i,2]
                H[i,3] = dLam

            # Observation prediction
            h = B @ np.array([L, S, C])
            v = y[t] - h
            S_t = H @ P_pred @ H.T + Sigma

            try:
                L_s = cholesky(S_t, lower=True)
                alpha = solve_triangular(L_s, v, lower=True)
                log_det = 2.0 * np.sum(np.log(np.diag(L_s)))
                log_lik += 0.5 * (alpha.T @ alpha + log_det + N * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                return 1e10 # large penalty for non-positive definite

            K = P_pred @ H.T @ inv(S_t)
            x_tt = x_pred + K @ v
            P_tt = (np.eye(4) - K @ H) @ P_pred

        return log_lik

    @staticmethod
    def _initialize_dns_params(y: np.ndarray, tau: np.ndarray,
                               lambda_init: float = 0.5, q_scale: float = 0.01):
        """
        Initial guess for EKF parameter estimation.
        """
        T, N = y.shape
        # PCA on centered yields
        y_centered = y - y.mean(axis=0)
        pca = PCA(n_components=3)
        fac = pca.fit_transform(y_centered)
        mu = np.array([fac[:,0].mean(), fac[:,1].mean(), fac[:,2].mean(), lambda_init])
        # AR(1) coeffs
        phi = []
        for i in range(3):
            model = AutoReg(fac[:, i], lags=1).fit()
            phi.append(model.params[1])
        phi.append(0.9)  # phi lambda
        phi = np.array(phi)
        # process noise
        q = np.tril(np.random.randn(4,4) * q_scale)
        q_flat = q.flatten()
        # obs noise
        sigma = np.std(y, axis=0) * 0.5 # pragmatic way of shrinking
        # pack
        params0 = np.concatenate([mu, phi, q_flat, sigma])
        
        return params0

    def fit(self, dates: pd.DatetimeIndex, maturities: np.ndarray,
            data: pd.DataFrame) -> np.ndarray:
        """
        Estimate DNS parameters via EKF.

        Parameters:
        dates (DatetimeIndex): Observation dates.
        maturities (ndarray): Maturities (N,).
        data (DataFrame): Yields (T x N).

        Returns:
        ndarray: Estimated parameter vector.
        """
        input_checks(dates, data, maturities)
        self.dates = dates
        self.maturities = maturities
        y = data.values
        tau = maturities
        # initialize
        params0 = self._initialize_dns_params(y, tau)
        self.logger.info(f"Initial parameters: {params0}")
        bounds = [(None,None)]*4 + [(0.001,0.999)]*4 + [(-1,1)]*16 + [(1e-6,None)]*len(tau)
        self.logger.info("Starting EKF parameter estimation...")
        result = minimize(self._ekf_loglik, params0,
                          args=(y, tau), bounds=bounds, method='L-BFGS-B')
        self.params = result.x
        self.logger.info(f"Optimized log-likelihood: {-result.fun}")
        
        return self


