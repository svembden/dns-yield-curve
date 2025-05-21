import pandas as pd
import numpy as np
from scipy.linalg import cholesky, solve_triangular, inv
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize
from scipy.stats import norm
from dnss.utils.logging import setup_logger
from dnss.utils.helpers import input_checks, generate_yield_curves
from datetime import datetime

class KALMAN:
    """Extended Kalman Filter for Dynamic Nelson-Siegel yield curve modeling."""

    def __init__(self, fix_lambda=False, lambda_value=0.4, custom_logger=None):
        """
        Initialize KALMAN model.

        Parameters:
        fix_lambda (bool): Whether to fix the lambda parameter at a constant value
        lambda_value (float): Value to use for lambda if fixed
        custom_logger (Logger, optional): Custom logger instance. If None, a default logger will be created.
        """
        if custom_logger is None:
            self.logger = setup_logger(__name__) # Create logger without log file
        else:
            self.logger = custom_logger
        self.params = None
        self.filtered_states = None
        self.covariances = None
        self.maturities = None
        self.dates = None
        self.fix_lambda = fix_lambda
        self.lambda_value = lambda_value

    def _ekf_loglik(self, params: np.ndarray, y: np.ndarray, tau: np.ndarray) -> float:
        """
        Extended Kalman Filter negative log-likelihood.
        """
        T, N = y.shape
        
        # Unpack params - different parameter structure when lambda is fixed
        if self.fix_lambda:
            # When lambda is fixed, we only have L, S, C as states to estimate
            mu = params[0:3]
            phi = params[3:6]
            q_flat = params[6:15]  # 3x3 lower triangular for process noise
            sigma_diag = params[15:15+N]
            
            # Construct full state with fixed lambda
            mu_full = np.zeros(4)
            mu_full[:3] = mu
            mu_full[3] = self.lambda_value
            
            phi_full = np.zeros(4)
            phi_full[:3] = phi
            phi_full[3] = 1.0  # Fixed lambda (no dynamics)
            
            # Create transition matrix
            A = np.diag(phi_full)
            
            # Create process noise matrix
            q = q_flat.reshape(3, 3)
            Q = np.zeros((4, 4))
            Q[:3, :3] = q @ q.T  # Only applies to L, S, C
        else:
            # Original case with lambda as a state
            mu_full = params[0:4]
            phi_full = params[4:8]
            q_flat = params[8:24]
            sigma_diag = params[24:24+N]
            
            A = np.diag(phi_full)
            q = q_flat.reshape(4, 4)
            Q = q @ q.T
        
        Sigma = np.diag(sigma_diag)

        log_lik = 0.0
        x_tt = mu_full.copy()
        P_tt = np.eye(4) * 0.1

        for t in range(T):
            # Prediction
            x_pred = mu_full + A @ (x_tt - mu_full)
            P_pred = A @ P_tt @ A.T + Q

            L, S, C, lam = x_pred
            if self.fix_lambda:
                lam = self.lambda_value  # Override with fixed value
            else:
                lam = np.clip(lam, 1e-4, 10)  # ensure positive
            
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
                H[i,0] = 1  # dR/dL = 1
                H[i,1] = B[i,1]  # dR/dS = B_1
                H[i,2] = B[i,2]  # dR/dC = B_2
                
                if not self.fix_lambda:
                    # Only compute lambda derivatives if lambda is not fixed
                    dS = (tau_i*lam*e - (1 - e)) / (tau_i * lam**2)
                    dC = dS + tau_i*e
                    dLam = S * dS + C * dC
                    H[i,3] = dLam
                else:
                    H[i,3] = 0  # No need for lambda derivative when fixed

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
            
            if self.fix_lambda:
                x_tt[3] = self.lambda_value  # Maintain fixed lambda
                
            P_tt = (np.eye(4) - K @ H) @ P_pred

        return log_lik

    def _initialize_dns_params(self, y: np.ndarray, tau: np.ndarray,
                               q_scale: float = 0.01):
        """
        Initial guess for EKF parameter estimation.
        """
        # PCA on centered yields
        y_centered = y - y.mean(axis=0)
        pca = PCA(n_components=3)
        fac = pca.fit_transform(y_centered)
        
        # Initial parameter setup differs with fixed lambda
        if self.fix_lambda:
            mu = np.array([fac[:,0].mean(), fac[:,1].mean(), fac[:,2].mean()])
            
            # AR(1) coeffs for L, S, C only
            phi = []
            for i in range(3):
                model = AutoReg(fac[:, i], lags=1).fit()
                phi.append(model.params[1])
            phi = np.array(phi)
            
            # Process noise for 3x3 matrix (L,S,C only)
            q = np.tril(np.random.randn(3, 3) * q_scale)
            q_flat = q.flatten()
        else:
            # Original case with lambda as a state
            mu = np.array([fac[:,0].mean(), fac[:,1].mean(), fac[:,2].mean(), self.lambda_value])
            
            # AR(1) coeffs including lambda
            phi = []
            for i in range(3):
                model = AutoReg(fac[:, i], lags=1).fit()
                phi.append(model.params[1])
            phi.append(0.9)  # phi lambda
            phi = np.array(phi)
            
            # Process noise for 4x4 matrix
            q = np.tril(np.random.randn(4, 4) * q_scale)
            q_flat = q.flatten()
        
        sigma = np.std(y, axis=0) * 0.5 # obs noise * pragmatic scaling
        params0 = np.concatenate([mu, phi, q_flat, sigma]) # pack
        
        return params0

    def fit(self, dates: pd.DatetimeIndex, maturities: np.ndarray,
            data: pd.DataFrame) -> "KALMAN":
        """
        Estimate DNS parameters via EKF.

        Parameters:
        dates (DatetimeIndex): Observation dates.
        maturities (ndarray): Maturities (N,).
        data (DataFrame): Yields (T x N).

        Returns:
        KALMAN: Self with fitted parameters.
        """
        input_checks(dates, data, maturities)
        self.dates = dates
        self.maturities = maturities
        y = data.values
        tau = maturities
        
        # Init
        params0 = self._initialize_dns_params(y, tau)
        self.logger.debug(f"Initial parameters: {params0}")
        
        # Bounds depend on whether lambda is fixed
        if self.fix_lambda:
            # mu (3), phi (3), q_elements (9), and sigma (N)
            bounds = [(None,None)]*3 + [(0.001,0.999)]*3 + [(-1,1)]*9 + [(1e-6,None)]*len(tau)
        else:
            bounds = [(None,None)]*4 + [(0.001,0.999)]*4 + [(-1,1)]*16 + [(1e-6,None)]*len(tau)
        
        # Optimize
        if self.fix_lambda:
            self.logger.info(f"Starting EKF parameter estimation with fixed lambda={self.lambda_value}...")
        else:
            self.logger.info("Starting EKF parameter estimation with variable lambda...")
        start_time = datetime.now()
        result = minimize(lambda p: self._ekf_loglik(p, y, tau), params0,
                          bounds=bounds, method='L-BFGS-B')
        self.params = result.x
        self.logger.info(f"Optimized log-likelihood: {-result.fun}. Time taken: {datetime.now()-start_time}")
        
        # Run filter with optimized params
        self.filtered_states, self.covariances = self._run_kalman_filter(self.params, y, tau)
        self.logger.debug(f"Stored {len(self.filtered_states)} filtered states and covariance matrices")
        
        return self

    def _run_kalman_filter(self, params, y, tau):
        """
        Run Kalman filter with given parameters and store states and covariances.
        """
        T, N = y.shape
        
        # Unpack params
        if self.fix_lambda:
            mu = params[0:3]
            phi = params[3:6]
            q_flat = params[6:15]
            sigma_diag = params[15:15+N]
            
            mu_full = np.zeros(4)
            mu_full[:3] = mu
            mu_full[3] = self.lambda_value
            
            phi_full = np.zeros(4)
            phi_full[:3] = phi
            phi_full[3] = 1.0  # fixed lambda
            
            A = np.diag(phi_full) # transition matrix
            q = q_flat.reshape(3, 3)
            Q = np.zeros((4, 4)) # process noise covariance
            Q[:3, :3] = q @ q.T 
        else:
            mu_full = params[0:4]
            phi_full = params[4:8]
            q_flat = params[8:24]
            sigma_diag = params[24:24+N]
            
            A = np.diag(phi_full)
            q = q_flat.reshape(4, 4)
            Q = q @ q.T

        Sigma = np.diag(sigma_diag)

        filtered_states = np.zeros((T, 4))
        covariances = []

        x_tt = mu_full.copy()
        P_tt = np.eye(4) * 0.1

        for t in range(T):
            # Prediction
            x_pred = mu_full + A @ (x_tt - mu_full)
            P_pred = A @ P_tt @ A.T + Q

            L, S, C, lam = x_pred
            if self.fix_lambda:
                lam = self.lambda_value
            else:
                lam = np.clip(lam, 1e-4, 10)  # ensure positive
            
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
                K = P_pred @ H.T @ inv(S_t)
                x_tt = x_pred + K @ v
                if self.fix_lambda:
                    x_tt[3] = self.lambda_value
                P_tt = (np.eye(4) - K @ H) @ P_pred
            except np.linalg.LinAlgError:
                self.logger.warning(f"Numerical issue at step {t}. Using previous state and covariance.")
                
            # Store filtered state and covariance
            filtered_states[t] = x_tt
            covariances.append(P_tt.copy())

        return filtered_states, covariances

    def predict(self, steps=10, conf_int=0.95, return_param_estimates=False):
        """
        Forecast future values of the yield curve using the Kalman filter.
        
        Parameters:
        steps (int): Number of steps ahead to forecast. Default is 10.
        conf_int (float): Confidence interval for the forecast. Default is 0.95.
        return_param_estimates (bool): Whether to return parameter estimates and forecast variance. Default is False.
        
        Returns:
        DataFrame: Forecasted yield curves.
        DataFrame: Forecasted parameters (if return_param_estimates is True).
        list: Forecasted covariance matrices (if return_param_estimates is True).
        tuple: Forecasted confidence intervals (if return_param_estimates is True).
        """
        if self.params is None or self.filtered_states is None:
            raise ValueError("Model not fitted. Please fit the model first.")
        
        self.logger.info(f"Forecasting {steps} steps ahead...")
        
        # Unpack params
        N = len(self.maturities)
        mu = self.params[0:4]
        phi = self.params[4:8]
        q_flat = self.params[8:24]
        
        A = np.diag(phi) 
        q = q_flat.reshape(4, 4)
        Q = q @ q.T 
        
        # Starting point foreecast
        last_state = self.filtered_states[-1]
        last_cov = self.covariances[-1]
        
        forecast_states = np.zeros((steps, 4))
        forecast_covs = []

        z_value = -1 * norm.ppf((1 - conf_int) / 2) # for CI
        
        x_pred = last_state.copy()
        P_pred = last_cov.copy()
        
        for i in range(steps):
            # State prediction
            x_pred = mu + A @ (x_pred - mu)
            P_pred = A @ P_pred @ A.T + Q
            forecast_states[i] = x_pred
            forecast_covs.append(P_pred.copy())
        
        # Create DF for forecasts
        freq = pd.infer_freq(self.dates)
        start_date = self.dates[-1] + pd.tseries.frequencies.to_offset(freq)
        forecast_index = pd.date_range(start=start_date, periods=steps, freq=freq)
        
        forecast_df = pd.DataFrame(forecast_states, index=forecast_index, 
                                columns=['L', 'S', 'C', 'lambda'])
        
        # Generate yield curves from forecasted parameters
        yield_curves = generate_yield_curves(forecast_df, self.maturities)
        
        # Calculate confidence intervals
        lower_bounds = []
        upper_bounds = []
        
        for i in range(steps):
            state_std = np.sqrt(np.diag(forecast_covs[i]))
            lower_bound = forecast_states[i] - z_value * state_std
            upper_bound = forecast_states[i] + z_value * state_std
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
        
        lower_bound_df = pd.DataFrame(lower_bounds, index=forecast_index, 
                                    columns=['L', 'S', 'C', 'lambda'])
        upper_bound_df = pd.DataFrame(upper_bounds, index=forecast_index, 
                                    columns=['L', 'S', 'C', 'lambda'])
        
        forecast_intervals = (lower_bound_df, upper_bound_df)
        
        self.logger.info("Forecasting complete.")
        
        if return_param_estimates:
            return yield_curves, forecast_df, forecast_covs, forecast_intervals
        
        return yield_curves
