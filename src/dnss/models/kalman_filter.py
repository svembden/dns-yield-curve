import pandas as pd
import numpy as np
from scipy.linalg import cholesky, solve_triangular, inv
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import minimize
from scipy.stats import norm
from dnss.utils.logging import setup_logger
from dnss.utils.helpers import input_checks
from datetime import datetime

class KALMAN:
    """Extended Kalman Filter for Dynamic Nelson-Siegel yield curve modeling."""

    def __init__(self, custom_logger=None):
        """
        Initialize KALMAN model.

        Parameters:
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

    @staticmethod
    def _ekf_loglik(params: np.ndarray, y: np.ndarray, tau: np.ndarray) -> float:
        """
        Extended Kalman Filter negative log-likelihood.
        """
        T, N = y.shape
        
        # Unpack params
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
        # P_tt = np.diag(np.var(fac, axis=0).tolist() + [1e-4]) # based on empirical variance of factors

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
                               lambda_init: float = 0.4, q_scale: float = 0.01):
        """
        Initial guess for EKF parameter estimation.
        """
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
        # phi = np.array([0.9, 0.9, 0.9, 0.9]) # fix phi to 0.9

        q = np.tril(np.random.randn(4,4) * q_scale) # process noise
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
        bounds = [(None,None)]*4 + [(0.001,0.999)]*4 + [(-1,1)]*16 + [(1e-6,None)]*len(tau)
        
        # Optimize
        self.logger.info("Starting EKF parameter estimation...")
        start_time = datetime.now()
        result = minimize(self._ekf_loglik, params0,
                          args=(y, tau), bounds=bounds, method='L-BFGS-B')
        self.params = result.x
        self.logger.info(f"Optimized log-likelihood: {-result.fun}. Time taken: {datetime.now()-start_time}")
        
        # Run filter with optimized params
        self.filtered_states, self.covariances = self._run_kalman_filter(self.params, y, tau)
        self.logger.debug(f"Stored {len(self.filtered_states)} filtered states and covariance matrices")
        
        return self

    def _run_kalman_filter(self, params, y, tau):
        """
        Run Kalman filter with given parameters and store states and covariances.
        
        Parameters:
        params (ndarray): Parameter vector.
        y (ndarray): Yield data.
        tau (ndarray): Maturities.
        
        Returns:
        tuple: (filtered_states, covariances)
        """
        T, N = y.shape
        
        # Unpack params
        mu = params[0:4]
        phi = params[4:8]
        q_flat = params[8:24]
        sigma_diag = params[24:24+N]

        A = np.diag(phi)
        q = q_flat.reshape(4, 4)
        Q = q @ q.T
        Sigma = np.diag(sigma_diag)

        filtered_states = np.zeros((T, 4))
        covariances = []
    
        x_tt = mu.copy()
        P_tt = np.eye(4) * 0.1

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
                K = P_pred @ H.T @ inv(S_t)
                x_tt = x_pred + K @ v
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
        
        A = np.diag(phi) # transition matrix
        q = q_flat.reshape(4, 4)
        Q = q @ q.T # process noise covariance
        
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
        yield_curves = self._generate_yield_curves(forecast_df, self.maturities)
        
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

    def _generate_yield_curves(self, params_df, maturities=None):
        """
        Generate yield curves from DNS parameters.
        
        Parameters:
        params_df (DataFrame): DNS parameters with columns ['L', 'S', 'C', 'lambda'].
        maturities (ndarray, optional): Maturities to use. If None, use self.maturities.
        
        Returns:
        DataFrame: Generated yield curves.
        """
        if maturities is None:
            maturities = self.maturities
        
        yields = np.zeros((len(params_df), len(maturities)))
        
        for i, (_, row) in enumerate(params_df.iterrows()):
            L, S, C, lam = row
            for j, tau in enumerate(maturities):
                # DNS loading functions
                e = np.exp(-lam * tau)
                term = (1 - e) / (lam * tau) if lam * tau > 1e-6 else 1.0
                yields[i, j] = L + S * term + C * (term - e)
        
        yield_curves = pd.DataFrame(yields, index=params_df.index, 
                                columns=[f'tau_{tau:.4f}' for tau in maturities])
        
        return yield_curves
