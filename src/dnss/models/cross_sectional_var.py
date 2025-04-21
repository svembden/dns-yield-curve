import pandas as pd
import numpy as np
from dnss.utils.logging import setup_logger
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR

# To add:
# - Fixed lambda optionality
# - Svenson

class CSVAR:
    """Cross-Sectional Vector Autoregression model for yield curve analysis."""
    
    def __init__(self, custom_logger=None):
        """
        Initialize CSVAR model.
        
        Parameters:
        custom_logger (Logger, optional): Custom logger instance. If None, a default logger will be created.
        """
        if custom_logger is None:
            self.logger = setup_logger(__name__)  # Create logger without log file
        else:
            self.logger = custom_logger
        self.params = None
        self.var_results = None
        self.maturities = None
    
    @staticmethod
    def _nelson_siegel_function(tau, beta0, beta1, beta2, lambda_):
        """Nelson-Siegel yield function.
        
        Parameters:
        tau (array-like): Maturities.
        beta0 (float): Long-term yield.
        beta1 (float): Short-term yield.
        beta2 (float): Medium-term yield.
        lambda_ (float): Decay factor.
        
        Returns:
        array-like: The yield curve values.
        """
        term1 = (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
        term2 = term1 - np.exp(-lambda_ * tau)
        
        return beta0 + beta1 * term1 + beta2 * term2

    
    def _input_checks(self, dates, data, maturities):
        """
        Perform input checks for the CSVAR model.
        
        Parameters:
        dates (datetime-like): The dates for which to estimate parameters.
        data (DataFrame): The input data containing yield curve information.
        maturities (array-like): The maturities corresponding to the columns of the data.
        
        Raises:
        ValueError: If any of the input checks fail.
        """
        if not isinstance(dates, (pd.DatetimeIndex, pd.Series)):
            raise ValueError("Input dates must be a pandas DatetimeIndex or Series.")
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise ValueError("Input data must be a pandas DataFrame or Series.")
        if data.empty:
            raise ValueError("Input data is empty.")
        if data.isnull().values.any():
            raise ValueError("Input data contains NaN values.")
        if not isinstance(maturities, (list, np.ndarray)):
            raise ValueError("Maturities must be provided as a list or numpy array.")
        if len(maturities) != data.shape[1]:
            raise ValueError("Length of maturities must match the number of columns in data.")
        if not len(dates) == len(data):
            raise ValueError("Length of dates must match the number of rows in data.")
    
    
    def _estimate_cross_sectional_parameters(self, dates, data, maturities):
        """
        Estimate cross-sectional DNS parameters (βs) at each point in time.
        
        Parameters:
        dates (datetime-like): The dates for which to estimate parameters.
        data (DataFrame): The input data containing yield curve information.
        maturities (array-like): The maturities corresponding to the columns of the data.
        
        Returns:
        DataFrame: A DataFrame containing the estimated parameters.
        """        
        # Store maturities for later use
        self.maturities = np.array(maturities, dtype=float)
        
        # Input Checks
        self._input_checks(dates, data, maturities)
        
        # Initialize nelson-siegel parameters
        params = pd.DataFrame(index=dates, columns=['beta0', 'beta1', 'beta2', 'lambda'])
        params = params.fillna(0).infer_objects(copy=False)
        params = params.astype(float)
        
        mean_yields = data.mean(axis=0)
        tau = np.array(maturities, dtype=float)
        
        # Initial parameter guesses
        initial_params = [
            mean_yields.iloc[-1],  # beta0
            mean_yields.iloc[0] - mean_yields.iloc[-1],  # beta1
            1.0,  # beta2
            0.4  # lambda
        ]
        
        self.logger.debug(f"Initial parameter guesses:\n{initial_params}")
        
        # Bounds for parameters
        bounds = [
            (0, None),  # beta0
            (None, None),  # beta1
            (0, None),  # beta2
            (1e-8, None)  # lambda
        ]
        
        # Objective function to minimize
        def objective_function(params, maturities, observed_yields):
            beta0, beta1, beta2, lambda_ = params
            model_yields = self._nelson_siegel_function(maturities, beta0, beta1, beta2, lambda_)
            return np.sum((observed_yields - model_yields) ** 2)
        
        self.logger.info("Starting parameter estimation...")
        
        # Estimate parameters for each date
        for i, date in enumerate(data.index):
            observed_yields = data.loc[date].values
            
            # Optimize to find the best parameters
            result = minimize(
                objective_function,
                initial_params,
                args=(tau, observed_yields),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            # Store optimized parameters
            params.loc[dates[i], 'beta0'] = result.x[0]
            params.loc[dates[i], 'beta1'] = result.x[1]
            params.loc[dates[i], 'beta2'] = result.x[2]
            params.loc[dates[i], 'lambda'] = result.x[3]
        
        self.params = params
        return params

    
    def _fit_var_model(self, params=None, maxlags=5, ic='aic'):
        """
        Fit a VAR model to the estimated parameters.
        
        Parameters:
        params (DataFrame, optional): The estimated parameters DataFrame. If None, uses previously estimated params.
        maxlags (int, optional): Maximum number of lags. Default is 5.
        ic (str, optional): Information criterion to use. Default is 'aic'.
        
        Returns:
        VARResults: The fitted VAR model results.
        """
        self.logger.info("Fitting VAR model...")
        
        if params is None:
            if self.params is None:
                raise ValueError("No parameters available. Please estimate parameters first.")
            params = self.params
        
        # Fit the VAR model
        model = VAR(params)
        results = model.fit(maxlags=maxlags, ic=ic)
        
        self.var_results = results
        return results
    
    
    def _generate_yield_curves(self, parameter_df):
        """
        Generate yield curves from parameters.
        
        Parameters:
        parameter_df (DataFrame): DataFrame containing beta0, beta1, beta2, and lambda parameters.
        
        Returns:
        DataFrame: Yield curves for each date in the parameter DataFrame.
        """
        if self.maturities is None:
            raise ValueError("Maturities not defined. Please estimate parameters first.")
        
        yield_curves = pd.DataFrame(index=parameter_df.index, columns=self.maturities)
        
        for date in parameter_df.index:
            beta0 = parameter_df.loc[date, 'beta0']
            beta1 = parameter_df.loc[date, 'beta1']
            beta2 = parameter_df.loc[date, 'beta2']
            lambda_ = parameter_df.loc[date, 'lambda']
            
            yield_curves.loc[date] = self._nelson_siegel_function(
                self.maturities, beta0, beta1, beta2, lambda_
            )
        
        # Ensure yield curves is df
        yield_curves = yield_curves.astype(float)
        yield_curves.index = pd.to_datetime(yield_curves.index)
        
        return yield_curves
    
    
    def fit(self, dates, maturities, data, maxlags=5, ic='aic', return_var_model=False):
        """
        Fit the Cross-Sectional VAR model in two steps:
        1. Estimate Nelson-Siegel parameters for each date
        2. Fit a VAR model to the estimated parameters
        
        Parameters:
        dates (datetime-like): The dates for which to estimate parameters.
        data (DataFrame): The input data containing yield curve information.
        maturities (array-like): The maturities corresponding to the columns of the data.
        maxlags (int, optional): Maximum number of lags for VAR model. Default is 5.
        ic (str, optional): Information criterion to use. Default is 'aic'.
        return_var_model (bool, optional): Whether to return the VAR model. Default is False.
        
        Returns:
        tuple or object: If return_var_model is True, returns (params, var_results), 
                        otherwise returns the CSVAR model instance.
        """
        # Step 1: Estimate cross-sectional parameters
        params = self._estimate_cross_sectional_parameters(dates, data, maturities)
        params.index.freq = pd.infer_freq(params.index)
        
        # Step 2: Fit VAR model on the estimated parameters
        var_results = self._fit_var_model(params, maxlags=maxlags, ic=ic)
        
        if return_var_model:
            return params, var_results
        
        return self
    
    
    def forecast(self, steps=10, conf_int=0.95, return_param_estimates=False):
        """
        Forecast future values of the yield curve parameters.
        
        Parameters:
        steps (int): Number of steps ahead to forecast. Default is 10.
        conf_int (float): Confidence interval for the forecast. Default is 0.95.
        return_param_estimates (bool): Whether to return parameter estimates and variance. Default is False.
        
        Returns:
        DataFrame: Forecasted yield curves.
        DataFrame: Forecasted parameters (if return_param_estimates is True).
        DataFrame: Forecasted variance (if return_param_estimates is True).
        tuple: Forecasted confidence intervals (if return_param_estimates is True).
        """
        if self.var_results is None:
            raise ValueError("VAR model not fitted. Please fit the model first.")
        
        self.logger.info(f"Forecasting {steps} steps ahead...")
        
        # Generate forecasts
        forecast = self.var_results.forecast(self.params.values, steps=steps)
        forecast_variance = self.var_results.forecast_cov(steps=steps)
        
        # Create DataFrame for forecasts
        freq = pd.infer_freq(self.params.index)
        start_date = self.params.index[-1] + pd.tseries.frequencies.to_offset(freq)
        forecast_index = pd.date_range(start=start_date, periods=steps, freq=freq)
        forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=self.params.columns)
        
        # Calculate confidence intervals (if statsmodels version supports it)
        try:
            forecast_intervals = self.var_results.forecast_interval(self.params.values, steps, alpha=(1-conf_int))
            lower_bound = pd.DataFrame(forecast_intervals[0], index=forecast_index, columns=self.params.columns)
            upper_bound = pd.DataFrame(forecast_intervals[1], index=forecast_index, columns=self.params.columns)
            forecast_intervals = (lower_bound, upper_bound)
        except (AttributeError, TypeError):
            # Fallback if forecast_interval is not available
            self.logger.warning("Confidence intervals calculation not supported in this statsmodels version.")
            forecast_intervals = None
        
        # Generate yield curves from forecasted parameters
        yield_curves = self._generate_yield_curves(forecast_df)
        
        if return_param_estimates:
            return yield_curves, forecast_df, forecast_variance, forecast_intervals
        
        return yield_curves