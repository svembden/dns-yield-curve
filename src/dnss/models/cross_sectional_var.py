import pandas as pd
import numpy as np
from dnss.utils.logging import setup_logger
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR
from dnss.utils.helpers import input_checks

class CSVAR:
    """Cross-Sectional Vector Autoregression model for yield curve analysis."""
    
    def __init__(self, fix_lambda=False, lambda_value=0.4, custom_logger=None):
        """
        Initialize CSVAR model.
        
        Parameters:
        custom_logger (Logger, optional): Custom logger instance. If None, a default logger will be created.
        fix_lambda (bool, optional): Whether to keep lambda fixed across time. Default is False.
        lambda_value (float, optional): Value for lambda when fixed. Default is 0.4.
        """
        if custom_logger is None:
            self.logger = setup_logger(__name__)  # Create logger without log file
        else:
            self.logger = custom_logger
        self.params = None
        self.var_results = None
        self.maturities = None
        self.fix_lambda = fix_lambda
        self.lambda_value = lambda_value
    
    @staticmethod
    def _nelson_siegel_function(tau, L, S, C, lambda_):
        """Nelson-Siegel yield function.
        
        Parameters:
        tau (array-like): Maturities.
        L (float): Long-term yield.
        S (float): Short-term yield.
        C (float): Medium-term yield.
        lambda_ (float): Decay factor.
        
        Returns:
        array-like: The yield curve values.
        """
        tau = np.array(tau, dtype=float)
        
        term1 = (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
        term2 = term1 - np.exp(-lambda_ * tau)
        
        return L + S * term1 + C * term2

    
    
    
    
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
        input_checks(dates, data, maturities)
        
        # Init params
        params = pd.DataFrame(index=dates, columns=['L', 'S', 'C', 'lambda'])
        params = params.fillna(0).infer_objects(copy=False)
        params = params.astype(float)
        
        mean_yields = data.mean(axis=0)
        tau = np.array(maturities, dtype=float)
        
        if self.fix_lambda:
            # Initial parameter guesses (without lambda)
            initial_params = [
                mean_yields.iloc[-1],  # L
                mean_yields.iloc[0] - mean_yields.iloc[-1],  # S
                1.0,  # C
            ]
            
            # Bounds for parameters (without lambda)
            bounds = [
                (0, None),  # L
                (None, None),  # S
                (0, None),  # C
            ]
            
            # Objective function to minimize with fixed lambda
            def objective_function(params, maturities, observed_yields, lambda_value):
                L, S, C = params
                model_yields = self._nelson_siegel_function(maturities, L, S, C, lambda_value)
                return np.sum((observed_yields - model_yields) ** 2)
            
            self.logger.info(f"Starting parameter estimation with fixed lambda={self.lambda_value}...")
            
            # Estimate parameters for each date
            for i, date in enumerate(data.index):
                observed_yields = data.loc[date].values
                
                # Optimize to find the best parameters
                result = minimize(
                    objective_function,
                    initial_params,
                    args=(tau, observed_yields, self.lambda_value),
                    bounds=bounds,
                    method='L-BFGS-B'
                )
                
                # Store optimized parameters
                params.loc[dates[i], 'L'] = result.x[0]
                params.loc[dates[i], 'S'] = result.x[1]
                params.loc[dates[i], 'C'] = result.x[2]
                params.loc[dates[i], 'lambda'] = self.lambda_value
        else:
            # Initial parameter guesses (with lambda)
            initial_params = [
                mean_yields.iloc[-1],  # L
                mean_yields.iloc[0] - mean_yields.iloc[-1],  # S
                1.0,  # C
                0.4  # lambda
            ]
            
            self.logger.debug(f"Initial parameter guesses:\n{initial_params}")
            
            # Bounds for parameters (with lambda)
            bounds = [
                (0, None),  # L
                (None, None),  # S
                (0, None),  # C
                (1e-8, None)  # lambda
            ]
            
            # Objective function to minimize with variable lambda
            def objective_function(params, maturities, observed_yields):
                L, S, C, lambda_ = params
                model_yields = self._nelson_siegel_function(maturities, L, S, C, lambda_)
                return np.sum((observed_yields - model_yields) ** 2)
            
            self.logger.info("Starting parameter estimation with variable lambda...")
            
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
                params.loc[dates[i], 'L'] = result.x[0]
                params.loc[dates[i], 'S'] = result.x[1]
                params.loc[dates[i], 'C'] = result.x[2]
                params.loc[dates[i], 'lambda'] = result.x[3]
        
        self.params = params
        return params

    
    def _fit_var_model(self, params=None, maxlags=1, ic='aic'):
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
        
        # If lambda is fixed, don't include it in the VAR model
        if self.fix_lambda:
            var_params = params[['L', 'S', 'C']].copy()
        else:
            var_params = params.copy()
        
        # Fit the VAR model
        model = VAR(var_params)
        results = model.fit(maxlags=maxlags, ic=ic)
        
        self.var_results = results
        return results
    
    
    def _generate_yield_curves(self, parameter_df):
        """
        Generate yield curves from parameters.
        
        Parameters:
        parameter_df (DataFrame): DataFrame containing L, S, C, and lambda parameters.
        
        Returns:
        DataFrame: Yield curves for each date in the parameter DataFrame.
        """
        if self.maturities is None:
            raise ValueError("Maturities not defined. Please estimate parameters first.")
        
        yield_curves = pd.DataFrame(index=parameter_df.index, columns=self.maturities)
        
        for date in parameter_df.index:
            L = parameter_df.loc[date, 'L']
            S = parameter_df.loc[date, 'S']
            C = parameter_df.loc[date, 'C']
            
            if 'lambda' in parameter_df.columns:
                lambda_ = parameter_df.loc[date, 'lambda']
            else:
                lambda_ = self.lambda_value  # Use fixed lambda if not in DataFrame
            
            yield_curves.loc[date] = self._nelson_siegel_function(
                self.maturities, L, S, C, lambda_
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
    
    
    def predict(self, steps=10, conf_int=0.95, return_param_estimates=False):
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
        
        # Prepare for forecasting
        if self.fix_lambda:
            # Get only beta parameters for forecasting
            current_values = self.params[['L', 'S', 'C']].values
        else:
            current_values = self.params.values
        
        # Generate forecasts
        forecast = self.var_results.forecast(current_values, steps=steps)
        forecast_variance = self.var_results.forecast_cov(steps=steps)
        
        # Create DataFrame for forecasts
        freq = pd.infer_freq(self.params.index)
        start_date = self.params.index[-1] + pd.tseries.frequencies.to_offset(freq)
        forecast_index = pd.date_range(start=start_date, periods=steps, freq=freq)
        
        if self.fix_lambda:
            # Create forecast DataFrame with only beta parameters
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['L', 'S', 'C'])
            # Add lambda column with fixed value
            forecast_df['lambda'] = self.lambda_value
        else:
            forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=self.params.columns)
        
        # Calculate confidence intervals (if statsmodels version supports it)
        try:
            forecast_intervals = self.var_results.forecast_interval(current_values, steps, alpha=(1-conf_int))
            
            if self.fix_lambda:
                # Handle fixed lambda case
                lower_bound = pd.DataFrame(forecast_intervals[0], index=forecast_index, columns=['L', 'S', 'C'])
                upper_bound = pd.DataFrame(forecast_intervals[1], index=forecast_index, columns=['L', 'S', 'C'])
                # Add lambda columns
                lower_bound['lambda'] = self.lambda_value
                upper_bound['lambda'] = self.lambda_value
            else:
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