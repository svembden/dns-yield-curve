import pandas as pd
import numpy as np
from dns.utils.logging import setup_logger
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR

logger = setup_logger(__name__, log_file="../logs/dns_model.log")

class CSVAR:
    """Cross-Sectional Vector Autoregression model for yield curve analysis."""
    
    def __init__(self):
        """Initialize CSVAR model."""
        self.logger = logger
        self.params = None
        self.var_results = None
        self.maturities = None
    
    @staticmethod
    def nelson_siegel_function(tau, beta0, beta1, beta2, lambda_):
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
    
    def estimate_cross_sectional_parameters(self, dates, data, maturities):
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
        self.maturities = np.array(maturities)
        
        # Input Checks
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
        
        # Initialize nelson-siegel parameters
        params = pd.DataFrame(index=dates, columns=['beta0', 'beta1', 'beta2', 'lambda'])
        params = params.fillna(0).infer_objects(copy=False)
        params = params.astype(float)
        
        mean_yields = data.mean(axis=0)
        tau = np.array(maturities)
        
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
            model_yields = self.nelson_siegel_function(maturities, beta0, beta1, beta2, lambda_)
            return np.sum((observed_yields - model_yields) ** 2)
        
        self.logger.info("Starting parameter estimation...")
        
        # Estimate parameters for each date
        for i, date in enumerate(data.index):
            observed_yields = data.loc[date].values
            
            # Optimize to find the best parameters
            result = minimize(
                objective_function,
                initial_params,
                args=(maturities, observed_yields),
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
    
    def fit_var_model(self, params=None, maxlags=5, ic='aic'):
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
    
    def forecast(self, steps=10, conf_int=0.95):
        """
        Forecast future values of the yield curve parameters.
        
        Parameters:
        steps (int): Number of steps ahead to forecast. Default is 10.
        conf_int (float): Confidence interval for the forecast. Default is 0.95.
        
        Returns:
        tuple: (forecast_df, forecast_variance, forecast_intervals)
            - forecast_df: DataFrame with point forecasts
            - forecast_variance: variance of the forecasts
            - forecast_intervals: confidence intervals for forecasts
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
        
        return forecast_df, forecast_variance, forecast_intervals
    
    def generate_yield_curves(self, parameter_df):
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
            
            yield_curves.loc[date] = self.nelson_siegel_function(
                self.maturities, beta0, beta1, beta2, lambda_
            )
        
        return yield_curves


def main():
    """Main function to run the CSVAR model."""
    # Load the data
    url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
    df = pd.read_csv(url, sep=';', index_col=0)
    
    maturities = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120])
    dates = pd.to_datetime(df.index, format="%Y:%m")
    
    # Create and use the CSVAR model
    model = CSVAR()
    
    # Estimate parameters
    params = model.estimate_cross_sectional_parameters(dates, df, maturities)
    logger.info("Estimated Parameters:")
    logger.info(params.head())
    
    # Fit VAR model
    var_results = model.fit_var_model()
    logger.info("VAR Model Results:")
    logger.info(var_results.summary())
    
    # Forecast future parameters
    forecast_horizon = 12  # 12 months ahead
    forecast_df, _, forecast_intervals = model.forecast(steps=forecast_horizon)
    logger.info("Parameter Forecasts (next 12 months):")
    logger.info(forecast_df.head())
    
    # Generate forecasted yield curves
    forecasted_yields = model.generate_yield_curves(forecast_df)
    logger.info("Forecasted Yield Curves:")
    logger.info(forecasted_yields.head())

if __name__ == "__main__":
    main()