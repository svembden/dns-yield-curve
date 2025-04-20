import pandas as pd
import numpy as np
from dns.utils.logging import setup_logger
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR

logger = setup_logger(__name__, log_file="../logs/dns_model.log")

# cd "C:\Users\semva\OneDrive\Documenten\Code Projects\DNS_py\dns-yield-curve\src" 
# python -m dns.models.cross_sectional_VAR

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


def estimate_cross_sectional_parameters(dates, data, maturities):
    """
    Estimate cross-sectional DNS parameters (βs) at each point in time.
    
    Parameters:
    dates (datetime-like): The dates for which to estimate parameters.
    data (DataFrame): The input data containing yield curve information.
    maturities (array-like): The maturities corresponding to the columns of the data.
    
    Returns:
    DataFrame: A DataFrame containing the estimated parameters.
    """
    
    # Input Checks
    if not isinstance(dates, (pd.DatetimeIndex, pd.Series)): # check if dates is a DatetimeIndex or Series
        raise ValueError("Input dates must be a pandas DatetimeIndex or Series.")
    if not isinstance(data, (pd.DataFrame, pd.Series)): # check if data is a DataFrame or Series
        raise ValueError("Input data must be a pandas DataFrame or Series.")
    if data.empty: # check if data is empty
        raise ValueError("Input data is empty.")
    if data.isnull().values.any(): # check if data contains NaN values
        raise ValueError("Input data contains NaN values.")
    if not isinstance(maturities, (list, np.ndarray)): # check if maturities is a list or numpy array
        raise ValueError("Maturities must be provided as a list or numpy array.")
    if len(maturities) != data.shape[1]: # check if length of maturities matches number of columns in data
        raise ValueError("Length of maturities must match the number of columns in data.")
    if not len(dates) == len(data): # check if length of dates matches number of rows in data
        raise ValueError("Length of dates must match the number of rows in data.")
    
    # Initialize nelson-siegel parameters
    params = pd.DataFrame(index=dates, columns=['beta0', 'beta1', 'beta2', 'lambda'])
    params = params.fillna(0).infer_objects(copy=False)
    params = params.astype(float)
    
    mean_yields = data.mean(axis=0)
    tau = np.array(maturities)
    
    # Initial parameter guesses
    initial_params = [
        mean_yields.iloc[-1], # beta0
        mean_yields.iloc[0] - mean_yields.iloc[-1], # beta1
        1.0, # beta2
        0.4 # lambda
    ]
    
    logger.debug(f"Initial parameter guesses:\n{initial_params}")

    # Bounds for parameters
    bounds = [
        (0, None),  # beta0
        (None, None),  # beta1
        (0, None),  # beta2
        (1e-8, None)   # lambda
    ]
    
    # Objective function to minimize
    def objective_function(params, maturities, observed_yields):
        beta0, beta1, beta2, lambda_ = params
        model_yields = nelson_siegel_function(maturities, beta0, beta1, beta2, lambda_)
        return np.sum((observed_yields - model_yields) ** 2)
    
    logger.info("Starting parameter estimation...")
    
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
    
    return params


def fit_var_model(params):
    """
    Fit a VAR model to the estimated parameters.
    
    Parameters:
    params (DataFrame): The estimated parameters DataFrame.
    
    Returns:
    VARResults: The fitted VAR model results.
    """
    logger.info("Fitting VAR model...")
    
    # Fit the VAR model
    model = VAR(params)
    results = model.fit(maxlags=5, ic='aic')
    
    return results





def main():
    # Load the data
    url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
    df = pd.read_csv(url, sep=';', index_col=0)
    
    maturities = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120])
    

    # Estimate parameters
    dates = pd.to_datetime(df.index, format="%Y:%m")
    params = estimate_cross_sectional_parameters(dates, df, maturities)
    
    logger.info("Estimated Parameters:")
    logger.info(params.head())
    
    # Fit VAR model
    var_results = fit_var_model(params)
    logger.info("VAR Model Results:")
    logger.info(var_results.summary())
    

if __name__ == "__main__":
    main()