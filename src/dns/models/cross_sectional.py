import pandas as pd
import numpy as np
from dns.utils.helpers import nelson_siegel_function
from scipy.optimize import minimize

def estimate_cross_sectional_parameters(data):
    """
    Estimate cross-sectional DNS parameters (βs) at each point in time.
    
    Parameters:
    data (DataFrame): The input data containing yield curve information.
    
    Returns:
    DataFrame: A DataFrame containing the estimated parameters.
    """
    
    # Input Checks
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    if data.empty:
        raise ValueError("Input data is empty.")
    if data.isnull().values.any():
        raise ValueError("Input data contains NaN values.")
    
    # Initialize nelson-siegel parameters
    params = pd.DataFrame(index=data.index, columns=['beta0', 'beta1', 'beta2', 'tau'])
    params = params.fillna(0)
    params = params.astype(float)
    
    # Estimate parameters

def estimate_cross_sectional_parameters(data, maturities):
    """
    Estimate cross-sectional DNS parameters (βs) at each point in time.
    
    Parameters:
    data (DataFrame): The input data containing yield curve information.
    
    Returns:
    DataFrame: A DataFrame containing the estimated parameters.
    """
    
    # Input Checks
    if not isinstance(data, pd.DataFrame): # check if data is a DataFrame
        raise ValueError("Input data must be a pandas DataFrame.")
    if data.empty: # check if data is empty
        raise ValueError("Input data is empty.")
    if data.isnull().values.any(): # check if data contains NaN values
        raise ValueError("Input data contains NaN values.")
    if not isinstance(maturities, (list, np.ndarray)): # check if maturities is a list or numpy array
        raise ValueError("Maturities must be provided as a list or numpy array.")
    if len(maturities) != data.shape[1]: # check if length of maturities matches number of columns in data
        raise ValueError("Length of maturities must match the number of columns in data.")
    
    # Initialize nelson-siegel parameters
    params = pd.DataFrame(index=data.index, columns=['beta0', 'beta1', 'beta2', 'lambda'])
    params = params.fillna(0)
    params = params.astype(float)
    

    mean_yields = data.mean(axis=0)
    tau = np.array(maturities)
    
    # Initial parameter guesses
    initial_params = [
        mean_yields[-1], # beta0
        mean_yields.iloc[0] - mean_yields.iloc[-1], # beta1
        1.0, # beta2
        0.4 # lambda
    ]
    
    print("Initial Parameters:")
    print(initial_params)
    
    
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
    
    # Estimate parameters for each date
    for date in data.index:
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
        params.loc[date, 'beta0'] = result.x[0]
        params.loc[date, 'beta1'] = result.x[1]
        params.loc[date, 'beta2'] = result.x[2]
        params.loc[date, 'lambda'] = result.x[3]
    
    return params


def main():
    # Load the data
    url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
    df = pd.read_csv(url, sep=';', index_col=0)
    
    maturities = np.array([3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120])

    # Estimate parameters
    params = estimate_cross_sectional_parameters(df, maturities)
    
    print("Estimated Parameters:")
    print(params.head())

if __name__ == "__main__":
    main()