import pandas as pd
import numpy as np

def load_data(url):
    """
    Load the US Yield Curve data from the specified URL.
    
    Parameters:
    url (str): The URL to load the data from.
    
    Returns:
    DataFrame: A pandas DataFrame containing the yield curve data.
    """
    df = pd.read_csv(url, sep=';', index_col=0)
    return df

def preprocess_data(df):
    """
    Preprocess the yield curve data.
    
    Parameters:
    df (DataFrame): The DataFrame containing the yield curve data.
    
    Returns:
    DataFrame: A preprocessed DataFrame.
    """
    # Example preprocessing steps
    df.index = pd.to_datetime(df.index)
    df = df.dropna()
    return df

def split_data(df, train_size=0.8):
    """
    Split the data into training and testing sets.
    
    Parameters:
    df (DataFrame): The DataFrame to split.
    train_size (float): The proportion of the data to include in the training set.
    
    Returns:
    tuple: A tuple containing the training and testing DataFrames.
    """
    train_size = int(len(df) * train_size)
    train, test = df[:train_size], df[train_size:]
    return train, test

def input_checks(dates, data, maturities):
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
            raise ValueError(f"Input data must be a pandas DataFrame or Series. Got {type(data)}.")
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

def nelson_siegel_function(tau, L, S, C, lam):
    """
    Nelson-Siegel yield function.
    
    Parameters:
    tau (float or array-like): Maturities.
    L (float): Long-term yield - Level.
    S (float): Short-term yield - Slope.
    C (float): Medium-term yield - Curvature.
    lam (float): Decay factor.
    
    Returns:
    float or array-like: The yield curve values.
    """
    tau = np.asarray(tau, dtype=float)
    ltau = lam * tau

    # Avoid division by zero or near-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = np.where(ltau > 1e-6, (1 - np.exp(-ltau)) / ltau, 1.0)
    term2 = term1 - np.exp(-ltau)

    return L + S * term1 + C * term2

def generate_yield_curves(params_df, maturities=None):
        """
        Generate yield curves from DNS parameters.
        
        Parameters:
        params_df (DataFrame): DNS parameters with columns ['L', 'S', 'C', 'lambda'].
        maturities (ndarray, optional): Maturities to use. If None, defaults to standard maturities.
        
        Returns:
        DataFrame: Generated yield curves.
        """
        if maturities is None:
            # Default maturities if none provided
            maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
        
        yields = np.zeros((len(params_df), len(maturities)))
        
        for i, (_, row) in enumerate(params_df.iterrows()):
            L, S, C, lam = row
            for j, tau in enumerate(maturities):
                yields[i, j] = nelson_siegel_function(tau, L, S, C, lam)
        
        yield_curves = pd.DataFrame(yields, index=params_df.index, 
                                columns=[f'tau_{tau:.4f}' for tau in maturities])
        
        return yield_curves