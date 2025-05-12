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