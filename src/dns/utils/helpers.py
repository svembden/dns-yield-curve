import pandas as pd

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