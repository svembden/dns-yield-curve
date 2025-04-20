import pandas as pd

def load_yield_curve_data(url='https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'):
    """
    Load the US Yield Curve data from the specified URL and preprocess it.
    
    Parameters:
    url (str): The URL to load the data from.
    
    Returns:
    pd.DataFrame: A DataFrame containing the yield curve data.
    """
    df = pd.read_csv(url, sep=';', index_col=0)
    # Additional preprocessing steps can be added here if necessary
    return df