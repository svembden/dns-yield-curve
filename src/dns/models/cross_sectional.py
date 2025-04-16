def estimate_cross_sectional_parameters(data):
    """
    Estimate cross-sectional DNS parameters (βs) at each point in time
    and model their dynamics using a VAR model.
    
    Parameters:
    data (DataFrame): The input data containing yield curve information.
    
    Returns:
    var_model: Fitted VAR model object.
    """
    import pandas as pd
    from statsmodels.tsa.api import VAR

    # Ensure the data is in the correct format
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    # Fit the VAR model
    model = VAR(data)
    var_model = model.fit()

    return var_model

def main():
    # Load the data
    url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
    df = pd.read_csv(url, sep=';', index_col=0)

    # Estimate parameters
    var_model = estimate_cross_sectional_parameters(df)
    print(var_model.summary())

if __name__ == "__main__":
    main()