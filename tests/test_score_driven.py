import pandas as pd
import numpy as np
import pytest
from src.dns.models.score_driven import ScoreDrivenModel

# Load the data
url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
df = pd.read_csv(url, sep=';', index_col=0)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

def test_score_driven_model():
    # Initialize the Score Driven model
    model = ScoreDrivenModel(train)

    # Fit the model
    model.fit()

    # Forecast
    forecast = model.forecast(steps=len(test))

    # Check if the forecast length matches the test set length
    assert len(forecast) == len(test)

    # Check if the forecast is not empty
    assert not forecast.empty

    # Optionally, check if the forecast values are within a reasonable range
    assert np.all(forecast >= 0)  # Assuming yields cannot be negative

if __name__ == "__main__":
    pytest.main()