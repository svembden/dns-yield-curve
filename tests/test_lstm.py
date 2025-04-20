#@TODO IMPLEMENT:

import pandas as pd
import numpy as np
import unittest
from dnss.models.lstm import LSTMModel  # Assuming LSTMModel is the class to be tested

class TestLSTMModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
        cls.df = pd.read_csv(url, sep=';', index_col=0)
        cls.model = LSTMModel()  # Initialize your LSTM model here

    def test_model_training(self):
        # Split the data into train and test sets
        train_size = int(len(self.df) * 0.8)
        train, test = self.df[:train_size], self.df[train_size:]

        # Train the model
        self.model.fit(train)

        # Check if the model has been trained (you can add more specific checks)
        self.assertIsNotNone(self.model.weights)

    def test_model_forecasting(self):
        # Forecast using the trained model
        forecast = self.model.forecast(steps=len(self.df) - int(len(self.df) * 0.8))

        # Check if the forecast is of the expected shape
        self.assertEqual(forecast.shape[0], len(self.df) - int(len(self.df) * 0.8))

if __name__ == '__main__':
    unittest.main()