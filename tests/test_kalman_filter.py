import unittest
import pandas as pd
from dns.models.kalman_filter import estimate_kalman_filter

class TestKalmanFilter(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the data
        url = 'https://www.dropbox.com/s/inpnlugzkddp42q/bonds.csv?dl=1'
        cls.df = pd.read_csv(url, sep=';', index_col=0)

    def test_estimate_kalman_filter(self):
        # Test the Kalman Filter estimation
        params = estimate_kalman_filter(self.df)
        self.assertIsNotNone(params)
        self.assertIsInstance(params, dict)
        self.assertIn('beta0', params)
        self.assertIn('beta1', params)
        self.assertIn('beta2', params)
        self.assertIn('beta3', params)
        self.assertIn('tau', params)

if __name__ == '__main__':
    unittest.main()