import unittest
import pandas as pd

from alpha.vol_5d import vol_5d

class TestVol5D(unittest.TestCase):
    def test_vol_5d(self):
        # Create a sample DataFrame
        data = {
            'code': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'],
            'date': pd.date_range(start='2023-01-01', periods=5).tolist() + pd.date_range(start='2023-01-01', periods=5).tolist(),
            'close_adj': [100, 102, 104, 98, 96, 200, 205, 207, 210, 215]
        }
        df = pd.DataFrame(data)

        # Call the function
        result_df = vol_5d(df)

        # Check the output
        expected_vol_values = [None, None, None, None, 0.042426, None, None, None, None, 0.037416]
        calculated_vol_values = result_df['vol_5d'].tolist()

        # Assert that the expected values are close to the calculated values
        for expected, actual in zip(expected_vol_values, calculated_vol_values):
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertAlmostEqual(expected, actual, places=5)

if __name__ == '__main__':
    unittest.main()
