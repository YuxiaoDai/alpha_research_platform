import pandas as pd
import numpy as np
from alpha.helper_function import *

def vol_5d(input_data):

    """
    calculate 5-day volatility: the standard deviation of daily return in the past 5 days (including today)
    """

    # Sort by stock code and date.
    input_data.sort_values(by=['code', 'date'], inplace=True)

    # calculate daily return
    input_data['daily_return'] = input_data.groupby('code')['close_adj'].pct_change()

    # calculate past 5 day volatility
    input_data['vol_5d'] = - input_data.groupby('code')['daily_return'].rolling(window=5).std().reset_index(level=0, drop=True)

    # standardize alpha score
    input_data_standardized = rank_and_demedian(input_data, 'vol_5d')

    return input_data[['code', 'date', 'vol_5d']], input_data_standardized[['code', 'date', 'vol_5d']]

