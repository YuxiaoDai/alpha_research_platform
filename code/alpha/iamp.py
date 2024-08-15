import pandas as pd
import numpy as np
from alpha.helper_function import *

def iamp(input_data):

    """
    Calculating amplitude over time for individual stocks, considering the differences in amplitude distribution across various price ranges. 
    1. The amplitude, amp, is calculated as: amp = high / low - 1
    2. Over time, the average amplitudes where the closing price is in the top n% and the bottom n% are calculated respectively as:
        amp_high = mean(amp where close is higher)
        amp_low = mean(amp where close is lower)
    3. The alpha is defined as the difference between these averages:
        alpha = amp_high - amp_low

    """ 

    def calculate_iamp(close_adj, percent):

        amp = input_data.loc[close_adj.index, 'amp']
        df = pd.DataFrame({'amp': amp, 'close_adj': close_adj})

        df = df.sort_values(by='close_adj')
        n = len(df)
        low_threshold = int(n * percent)
        high_threshold = int(n * (1 - percent))
            
        low_amps = df.iloc[:low_threshold]['amp']
        high_amps = df.iloc[high_threshold:]['amp']
            
        low_amps_mean = low_amps.mean()
        high_amps_mean = high_amps.mean()
        alpha = high_amps_mean - low_amps_mean
            
        return alpha
    
    # Calculate amplitude, amp = high / low - 1, over time for individual stocks;
    input_data['amp'] = input_data['high_adj'] / input_data['low_adj'] - 1

    # Set window length and percentage;
    window = 20
    percent = 0.3

    # Sort by stock code and date.
    input_data.sort_values(by=['code', 'date'], inplace=True)

    # dropna
    input_data.dropna(subset=['close_adj', 'amp'])

    # calculate alpha
    result = input_data.groupby('code').rolling(window = window, min_periods = window)['close_adj'].apply(lambda x: calculate_iamp(x, percent = percent), raw=False).reset_index(drop = True)
    result.index = input_data.index
    input_data['iamp']= - result

    # prepare the result
    input_data = input_data[['code', 'date', 'iamp']]
    input_data.dropna(subset='iamp', inplace=False)

    # standardize alpha score
    input_data_standardized = rank_and_demedian(input_data, 'iamp')

    return input_data[['code', 'date', 'iamp']], input_data_standardized[['code', 'date', 'iamp']]
