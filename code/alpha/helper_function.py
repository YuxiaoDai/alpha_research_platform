import pandas as pd
import numpy as np

def compute_adv(input_data, k):

    """
    Compute the average daily volume in k days
    :param input_data: Name of the dataset to load, containing: code, date, volume
    :param k: window
    """
    return input_data.groupby('code').rolling(window = k)['accvolume_adj'].mean().reset_index(0, drop=True)


def scale(input_data):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return input_data.div(input_data.abs().sum(), axis=0)


def rank_and_demedian(input_data, alpha_name):

    # copy
    input_data_standardized = input_data.copy()

    # rank
    input_data_standardized['rank'] = input_data_standardized.groupby(['date'])[alpha_name].rank()

    # demedian
    input_data_standardized['median'] = input_data_standardized.groupby(['date'])['rank'].transform('median')
    input_data_standardized[alpha_name] = input_data_standardized['rank'] - input_data_standardized['median']
    
    return input_data_standardized[['code', 'date', alpha_name]]


def calc_zscore(input_data, alpha_name):
    
    # copy
    input_data_standardized = input_data.copy()

    # mean
    input_data_standardized['mean'] = input_data_standardized.groupby(['date'])[alpha_name].transform('mean')

    # std
    input_data_standardized['std'] = input_data_standardized.groupby(['date'])[alpha_name].transform('std')

    # zscore
    input_data_standardized[alpha_name] = (input_data_standardized[alpha_name] - input_data_standardized['mean'])/input_data_standardized['std']

    return input_data_standardized[['code', 'date', alpha_name]]
