import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zipfile import ZipFile

from utils.PathManager import PathManager
from helper.RawDataLoaderHelper import RawDataLoaderHelper

class RawDataLoader:

    def __init__(self):
        
        self.raw_data_loader_helper = RawDataLoaderHelper()

        # This map will connect tab names with the corresponding data loading functions
        self.tab_name_map = {
            # raw data
            "adj_fct": [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "idx":     [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "lmt":     [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "mkt_val": [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "sw":      [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "hs300":   [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "zz500":   [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "zz800":   [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "zz1000":  [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "zz9999":  [self.raw_data_loader_helper.daily_stock_data_loader, self.raw_data_loader_helper.get_available_dates_daily_stock_data],
            "halt_date": [self.raw_data_loader_helper.date_data_loader],
            "lst_date":  [self.raw_data_loader_helper.date_data_loader],
            "st_date":   [self.raw_data_loader_helper.date_data_loader],
            "trd_date":  [self.raw_data_loader_helper.date_data_loader],
            "1min_pv":   [self.raw_data_loader_helper.min_pv_loader, self.raw_data_loader_helper.get_available_dates_1min_pv],
            
        }
        
    def loading(self, tab_name, start, end, fields = None):
        # Check if tab_name is in the map and call the corresponding function
        if tab_name in self.tab_name_map:
            return self.tab_name_map[tab_name][0](tab_name, start, end, fields)
        else:
            raise ValueError(f"Unknown table name: {tab_name}")

    # Loading function for a specific date with a window
    def loading_with_window(self, tab_name, date, window, fields = None):
        # Calculate start and end dates based on the window
        end = date 
        available_dates = self.tab_name_map[tab_name][1](tab_name, date, window)
        start = datetime.strftime(min(available_dates), '%Y-%m-%d')
        print(f'Loading {tab_name} data from {start} to {end}')
        # Call the general loading function with the calculated date range
        return self.loading(tab_name, start, end, fields)
