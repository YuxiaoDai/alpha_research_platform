
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path

from utils.PathManager import PathManager
from helper.ProcessedDataLoaderHelper import ProcessedDataLoaderHelper

class ProcessedDataLoader:

    def __init__(self):
        self.processed_data_loader_helper = ProcessedDataLoaderHelper()
    
    def loading(self, tab_name, start, end, fields = None):
        # Check if tab_name is in the map and call the corresponding function
        if tab_name in ['halt_date_processed', 'lst_date_processed', 'st_date_processed', 'trd_date_processed']:
            return self.processed_data_loader_helper.processed_date_data_loader(tab_name, start, end, fields)
        else:
            return self.processed_data_loader_helper.processed_data_loader(tab_name, start, end, fields)
        
    # Loading function for a specific date with a window
    def loading_with_window(self, tab_name, date, window, fields = None):
        # Calculate start and end dates based on the window
        end = date 
        available_dates = self.processed_data_loader_helper.get_available_dates(tab_name, date, window)
        start = datetime.strftime(min(available_dates), '%Y-%m-%d')
        print(f'Loading {tab_name} data from {start} to {end}')
        # Call the general loading function with the calculated date range
        return self.loading(tab_name, start, end, fields)
