
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path
from utils.PathManager import PathManager

class ProcessedDataLoaderHelper:

    def __init__(self):
        self.path_manager = PathManager()

    
    def processed_data_loader(self, tab_name, start, end, fields = None):
        """
        Loads daily stock data for given date range and fields.
        Including adj_fct, idx, lmt, mkt_val, sw
        
        :param tab_name: Name of the dataset to load
        :param start: Start date placeholder 
        :param end: End date placeholder
        :param fields: List of fields to retrieve (e.g., ['code', 'cum_adj'])
        """

        # Load path from items table 
        items_path = self.path_manager.get('items')
        items = pd.read_csv(items_path)
        path_folder = items.loc[items['item'] == tab_name,'saved path'].iloc[0]
        path_folder = os.path.join(path_folder, tab_name)
        print(f"Loading data from {path_folder}")

        # Convert strings to datetime objects
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')

        dfs = []
        # Iterate over the folder 
        if os.path.exists(path_folder): 
            for file_name in os.listdir(path_folder):
                file_path = os.path.join(path_folder, file_name)
                # Extract the date from the file name (assuming it's in YYYYMMDD format)
                file_date = datetime.strptime(file_name.split('.')[0], '%Y%m%d').date()
                if start_date.date() <= file_date <= end_date.date():
                    df = pd.read_parquet(file_path)
                    dfs.append(df)
                        
        # Concatenate all data frames into one
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df
    
    def processed_date_data_loader(self, tab_name, start = None, end = None, fields = None):
        """
        Loads date data for given date range and fields.
        Including halt_date, lst_date, st_date, trd_date
        
        :param tab_name: Name of the dataset to load
        :param start: Start date in YYYY-MM-DD format
        :param end: End date in YYYY-MM-DD format
        :param fields: List of fields to retrieve (e.g., ['code', 'cum_adj'])
        """
        # Load path from path manager
        root_path = self.path_manager.get(tab_name)
        tab_name = tab_name[:-len('_processed')]
        path = os.path.join(root_path, tab_name, f'{tab_name}.parquet')
        print(f"Loading data from {path}")

        df = pd.read_parquet(path)
        return df
    
    def get_available_dates(self, tab_name, date, window, fields=None):
        """
        Given a start date, finds the specified number of available trading dates before that date within the given tab_name folder.

        :param tab_name: The folder name where the data is stored (e.g., 'adj_fct').
        :param date: The starting date in 'YYYY-MM-DD' format.
        :param window: The number of available dates to look back for.
        :param fields: Optional list of fields to keep in the DataFrame.
        :return: A list of  date objects for the available dates.
        """

        # Load path from items table 
        items_path = self.path_manager.get('items')
        items = pd.read_csv(items_path)
        data_folder = items.loc[items['item'] == tab_name,'saved path'].values[0]
        data_folder = os.path.join(data_folder, tab_name)
        data_folder = Path(data_folder)
        print(f"Searching for available dates in {data_folder}")
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        available_dates = []
        data_frames = []

        # Iterate through the data folder to find available dates
        for file in sorted(data_folder.iterdir(), reverse=True):
            file_date_str = file.stem
            file_date_obj = datetime.strptime(file_date_str, '%Y%m%d').date()
            # Only consider dates before the start date
            if file_date_obj < date_obj.date():
                available_dates.append(file_date_obj)
                if len(available_dates) == window:
                    break
        return available_dates

    