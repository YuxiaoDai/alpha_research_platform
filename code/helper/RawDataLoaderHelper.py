import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from zipfile import ZipFile
from utils.PathManager import PathManager

class RawDataLoaderHelper:

    def __init__(self):
        self.path_manager = PathManager()

    def daily_stock_data_loader(self, tab_name, start, end, fields = None):
        """
        Loads daily stock data for given date range and fields.
        Including adj_fct, idx, lmt, mkt_val, sw
        
        :param tab_name: Name of the dataset to load
        :param start: Start date placeholder 
        :param end: End date placeholder
        :param fields: List of fields to retrieve (e.g., ['code', 'cum_adj'])
        """

        # Load path from path manager
        path = self.path_manager.get(tab_name)
        print(f"Loading data from {path}")

        # Convert strings to datetime objects
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')

        dfs = []
        # Iterate over years within the date range
        for year in range(start_date.year, end_date.year + 1):
            year_folder = os.path.join(path, str(year))
            
            if os.path.exists(year_folder): 
                for file_name in os.listdir(year_folder):
                    file_path = os.path.join(year_folder, file_name)
                    # Extract the date from the file name (assuming it's in YYYYMMDD format)
                    file_date = datetime.strptime(file_name.split('.')[0], '%Y%m%d').date()
                    if start_date.date() <= file_date <= end_date.date():
                        df = pd.read_csv(file_path, usecols=fields)
                        if 'date' not in df.columns:  # TODO: Some datasets does  not contain date column. How to add it when it is not avail
                            df.loc[:,'date'] = file_name.split('.')[0]
                        dfs.append(df)
                        
        # Concatenate all data frames into one
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df
    
    def date_data_loader(self, tab_name, start = None, end = None, fields = None):
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
        path = os.path.join(root_path, f'{tab_name}.csv')
        print(f"Loading data from {path}")

        df = pd.read_csv(path, usecols=fields)
        return df

    def min_pv_loader(self, tab_name, start, end, fields = None):
        """
        Loads 1 minute level data for all stocks for a given date range.
        
        :param start: Start date in 'YYYY-MM-DD' format.
        :param end: End date in 'YYYY-MM-DD' format.
        :param fields: List of fields to retrieve.
        :return: DataFrame containing the data for all stocks within the date range.
        """
        # Load path from path manager
        path = self.path_manager.get(tab_name)
        print(f"Loading data from {path}")
        
        start_date = datetime.strptime(start, '%Y-%m-%d')
        end_date = datetime.strptime(end, '%Y-%m-%d')
        
        all_data = []  # List to hold all data frames

        # Loop through each day in the date range
        for single_date in pd.date_range(start_date, end_date):
            date_str = single_date.strftime('%Y%m%d')
            day_folder = path / f'{date_str}.zip'

            # Check if day folder exists
            if not day_folder.exists():
                continue  # Skip if folder for the current day doesn't exist

            with ZipFile(day_folder, 'r') as zip_files:
                for filename in zip_files.namelist():
                    with zip_files.open(filename) as file:
                        df = pd.read_csv(file, usecols=fields)
                        all_data.append(df)
        # Concatenate all data frames into a single data frame
        return pd.concat(all_data, ignore_index=True)

    
    def get_available_dates_daily_stock_data(self, tab_name, date, window):
        """
        Given a start date, finds the specified number of available trading dates before that date within the given tab_name folder.

        :param tab_name: The folder name where the data is stored (e.g., 'adj_fct').
        :param date: The starting date in 'YYYY-MM-DD' format.
        :param window: The number of available dates to look back for.
        :param fields: Optional list of fields to keep in the DataFrame.
        :return: A list of  date objects for the available dates.
        """
        # Load path from path manager
        data_folder = self.path_manager.get(tab_name)
        print(f"Searching for available dates in {data_folder}")
        
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        available_dates = []
        data_frames = []

        # Iterate through the data folder to find available dates
        for year_folder in sorted(data_folder.iterdir(), reverse=True):
            if year_folder.is_dir() and len(available_dates) < window:
                for file in sorted(year_folder.iterdir(), reverse=True):
                    file_date_str = file.stem
                    file_date_obj = datetime.strptime(file_date_str, '%Y%m%d').date()
                    # Only consider dates before the start date
                    if file_date_obj < date_obj.date():
                        available_dates.append(file_date_obj)
                        if len(available_dates) == window:
                            break
        return available_dates

    def get_available_dates_1min_pv(self, tab_name, date, window):
        """
        Given a start date, finds all available trading dates within a specified window before that date.

        :param start_date: The starting date in 'YYYY-MM-DD' format.
        :param window: The number of available dates to look back for.
        :return: A list of  date objects for the available dates.
        """

        # Load path from path manager
        path = self.path_manager.get(tab_name)
        print(f"Searching for available dates in {path}")

        available_dates = []
        current_date = datetime.strptime(date, '%Y-%m-%d')
        days_looked_back = 0
        
        # Keep checking previous days until we find the required number of available dates
        while len(available_dates) < window:
            days_looked_back += 1
            check_date = current_date - timedelta(days=days_looked_back)
            check_date_str = datetime.strftime(check_date, '%Y%m%d')
            day_folder = path / check_date_str
            if day_folder.exists():
                available_dates.append(check_date)

        return available_dates
    