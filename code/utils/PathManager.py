
from pathlib import Path
import os

class PathManager:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(PathManager, cls).__new__(cls)
            cls._instance.base_data_path = Path("")
        return cls._instance
 
    def __init__(self):
        # Initialize data paths that depend on base_data_path
        self.update_paths()

    def update_paths(self):
        # This method updates paths based on the current base_data_path
        self.data_paths = {

            # Raw data
            "adj_fct":    self.base_data_path / "Raw data" / "other" / "adj_fct",
            "idx":        self.base_data_path / "Raw data" / "other" / "idx",
            "lmt":        self.base_data_path / "Raw data" / "other" / "lmt",
            "mkt_val":    self.base_data_path / "Raw data" / "other" / "mkt_val",
            "sw":         self.base_data_path / "Raw data" / "other" / "sw",
            "hs300":      self.base_data_path / "Raw data" / "other" / "univ" / "hs300",
            "zz500":      self.base_data_path / "Raw data" / "other" / "univ" / "zz500",
            "zz800":      self.base_data_path / "Raw data" / "other" / "univ" / "zz800",
            "zz1000":     self.base_data_path / "Raw data" / "other" / "univ" / "zz1000",
            "zz9999":     self.base_data_path / "Raw data" / "other" / "univ" / "zz9999",
            "halt_date":  self.base_data_path / "Raw data" / "other" / "date",
            "lst_date":   self.base_data_path / "Raw data" / "other" / "date",
            "st_date":    self.base_data_path / "Raw data" / "other" / "date",
            "trd_date":   self.base_data_path / "Raw data" / "other" / "date",
            "1min_pv":    self.base_data_path / "Raw data" / "qishi_1min",
            
            # Processed data
            "adj_fct_processed": self.base_data_path / "Processed data" / "stock_data_daily" ,
            "idx_processed":     self.base_data_path / "Processed data" / "index_data" ,
            "lmt_processed":     self.base_data_path / "Processed data" / "stock_data_daily",
            "mkt_val_processed": self.base_data_path / "Processed data" / "stock_data_daily" ,
            "sw_processed":      self.base_data_path / "Processed data" / "stock_data_daily" ,
            "hs300_processed":   self.base_data_path / "Processed data" / "stock_data_daily" ,
            "zz500_processed":   self.base_data_path / "Processed data" / "stock_data_daily" ,
            "zz800_processed":   self.base_data_path / "Processed data" / "stock_data_daily" ,
            "zz1000_processed":  self.base_data_path / "Processed data" / "stock_data_daily" ,
            "zz9999_processed":  self.base_data_path / "Processed data" / "stock_data_daily" ,
            "halt_date_processed":  self.base_data_path / "Processed data" / "date_data" ,
            "lst_date_processed":   self.base_data_path / "Processed data" / "date_data" ,
            "st_date_processed":    self.base_data_path / "Processed data" / "date_data" ,
            "trd_date_processed":   self.base_data_path / "Processed data" / "date_data" ,
            "1min_pv_processed":  self.base_data_path / "Processed data" / "stock_data_minute",
            "1min_pv_aggregated": self.base_data_path / "Processed data" / "stock_data_daily" ,

            # item table
            "items":  self.base_data_path / 'items.csv',

            # alpha configuration yaml file
            "alpha_calc_config":  self.base_data_path /"alpha_config"/ "alpha_config.yaml",

            # alpha score and alpha portfolio
            "raw_alpha_score":  self.base_data_path / "alpha" / "Raw_Alpha_Score", # raw alpha score
            "alpha_score":      self.base_data_path / "alpha" / "Alpha_Score",     # standardized alpha score
            "alpha_portfolio":  self.base_data_path / "alpha" / "Alpha_Portfolio", # alpha portfolio constructed from alpha score

        }

    def set_base_data_path(self, new_path):
        # Method to update the base_data_path and all dependent paths
        self.base_data_path = Path(new_path)
        self.update_paths()

    def add_data_path(self, key, relative_path):
        """
        Adds a new data path entry to the data_paths dictionary.
        :param key: The key name for the data, such as 'new_data'.
        :param relative_path: The relative path from the base_data_path.
        """
        if key not in self.data_paths:
            self.data_paths[key] = self.base_data_path / Path(relative_path)
        else:
            raise KeyError(f"Key '{key}' already exists in data_paths.")

    def get(self, data_key):
        """
        Returns the full path based on the data key name.
        :param data_key: The data key name, such as 'adj_fct' or 'lmt'.
        :return: The Path object corresponding to the data.
        """
        if data_key in self.data_paths:
            return self.data_paths[data_key]
        else:
            raise ValueError(f"Path for {data_key} not found.")

    def check_path_exists(self, path):
        """
        Checks if the specified path exists.
        :param path: The path to check.
        """
        if not os.path.exists(path):
            raise ValueError(f'Path does not exist. Path is {path}')
        else:
            print(f'Path exists. {path}')
