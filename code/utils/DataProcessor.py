import pandas as pd
import numpy as np
import os

from utils.PathManager import PathManager
from utils.RawDataLoader import RawDataLoader
from helper.DataProcessorHelper import DataProcessorHelper


class DataProcessor:
    def __init__(self):
        self.data_loader = RawDataLoader()
        self.path_manager = PathManager()
        self.data_processor_helper = DataProcessorHelper()
        self.process_map = {
            'adj_fct': self.data_processor_helper.adj_fct_processor,
            'lmt':     self.data_processor_helper.lmt_processor,
            'mkt_val': self.data_processor_helper.mkt_val_processor,
            'sw':      self.data_processor_helper.sw_processor,
            'hs300':   self.data_processor_helper.univ_processor,
            'zz500':   self.data_processor_helper.univ_processor,
            'zz800':   self.data_processor_helper.univ_processor,
            'zz1000':  self.data_processor_helper.univ_processor,
            'zz9999':  self.data_processor_helper.univ_processor,
            'idx':     self.data_processor_helper.idx_processor,
            '1min_pv': self.data_processor_helper.min_pv_processor,
            'halt_date': self.data_processor_helper.date_processor,
            'lst_date':  self.data_processor_helper.date_processor,
            'st_date':   self.data_processor_helper.date_processor,
            'trd_date':  self.data_processor_helper.date_processor,
        }

    def load_and_process(self, tab_name, start, end, fields=None):
        print(f'Loading data.')
        data = self.data_loader.loading(tab_name, start, end, fields)

        print(f'Processing data.')
        processed_data = self.process_map[tab_name](data, tab_name)

        print('Saving data')
        self.save_data(processed_data, tab_name, f'{tab_name}_processed')

        return processed_data

    def load_and_aggregate(self, tab_name, start, end, fields=None, func = ['mean', 'std']):
        """
        Aggregate the minute level data to daily data. Only supposed to use for 1min_pv data
        """
        print(f'Loading data.')
        data = self.data_loader.loading(tab_name, start, end, fields)

        print(f'Processing data.')
        processed_data = self.process_map[tab_name](data, tab_name)

        print(f'Aggregating minute-level data to daily data.')
        aggregated_data = self.data_processor_helper.min_pv_aggregator(processed_data, func)
        
        print('Saving aggregated data')
        self.save_data(aggregated_data, tab_name, f'{tab_name}_aggregated')

        return aggregated_data

    def save_data(self, dataframes_dict, tab_name, tab_name_processed):
        
        base_path = self.path_manager.get(tab_name_processed)

        for item, data in dataframes_dict.items():
            item_path = os.path.join(base_path, item)
            os.makedirs(item_path, exist_ok=True)
            
            if tab_name in ['halt_date', 'lst_date', 'st_date', 'trd_date']:
                file_path = os.path.join(item_path, f"{item}.parquet")
                data.to_parquet(file_path) 
            else:
                for date, group in data.groupby('date'):
                    date_str = pd.to_datetime(date, format='%Y%m%d').strftime('%Y%m%d')
                    file_path = os.path.join(item_path, f"{date_str}.parquet")
                    group.to_parquet(file_path) 

            self.update_items_table(tab_name, tab_name_processed, item)

    def update_items_table(self, tab_name, tab_name_processed, item):
        
        items_info = pd.DataFrame({
            'item': [item],
            'description': ['desciption'],
            'source data': [tab_name],
            'source path': [self.path_manager.get(f'{tab_name}')],
            'saved path':  [self.path_manager.get(f'{tab_name_processed}')]
        })
        
        items_file = self.path_manager.get('items')
        if os.path.exists(items_file):

            # read items table
            items_table = pd.read_csv(items_file)

            # if the item already exists in items table, drop it 
            items_table = items_table[items_table['item'] != item]

            # add new item
            items_table = pd.concat([items_table, items_info]).drop_duplicates()
        else:
            items_table = items_info
        
        items_table.to_csv(items_file, index=False)
