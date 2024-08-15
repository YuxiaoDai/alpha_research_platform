
import pandas as pd
import numpy as np
import os
import yaml
import multiprocessing
from utils.PathManager import PathManager
from utils.ProcessedDataLoader import ProcessedDataLoader
from alpha.vol_5d import vol_5d
from alpha.iamp import iamp

class AlphaCalculator:

    def __init__(self):
        self.path_manager = PathManager()
        self.data_loader = ProcessedDataLoader()
        self.alpha_calc_map = {
            'vol_5d': vol_5d,
            'iamp': iamp,
         }

    def calc_alpha(self, alpha_name_list):
        
        # Create a process pool
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        
        # Asynchronously execute calc_single_alpha
        result = pool.map_async(self.calc_single_alpha, alpha_name_list)
        
        # Close the process pool and wait for all processes to complete
        pool.close()
        pool.join()
        
        # Retrieve all results
        results = result.get()
        return results

    def calc_single_alpha(self, alpha_name):

        # load configuration file
        print('Loading configuration file')
        alpha_calc_config = self.load_alpha_calc_config(alpha_name)
            
        # load input data
        print('Loading input data')
        input_data = self.load_input_data(alpha_calc_config)
            
        # load universe data
        print('Loading universe data')
        universe_data = self.load_universe_data(alpha_calc_config)
            
        # calculate alpha score
        print('Calculting alpha score.')
        raw_alpha_score, alpha_score = self.alpha_calc_map[alpha_name](input_data)

        # calculate alpha portfolio
        alpha_portfolio = self.calc_alpha_portfolio(alpha_name, alpha_score, universe_data, alpha_calc_config)

        # save alpha score  
        self.save_data_alpha_score(raw_alpha_score, alpha_name)

        # save alpha portfolio
        self.save_data_alpha_portfolio(alpha_portfolio)

        return alpha_portfolio

    def load_alpha_calc_config(self, alpha_name):
        
        # find the path for configuration file
        path = self.path_manager.get('alpha_calc_config')
        self.path_manager.check_path_exists(path)

        # load the configuration file
        with open(path, 'r') as file:
            alpha_calc_config_all = yaml.safe_load(file)
        
        alpha_calc_config = alpha_calc_config_all[alpha_name]

        return alpha_calc_config

    def load_input_data(self, alpha_calc_config):

        # load config
        input_data_items = alpha_calc_config['input_data_items']
        start = alpha_calc_config['start']
        end = alpha_calc_config['end']

        # Loop through each item and load its data
        for item in input_data_items:
            input_data = self.data_loader.loading(item, start, end)
            input_data = input_data.rename(columns={'val': input_data['item'].iloc[0]})
            del input_data['item']

            if item == input_data_items[0]:
                input_data_all = input_data
            else:
                input_data_all = pd.merge(input_data_all, input_data, on = ['code', 'date'], how='outer')
        
        return input_data_all


    def load_universe_data(self, alpha_calc_config):

        # load config
        universe_list = alpha_calc_config['universe_list']
        start = alpha_calc_config['start']
        end = alpha_calc_config['end']

        # TODO: Check the universe data. Why zz500/800/1000/9999 only has 300 rows each day?
        # TODO: add in processed data loader: load a list of processed data
        # Loop through each item and load its data
        for item in universe_list:
            universe_data = self.data_loader.loading(item, start, end)
            universe_data = universe_data.rename(columns={'val': universe_data['item'].iloc[0]})
            del universe_data['item']

            if item == universe_list[0]:
                universe_data_all = universe_data
            else:
                universe_data_all = pd.merge(universe_data_all, universe_data, on = ['code', 'date'], how='outer')
        
        return universe_data_all

    def convert_score_to_portfolio(self, group, alpha_calc_config):
        
        # Calculate the median
        median_alpha = group['alpha_score'].median()

        # Differentiate between long and short positions
        group['position'] = group['alpha_score'].apply(lambda x: 'long' if x > median_alpha else 'short')

        # Assign portfolio weights
        if alpha_calc_config['weight_scheme'] == 'equal_weight':
            half_count = len(group) // 2
            long_weight = 1.0 / half_count  
            short_weight = -1.0 / (len(group) - half_count)  
            group['val'] = group.apply(lambda x: long_weight if x['position'] == 'long' else short_weight, axis=1)

        elif alpha_calc_config['weight_scheme'] == 'value_weight':
            
            sum_long = group.loc[group['position'] == 'long', 'alpha_score'].sum()
            sum_short = group.loc[group['position'] == 'short', 'alpha_score'].sum()

            group['val'] = group.apply(lambda x: x['alpha_score'] / sum_long if x['position'] == 'long' else x['alpha_score']/ np.abs(sum_short), axis=1)

        return group[['code', 'date', 'val']]
    

    def calc_alpha_portfolio(self, alpha_name, alpha_score, universe_data, alpha_calc_config):
        
        # TODO: align code and date data type
        alpha_score['code'] = alpha_score['code'].astype(int)
        alpha_score['date'] = alpha_score['date'].astype(int)
        universe_data['code'] = universe_data['code'].astype(int)
        universe_data['date'] = universe_data['date'].astype(int)

        # merge alpha and universe data
        alpha_universe_data = pd.merge(alpha_score, universe_data, on = ['code', 'date'], how='outer')

        # loop through the universe list to create alpha portfolio
        alpha_portfolio_universe_dict = {}
        # alpha_name = alpha_calc_config['alpha_name']
        universe_list = alpha_calc_config['universe_list']
        for universe_name in universe_list:
            # filter universe data
            alpha_score = alpha_universe_data[alpha_universe_data[alpha_name].notnull() & (alpha_universe_data[universe_name].notnull())]
            alpha_score = alpha_score[['code', 'date', alpha_name, universe_name]]
            alpha_score.rename(columns={alpha_name: 'alpha_score'}, inplace=True)

            # convert alpha score to portfolio
            alpha_portfolio = alpha_score.groupby(['date']).apply(lambda x: self.convert_score_to_portfolio(x, alpha_calc_config)).reset_index(drop=True)

            # reformat alpha portfolio dataframe
            alpha_portfolio['item'] = f'{alpha_name}_{universe_name}'

            # save it to alpha_portfolio_dict
            alpha_portfolio_universe_dict[f'{alpha_name}_{universe_name}'] = alpha_portfolio

        return alpha_portfolio_universe_dict

    # update items table
    def update_items_table(self, item, alpha_type):
        # update items.parquet file
        items_info = pd.DataFrame({
            'item': [item],
            'description': ['desciption'],
            'source data': [''],
            'source path': [''],
            'saved path':  [self.path_manager.get(alpha_type)]
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

    def save_data_alpha_score(self, alpha_score, alpha_name):

        # Save data to the file system
        base_path = self.path_manager.get('alpha_score')

        # Create a folder for each alpha item
        item_path = os.path.join(base_path, alpha_name)
        os.makedirs(item_path, exist_ok=True)

        # reformat alpha_score
        alpha_score = alpha_score.rename(columns={alpha_name: 'val'})
        alpha_score['item'] = alpha_name
        alpha_score['code'] = alpha_score['code'].astype('Int32') 
        alpha_score['date'] = alpha_score['date'].astype('Int32') 
        alpha_score['item'] = alpha_score['item'].astype('string')
                
        # Save a separate file for each date
        for date, group in alpha_score.groupby('date'):
            date_str = pd.to_datetime(date, format='%Y%m%d').strftime('%Y%m%d')
            file_path = os.path.join(item_path, f"{date_str}.parquet")
            group.to_parquet(file_path) 

        # update items table
        self.update_items_table(alpha_name, alpha_type = 'alpha_score')
        
        
    # save alpha_portfolio_dict
    def save_data_alpha_portfolio(self, alpha_portfolio_dict):
        
        # Save data to the file system
        base_path = self.path_manager.get('alpha_portfolio')

        for item, data in alpha_portfolio_dict.items():

            # Create a folder for each alpha item
            item_path = os.path.join(base_path, item)
            os.makedirs(item_path, exist_ok=True)
                
            # Save a separate file for each date
            for date, group in data.groupby('date'):
                date_str = pd.to_datetime(date, format='%Y%m%d').strftime('%Y%m%d')
                file_path = os.path.join(item_path, f"{date_str}.parquet")
                group.to_parquet(file_path) 

            # update items table
            self.update_items_table(item, alpha_type = 'alpha_portfolio')

    