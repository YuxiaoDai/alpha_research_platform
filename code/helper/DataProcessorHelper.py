import pandas as pd
import numpy as np
from utils.PathManager import PathManager
from utils.RawDataLoader import RawDataLoader

class DataProcessorHelper:
    def __init__(self):
        self.data_loader = RawDataLoader()
        self.path_manager = PathManager()

    def adj_fct_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = self.reformat_data(data)
        return data

    def lmt_processor(self, data, tab_name):
    
        data = self.clean_data(data, tab_name)
        data = self.reformat_data(data)
        return data

    def mkt_val_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = self.reformat_data(data)
        return data

    def sw_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = self.reformat_data(data)
        return data
    
    def univ_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = self.reformat_data_univ(data, tab_name)
        return data
    
    def idx_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = self.reformat_data_idx(data)
        return data
    
    def min_pv_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = self.reformat_data_1min_pv(data)
        return data

    def date_processor(self, data, tab_name):
        
        data = self.clean_data(data, tab_name)
        data = {tab_name: data}
        return data
    
    def min_pv_aggregator(self, processed_data, func):
        
        aggregated_data = {}

        for item, df in processed_data.items():
            
            grouped = df.groupby(['code', 'date'])

            # Apply aggregation functions to each group;
            if isinstance(func, list):  
                aggregated = grouped['val'].agg(func).reset_index()
            else:  
                aggregated = grouped['val'].agg([func]).reset_index()

            # Rename columns, adding 'item' as a prefix only for the results of aggregation functions;
            aggregated.columns = [f"{item}" if col != 'code' and col != 'date' else col for col in aggregated.columns]
    
            # Transform data format to code, date, item, val;
            aggregated_dict = self.reformat_data(aggregated)

            # Append aggregated results to the result dictionary.
            aggregated_data = aggregated_data | aggregated_dict

        return aggregated_data

    def clean_data(self, data, tab_name):
        
        data = data.dropna(how='all') 
        
        # Additional data cleaning logic can be added here, such as handling outliers.
        # TODO: In the LMT data, do we need to address extreme values where uplimit >= 1e5 or down_limit = 0.01?
        # TODO: In the MKT_VAL data, there is one row with negative values for neg_shares and neg_mkt_val, while the surrounding values are normal. This needs to be fixed.
        # TODO: In the 1min_pv data, some rows are filled entirely with zeros. 
        
        # Replace these zeros with NA (replace all zero values with NaN).
        if(tab_name == '1min_pv'):
            data.replace(0, np.nan, inplace=True)
            
        return data

    def reformat_data(self, data):
        
        items = [col for col in data.columns if col not in ['date', 'code']]

        if not isinstance(items, list):
            items = [items]

        dataframes_dict = {}

        for item in items:
            tmp = data[['code', 'date', item]].rename(columns={item: 'val'})
            tmp['item'] = item

            # keep data dtype consistent
            tmp['code'] = tmp['code'].astype('Int32') 
            tmp['date'] = tmp['date'].astype('Int32') 
            tmp['item'] = tmp['item'].astype('string')

            dataframes_dict[item] = tmp

        return dataframes_dict
    
    def reformat_data_univ(self, data, tab_name):
        
        data[[tab_name]] = 1

        if tab_name != 'zz9999':
            items = [col for col in data.columns if col not in ['date', 'code', 'name']]
        else:
            items = [col for col in data.columns if col not in ['date', 'code']]

        if not isinstance(items, list):
            items = [items]

        dataframes_dict = {}

        for item in items:
            tmp = data[['code', 'date', item]].rename(columns={item: 'val'})
            tmp['item'] = item
            
            # keep data dtype consistent
            tmp['code'] = tmp['code'].astype('Int32') 
            tmp['date'] = tmp['date'].astype('Int32') 
            tmp['item'] = tmp['item'].astype('string')

            dataframes_dict[item] = tmp

        return dataframes_dict

    def reformat_data_idx(self, data):
        
        items = [col for col in data.columns if col not in ['date', 'code']]

        if not isinstance(items, list):
            items = [items]

        dataframes_dict = {}

        for item in items:
            tmp = data[['code', 'date', item]].rename(columns={item: 'val'})
            item = f'{item}_idx'
            tmp['item'] = item

            # keep data dtype consistent
            tmp['code'] = tmp['code'].astype('Int32') 
            tmp['date'] = tmp['date'].astype('Int32') 
            tmp['item'] = tmp['item'].astype('string')

            dataframes_dict[item] = tmp

        return dataframes_dict
    
    def reformat_data_1min_pv(self, data):

        # Load adjustment factor data
        start = pd.to_datetime(data['date'], format='%Y%m%d').min().strftime('%Y-%m-%d')
        end   = pd.to_datetime(data['date'], format='%Y%m%d').max().strftime('%Y-%m-%d')
        adj_factors = self.data_loader.loading('adj_fct', start, end)
        adj_factors['date']= adj_factors['date'].astype(int)
        
        # Perform adjustment and reformat the data
        items =  [col for col in data.columns if col not in ['date', 'code', 'time']]
        price_columns = [col for col in items if col in ['pre_close', 'open', 'high', 'low', 'close']]
        volume_columns = [col for col in items if col in ['volume', 'turover', 'accvolume', 'accturover']]
        data_adj = pd.merge(data, adj_factors, on = ['date', 'code'], how = 'left')
        for col in price_columns:
            data_adj[col] = data_adj[col] * (data_adj['cum_adjf'] / 100000)
        for col in volume_columns:
            data_adj[col] = data_adj[col] / (data_adj['cum_adjf'] / 100000)
        del data_adj['cum_adjf']
                
        # Convert the data format to: code, date, item, val
        if not isinstance(items, list):
            items = [items]

        dataframes_dict = {}

        for item in items:

            # Add original data (unadjusted)
            tmp = data[['code', 'date', 'time', item]].rename(columns={item: 'val'})
            item = f'{item}'
            tmp['item'] = item
            tmp['code'] = tmp['code'].astype('Int32') 
            tmp['date'] = tmp['date'].astype('Int32') 
            tmp['item'] = tmp['item'].astype('string')
            dataframes_dict[item] = tmp

            # Add adjusted data (after applying the adjustment factors)
            if item in price_columns or item in volume_columns:
                tmp = data_adj[['code', 'date', 'time', item]].rename(columns={item: 'val'})
                item = f'{item}_adj'
                tmp['item'] = item
                tmp['code'] = tmp['code'].astype('Int32') 
                tmp['date'] = tmp['date'].astype('Int32') 
                tmp['item'] = tmp['item'].astype('string')
                dataframes_dict[item] = tmp

        return dataframes_dict
    
    