import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
from tabulate import tabulate
from utils.PathManager import PathManager
from utils.ProcessedDataLoader import ProcessedDataLoader



class AlphaPerformanceAnalyzer:

    # TODO: Align start and end. 
    # AlphaCalculator: start and end of loader, start adn end of alpha score
    # AlphaPerformanceAnalysis: start and end of loader

    def __init__(self):
        self.path_manager = PathManager()
        self.data_loader = ProcessedDataLoader()


    def analyze_alpha_performance(self, alpha_portf_name_list, start, end, is_limit_adj = True, rebalance_day = 'delay1', cum_rtrn_method = 'add'):

        """
        analyze alpha performance

        :param alpha_portf_name_list: A list of alphas to analyze 
        :param start: Start date in YYYY-MM-DD format
        :param end: End date in YYYY-MM-DD format
        :param limit_adj: adjust based on up limit and down limit
        :param rebalance_day: When to rebalance, delay0 or delay1. 
                delay0: Calculate position and trade on day (t), calculate PnL on day (t+1).
                delay1: Calculate position on day (t), trade on day (t+1), and calculate PnL on day (t+2).
        """

        # Create a process pool
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

        # Create a list containing all the parameters to be passed to the helper function
        tasks = [(name, start, end, is_limit_adj, rebalance_day, cum_rtrn_method) for name in alpha_portf_name_list]

        # Asynchronously map the helper function to the task list
        result = pool.map_async(self.analyze_single_alpha_performance, tasks)

        # Close the process pool and wait for all processes to complete
        pool.close()
        pool.join()

        # retrieve all results
        results = result.get()

        # show results
        for i in range(len(alpha_portf_name_list)):
            peformance_df_summary, ic_summary, summary_plot = results[i][0], results[i][1], results[i][2]
            print(tabulate(peformance_df_summary, headers='keys', tablefmt='psql'))
            print(tabulate(ic_summary, headers='keys', tablefmt='psql'))
            summary_plot.show()

            # Combine all portfolio returns and calculate the correlation
            if i == 0:
                portfolio_return_all = results[i][3]
            else:
                portfolio_return_all = pd.merge(portfolio_return_all, results[i][3], on = 'date', how='outer')

        # calculate alpha corr
        portfolio_return_all.drop('date', axis=1, inplace=True)
        alpha_corr = portfolio_return_all.corr()
        print(tabulate(alpha_corr, headers='keys', tablefmt='psql'))

    def analyze_single_alpha_performance(self, args):

        alpha_portf_name, start, end, is_limit_adj, rebalance_day, cum_rtrn_method = args

        # load alpha score
        alpha_score_name, _,_ = alpha_portf_name.rpartition("_")
        alpha_score = self.data_loader.loading(alpha_score_name, start, end) 

        # load alpha portfolio
        alpha_portfolio = self.data_loader.loading(alpha_portf_name, start, end)

        # alpha_score = alpha_portfolio
        
        # If delay0 is used, the alpha weight is calculated on day t and adjusted on day t
        # If delay1 is used, the alpha weight is calculated on day t and adjusted on day t+1
        if rebalance_day == 'delay1':
            alpha_portfolio[['val']] = alpha_portfolio.groupby('code')[['val']].shift(1)
        
        # adjust alpha portfolio for up and down limit
        if is_limit_adj:
            alpha_portfolio = self.adjust_position_for_lmt(alpha_portfolio)
        
        # calculate forward return
        forward_return = self.calc_forward_daily_return(start, end, 1)

        # calculate portfolio return
        portfolio_return = self.calc_portfolio_return(alpha_portfolio, forward_return)
        portfolio_return.rename(columns={'portfolio_return': alpha_portf_name}, inplace = True)

        # calculate portfolio performance
        peformance_df, portfolio_cum_return             = self.calc_portf_performance(alpha_portfolio, forward_return, alpha_portf_name, cum_rtrn_method)
        peformance_df_long, portfolio_cum_return_long   = self.calc_portf_performance(alpha_portfolio[alpha_portfolio['val'] > 0], forward_return, alpha_portf_name, cum_rtrn_method)
        peformance_df_short, portfolio_cum_return_short = self.calc_portf_performance(alpha_portfolio[alpha_portfolio['val'] < 0], forward_return, alpha_portf_name, cum_rtrn_method)
        peformance_df['portfolio'] = 'all'
        peformance_df_long['portfolio'] = 'long'
        peformance_df_short['portfolio'] = 'short'
        peformance_df_summary = pd.concat([peformance_df, peformance_df_long, peformance_df_short])

        # calculate ic
        ic_summary, ic_pearson, ic_spearman = self.calc_ic(alpha_portfolio, forward_return)

        # plot
        summary_plot = self.plot_summary(start, end, alpha_score, alpha_portfolio, 
                                         portfolio_cum_return, portfolio_cum_return_long, portfolio_cum_return_short,
                                         ic_pearson, ic_spearman, alpha_portf_name, forward_return, cum_rtrn_method)

        return peformance_df_summary, ic_summary, summary_plot, portfolio_return
    
    
    def adjust_position_for_lmt(self, alpha_portfolio): 
        
        '''
        Handle the impact of limit-up and limit-down prices
        '''

        # Load data: limit-up prices, limit-down prices, and closing prices
        start = datetime.strptime(str(alpha_portfolio['date'].min()), '%Y%m%d').strftime('%Y-%m-%d')
        end = datetime.strptime(str(alpha_portfolio['date'].max()), '%Y%m%d').strftime('%Y-%m-%d')
        data_loader_processed = ProcessedDataLoader()
        up_limit = data_loader_processed.loading('up_limit', start, end)
        down_limit = data_loader_processed.loading('down_limit',  start, end)
        up_limit['date'] = up_limit['date'].astype(int)
        down_limit['date'] = down_limit['date'].astype(int)
        close = data_loader_processed.loading('close',  start, end)
        close['val'] = close['val']/10000 

        del up_limit['item']
        del down_limit['item']
        del close['item']

        # merge dataset
        merged = alpha_portfolio.merge(up_limit, on=['code', 'date'], suffixes=('', '_up'))
        merged = merged.merge(down_limit, on=['code', 'date'], suffixes=('', '_down'))
        merged = merged.merge(close, on=['code', 'date'], suffixes=('', '_close'))

        # adjust based on up_limit and down_limit
        # can't long stock if the up-limit is hit, or short stock if the down-limit is hit.
        merged.sort_values(by=['code', 'date'], inplace=True)
        merged['hit'] = 0
        merged['hit'] = merged['hit'].mask(merged['val_close'] >= merged['val_up'], 1)
        merged['hit'] = merged['hit'].mask(merged['val_close'] <= merged['val_down'], -1)
        lmt_hit = pd.pivot_table(merged[['code', 'date', 'hit']], index='date', columns='code', values='hit')
        alpha = pd.pivot_table(merged[['code', 'date', 'val']], index='date', columns='code', values='val')
        lmt_hit = lmt_hit.reindex(index=alpha.index, columns=alpha.columns).fillna(0)
        alpha_pos = alpha.copy()
        pre_pos = pd.Series(0, index=alpha.columns)
        for dt, pos in alpha_pos.iterrows():
            lmt_hit_row = lmt_hit.loc[dt]
            # can't long stock if the up-limit is hit, or short stock if the down-limit is hit.
            mask = ((pos > pre_pos) & (lmt_hit_row == 1)) | ((pos < pre_pos) & (lmt_hit_row == -1))  
            alpha_pos.loc[dt] = pos.mask(mask, pre_pos)
            pre_pos = alpha_pos.loc[dt]
        alpha_pos = alpha_pos.reset_index()
        alpha_pos_long = pd.melt(alpha_pos, id_vars=['date'], var_name='code', value_name='val')
        alpha_pos_long['item'] = merged['item'].iloc[0]

        return alpha_pos_long[['code', 'date', 'item','val']]

    def calc_daily_return(self, start, end):

        # load daily close data
        df = self.data_loader.loading('close_adj', start, end)
        del df['item']

        # calculate daily return
        df.sort_values(by=['code', 'date'], inplace=True)
        df['daily_return'] = df.groupby('code')['val'].pct_change()
        del df['val']

        return df

    def calc_forward_daily_return(self, start, end, day):

        # calculate daily return
        df = self.calc_daily_return(start, end)

        # calculate forward 1 day return
        df.sort_values(by=['code', 'date'], inplace=True)
        df['daily_return_forward'] = df.groupby('code')['daily_return'].shift(-day) 

        return df
    
    def calc_portf_performance(self, alpha_portfolio, forward_return, alpha_portf_name, cum_rtrn_method = 'add'):

        # calculate portfolio return
        portfolio_return = self.calc_portfolio_return(alpha_portfolio, forward_return)

        # calculate cumulative portfolio return （pnl)
        portfolio_cum_return = self.calc_portfolio_cumulative_return(portfolio_return, cum_rtrn_method)

        # calculate annualized return
        annualized_return = self.calc_annualized_return(portfolio_cum_return, cum_rtrn_method)
            
        # calculate annualized risk
        annualized_risk = self.calc_annualized_risk(portfolio_return)

        # calculate annualized Sharpe
        annualized_IR = self.calc_IR(portfolio_return, portfolio_cum_return, cum_rtrn_method)

        # calculate maximum drawdown
        maximum_drawdown = self.calc_max_drawdown(portfolio_cum_return)

        # calculate turnover
        average_daily_turnover = self.calc_average_daily_turnover(alpha_portfolio)

        # summarize results to dataframe
        peformance_df = pd.DataFrame({'alpha' : [alpha_portf_name],
                                    'annualized_return': [annualized_return],
                                    'annualized_risk': [annualized_risk],
                                    'annualized_IR': [annualized_IR],
                                    'maximum_drawdown': [maximum_drawdown],
                                    'average_daily_turnover': [average_daily_turnover]})
        
        return peformance_df, portfolio_cum_return

    def calc_ic(self, alpha_portfolio, forward_return):

        def calculate_correlation(group, method):
            # Pearson's correlation
            return group['val'].corr(group['daily_return_forward'], method = method)

        # merge alpha_portfolio and forward_return
        alpha_portfolio_return = pd.merge(alpha_portfolio, forward_return, on = ['code', 'date'], how = 'left')
        del alpha_portfolio_return['item']

        # calculate ic by date (pearson)
        ic_pearson = alpha_portfolio_return.groupby('date').apply(calculate_correlation, 'pearson')
        ic_pearson = ic_pearson.reset_index().rename(columns={0: 'IC'})
        ic_pearson_mean = ic_pearson['IC'].mean()
        ic_pearson_se = ic_pearson['IC'].std() / np.sqrt(len(ic_pearson['IC']))
        ic_pearson_tstat = ic_pearson_mean / ic_pearson_se

        # calculate ic by date (spearman)
        ic_spearman = alpha_portfolio_return.groupby('date').apply(calculate_correlation, 'spearman')
        ic_spearman = ic_spearman.reset_index().rename(columns={0: 'IC'})
        ic_spearman_mean = ic_spearman['IC'].mean()
        ic_spearman_se = ic_spearman['IC'].std() / np.sqrt(len(ic_spearman['IC']))
        ic_spearman_tstat = ic_spearman_mean / ic_spearman_se

        # summary
        ic_summary = pd.DataFrame({'Stats': ['mean', 'tstat'],
                                  'IC (pearson)':  [ic_pearson_mean, ic_pearson_tstat],
                                  'IC (spearman - rank)': [ic_spearman_mean, ic_spearman_tstat]})

        return ic_summary, ic_pearson, ic_spearman

    def calc_portfolio_return(self, alpha_portfolio, forward_return):

        # merge alpha_portfolio and forward_return
        alpha_portfolio_return = pd.merge(alpha_portfolio, forward_return, on = ['code', 'date'], how = 'left')
        del alpha_portfolio_return['item']

        # calculate each stock's contribution
        alpha_portfolio_return['contribution'] = alpha_portfolio_return['val'] * alpha_portfolio_return['daily_return_forward']

        # calculate portfolio return
        portfolio_return = alpha_portfolio_return.groupby('date')['contribution'].sum().reset_index(name='portfolio_return')

        return portfolio_return
    
    def calc_portfolio_cumulative_return(self, portfolio_return, cum_rtrn_method = 'add'):

        # sort by date
        portfolio_return = portfolio_return.sort_values(by='date')
        
        # calculate cumulative return
        if cum_rtrn_method == 'compound':
            portfolio_return['portfolio_cumulative_return'] = (1 + portfolio_return['portfolio_return']).cumprod() - 1
        elif cum_rtrn_method == 'add':
            portfolio_return['portfolio_cumulative_return'] = portfolio_return['portfolio_return'].cumsum() 

        del portfolio_return['portfolio_return']

        return portfolio_return

    def calc_annualized_return(self, portfolio_cum_return, cum_rtrn_method):
        # sort by date
        portfolio_cum_return = portfolio_cum_return.sort_values('date')
        total_return = portfolio_cum_return['portfolio_cumulative_return'].iloc[-1]
        num_days = portfolio_cum_return.shape[0]

        if  cum_rtrn_method == 'compound':
            annualized_return = (1 + total_return) ** (252.0/ num_days) - 1
        elif cum_rtrn_method == 'add':
            annualized_return = total_return / num_days * 252
        return annualized_return

    def calc_annualized_risk(self, portfolio_return):
        daily_risk = portfolio_return['portfolio_return'].std()
        annualized_risk = daily_risk * np.sqrt(252) 
        return annualized_risk
    
    def calc_IR(self, portfolio_return, portfolio_cum_return, cum_rtrn_method):
        ann_return = self.calc_annualized_return(portfolio_cum_return, cum_rtrn_method)
        ann_risk = self.calc_annualized_risk(portfolio_return)
        return ann_return / ann_risk if ann_risk != 0 else np.nan
    
    def calc_max_drawdown(self, portfolio_cum_return):
        drawdown = (1+portfolio_cum_return['portfolio_cumulative_return']) / (1+portfolio_cum_return['portfolio_cumulative_return']).cummax() - 1
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def calc_daily_turnover(self, alpha_portfolio):

        # sort by code and date
        alpha_portfolio = alpha_portfolio.sort_values(by=['code', 'date'])
        
        # calculate positive change for each stock
        alpha_portfolio['position_change'] = alpha_portfolio.groupby('code')['val'].diff().abs()
        
        # Summarize the total turnover rate for each day (two-way turnover)
        daily_turnover = alpha_portfolio.groupby('date')['position_change'].sum()

        # convert to one-way turnover
        daily_turnover = daily_turnover / 2
        
        return daily_turnover
    
    def calc_average_daily_turnover(self, alpha_portfolio):
        
        # sort by code and date
        alpha_portfolio = alpha_portfolio.sort_values(by=['code', 'date'])
        
        # calculate positive change for each stock
        alpha_portfolio['position_change'] = alpha_portfolio.groupby('code')['val'].diff().abs()
        
        # Summarize the total turnover rate for each day (two-way turnover)
        daily_turnover = alpha_portfolio.groupby('date')['position_change'].sum()
        
        # calculate average turnover
        average_turnover = daily_turnover.mean()

        # convert to one-way turnover
        average_turnover = average_turnover / 2

        return average_turnover

    def plot_cumulative_return(self, ax, portfolio_cum_return):
        ax[0,0].plot(pd.to_datetime(portfolio_cum_return['date'].astype(str), format='%Y%m%d'),
                    portfolio_cum_return['portfolio_cumulative_return'])
        ax[0,0].set_title(f'Cumulative Return')
        ax[0,0].grid(True)
        ax[0,0].tick_params(axis='x', rotation=45)
    
    def plot_long_short(self, ax, portfolio_cum_return_long, portfolio_cum_return_short):
        ax[0,1].plot(pd.to_datetime(portfolio_cum_return_long['date'].astype(str), format='%Y%m%d'),
                    portfolio_cum_return_long['portfolio_cumulative_return'], color='red', label='Long side')
        ax[0,1].plot(pd.to_datetime(portfolio_cum_return_short['date'].astype(str), format='%Y%m%d'),
                    portfolio_cum_return_short['portfolio_cumulative_return'], color='green', label='Short side')
        ax[0,1].set_title(f'Cumulative Return - Long vs Short')
        ax[0,1].grid(True)
        ax[0,1].tick_params(axis='x', rotation=45)
        ax[0,1].legend()

    def plot_drawdown(self, ax, portfolio_cum_return):
        # calculate drawdown
        drawdown = (1+portfolio_cum_return['portfolio_cumulative_return']) / \
                    (1+portfolio_cum_return['portfolio_cumulative_return']).cummax() - 1
        
        # plot
        ax[0,2].plot(pd.to_datetime(portfolio_cum_return['date'].astype(str), format='%Y%m%d'), 
                    drawdown)
        ax[0,2].set_title('Drawdown')
        ax[0,2].grid(True)
        ax[0,2].tick_params(axis='x', rotation=45)

    def plot_daily_turnover(self, ax, alpha_portfolio):
        # calculate turnover
        alpha_portfolio = alpha_portfolio.sort_values(by=['code', 'date'])
        alpha_portfolio['position_change'] = alpha_portfolio.groupby('code')['val'].diff().abs()
        daily_turnover = alpha_portfolio.groupby('date')['position_change'].sum().reset_index()
        daily_turnover = daily_turnover.drop(daily_turnover.index[0])

        # convert 2-way turnover to 1-way turnover
        daily_turnover['position_change'] = daily_turnover['position_change'] / 2
        
        #plot
        ax[0,3].plot(pd.to_datetime(daily_turnover['date'].astype(str), format='%Y%m%d'), 
                     daily_turnover['position_change'])
        ax[0,3].set_title('Turnover')
        ax[0,3].grid(True)
        ax[0,3].tick_params(axis='x', rotation=45)

    def plot_ic_pearson(self, ax, ic_pearson):
        ax[1,0].hist(ic_pearson['IC'], bins=20, color='blue', alpha=0.7)
        ax[1,0].set_title(f'Distribution of IC (Pearson Correlation)')
        ax[1,0].grid(True)
        ax[1,0].tick_params(axis='x', rotation=45)

    def plot_ic_spearman(self, ax, ic_spearman):
        ax[1,1].hist(ic_spearman['IC'], bins=20, color='blue', alpha=0.7)
        ax[1,1].set_title(f'Distribution of IC (Spearman Correlation - Rank)')
        ax[1,1].grid(True)
        ax[1,1].tick_params(axis='x', rotation=45)

    def plot_spread_return(self, ax, alpha_portfolio, forward_return, alpha_portf_name, cum_rtrn_method):
        # Rank alpha portfolio by value in descending order, grouped by date
        alpha_portfolio['rank'] = alpha_portfolio.groupby('date')['val'].rank(pct=True, ascending=False, method='max')
        
        # Define the rank thresholds and initialize a dictionary to store performance data
        rank_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
        performance_dfs = {}
        
        # Iterate over the thresholds, calculate weights, and compute portfolio performance
        for i, threshold in enumerate(rank_thresholds, 1):
            lower_bound = rank_thresholds[i-2] if i > 1 else 0
            group = alpha_portfolio[(alpha_portfolio['rank'] > lower_bound) & (alpha_portfolio['rank'] <= threshold)]
            group['val'] = 1 / group.groupby('date')['date'].transform('size')
            
            # Compute performance if the group is not empty; otherwise, create a default performance DataFrame
            if not group.empty:
                performance_dfs[f'top{int(threshold*100)}'], _ = self.calc_portf_performance(group, forward_return, alpha_portf_name, cum_rtrn_method)
            else:
                performance_dfs[f'top{int(threshold*100)}'] = pd.DataFrame({
                    'alpha': [f'top{int(threshold*100)}'], 
                    'annualized_return': [0], 
                    'annualized_risk': [0], 
                    'annualized_IR': [0], 
                    'maximum_drawdown': [0], 
                    'average_daily_turnover': [0]
                })
            
            # Label each alpha performance DataFrame
            performance_dfs[f'top{int(threshold*100)}']['alpha'] = f'{int(threshold*100)}%'
        
        # Concatenate all performance DataFrames
        performance_df_all_buckets = pd.concat(performance_dfs.values())
        
        # Plot the results
        ax[1, 2].bar(performance_df_all_buckets['alpha'], performance_df_all_buckets['annualized_return'], color='blue')
        
        # Adding titles, labels, and formatting
        ax[1, 2].set_title('Alpha-Return Monotonicity')
        ax[1, 2].grid(True)
        ax[1, 2].tick_params(axis='x', rotation=45)
        ax[1, 2].legend()


    def plot_alpha_decay(self, ax, alpha_portfolio, start, end):

        ic_iday_list = []
        daily_return = self.calc_daily_return(start, end)
        daily_return.sort_values(by=['code', 'date'], inplace=True)
        for i in range(1, 21):
            
            daily_return['daily_return_forward'] = daily_return.groupby('code')['daily_return'].shift(-i) 
            ic_iday, _, _ = self.calc_ic(alpha_portfolio, daily_return)

            ic_iday['Day'] = i
            ic_iday = ic_iday[ic_iday['Stats'] == 'mean']
            del ic_iday['Stats']

            ic_iday_list.append(ic_iday) 
        alpha_decay = pd.concat(ic_iday_list)

        ax[1,3].plot(alpha_decay['Day'], alpha_decay['IC (pearson)'], color='red', label='IC (pearson)')
        ax[1,3].plot(alpha_decay['Day'], alpha_decay['IC (spearman - rank)'], color='green', label='IC (spearman - rank)')
        ax[1,3].set_title(f'Alpha Decay')
        ax[1,3].grid(True)
        ax[1,3].tick_params(axis='x', rotation=45)
        ax[1,3].legend()

    def plot_position_distribution(self, ax, alpha_portfolio):
        ax[2,0].hist(alpha_portfolio['val'], bins=50, color='blue', alpha=0.7)
        ax[2,0].set_title(f'Distribution of Portfolio Position')
        ax[2,0].grid(True)
        ax[2,0].tick_params(axis='x', rotation=45)

    
    def plot_weight_distribution_overtime(self, ax, alpha_portfolio):
        def calculate_weight_distribution(alpha_portfolio, ascending):
            thresholds = {
                'top5%': 0.05,
                'top10%': 0.10,
                'top20%': 0.20,
                'top40%': 0.40,
                'all': 1.00
            }
            weight_distributions = {}
            
            for label, threshold in thresholds.items():
                subset = alpha_portfolio[alpha_portfolio['rank'] <= threshold]
                weight_distributions[label] = subset.groupby('date')['val'].sum().reset_index().rename(columns={'val': label})

            # Merge all distributions into one DataFrame
            merged_distribution = weight_distributions['top5%']
            for label in list(thresholds.keys())[1:]:
                merged_distribution = pd.merge(merged_distribution, weight_distributions[label], on='date', how='outer')
            
            return merged_distribution

        # Prepare long and short legs
        alpha_portfolio_long = alpha_portfolio[alpha_portfolio['val'] > 0]
        alpha_portfolio_short = alpha_portfolio[alpha_portfolio['val'] < 0]

        # Rank within each date for long and short legs
        alpha_portfolio_long['rank'] = alpha_portfolio_long.groupby('date')['val'].rank(pct=True, ascending=False, method='max')
        alpha_portfolio_short['rank'] = alpha_portfolio_short.groupby('date')['val'].rank(pct=True, ascending=True, method='max')

        # Calculate weight distributions
        long_distribution = calculate_weight_distribution(alpha_portfolio_long, ascending=False)
        short_distribution = calculate_weight_distribution(alpha_portfolio_short, ascending=True)

        # Merge long and short distributions
        long_short_distribution = pd.merge(long_distribution, short_distribution, on='date', how='outer', suffixes=('_Long', '_Short'))

        # Plotting weight distribution over time
        dates = pd.to_datetime(long_short_distribution['date'].astype(str), format='%Y%m%d')
        for col in long_short_distribution.columns[1:]:
            if 'Long' in col:
                ax[2, 1].plot(dates, long_short_distribution[col], label=col.replace('_Long', ''), color='red')
            else:
                ax[2, 1].plot(dates, long_short_distribution[col], label=col.replace('_Short', ''), color='green')

        # Adding titles and labels
        ax[2, 1].set_title('Weight Distribution Over Time')
        ax[2, 1].grid(True)
        ax[2, 1].tick_params(axis='x', rotation=45)
        ax[2, 1].legend()

    def plot_score_distribution(self, ax, alpha_score):
        # remove inf and -inf
        alpha_score = alpha_score[(alpha_score['val'] != np.inf) & (alpha_score['val'] != - np.inf)]

        ax[2,2].hist(alpha_score['val'], bins=50, color='blue', alpha=0.7)
        ax[2,2].set_title(f'Distribution of Alpha Score')
        ax[2,2].grid(True)
        ax[2,2].tick_params(axis='x', rotation=45)

    def plot_summary(self, start, end, alpha_score, alpha_portfolio, portfolio_cum_return, portfolio_cum_return_long, portfolio_cum_return_short,
                     ic_by_code, ic_by_date, alpha_portf_name, forward_return, cum_rtrn_method):
        fig, ax = plt.subplots(3, 4, figsize=(24, 18))
        self.plot_cumulative_return(ax, portfolio_cum_return)
        self.plot_long_short(ax, portfolio_cum_return_long, portfolio_cum_return_short)
        self.plot_drawdown(ax, portfolio_cum_return)
        self.plot_daily_turnover(ax, alpha_portfolio)
        self.plot_ic_pearson(ax, ic_by_code)
        self.plot_ic_spearman(ax, ic_by_date)
        self.plot_spread_return(ax, alpha_portfolio, forward_return, alpha_portf_name, cum_rtrn_method)
        self.plot_alpha_decay(ax, alpha_portfolio, start, end)
        self.plot_position_distribution(ax, alpha_portfolio)
        self.plot_weight_distribution_overtime(ax, alpha_portfolio)
        self.plot_score_distribution(ax, alpha_score)
        fig.suptitle(alpha_portf_name, fontsize=18)

        return fig
    