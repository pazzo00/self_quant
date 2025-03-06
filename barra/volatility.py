from base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from barra.functions import gen_weights, dfs_to_df
from barra.run_ols import  RunOLS
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    data = BaseDataLoader.load_data('../../数据/stock_bar_1day.parquet', fields=['close'])
    end = data.trade_days[-1]
    data.data = data.data[1:] / data.data[:-1] - 1
    data.trade_days = data.trade_days[1:]

    index_data = BaseDataLoader.load_data('../../数据/index_bar_1day.parquet', fields=['close'],
                                          end=end, codes=['000300.SH'])
    index_data.data = index_data.data[1:] / index_data.data[:-1] - 1
    index_data.trade_days = index_data.trade_days[1:]

    start = data.trade_days.index(index_data.trade_days[0])
    data.data = data.data[start:]
    data.trade_days = data.trade_days[start:]
    ret = pd.concat([data.to_dataframe('close'), index_data.to_dataframe('close')], axis=1)

    data = RunOLS(index_name='000300.SH', std_halflife=42, ols_halflife=63,
                  window=252, cal_std=True, only_alpha=False)
    data()

    dfs = {'beta': pd.concat(data.beta, axis=1).T, 'hist_sigma': pd.concat(data.hist_sigma, axis=1).T,
           'daily_std': pd.concat(data.daily_std)}

    cumulative_range = []
    ret = np.log(1 + ret)
    for i in range(1, 13):
        cumulative_range.append(ret.rolling(21 * i, min_periods=1).sum().iloc[:, :-1].values)

    cumulative_range = np.stack(cumulative_range, axis=2)
    cumulative_range = np.max(cumulative_range, axis=2) - np.min(cumulative_range, axis=2)
    cumulative_range = pd.DataFrame(data=cumulative_range, index=data.trade_days, columns=data.codes)
    dfs['cumulative_range'] = cumulative_range
    dfs['alpha'] = pd.concat(data.alpha, axis=1).T

    for key in dfs:
        dfs[key].index.name = 'datetime'
        dfs[key].reset_index(inplace=True)
        dfs[key] = pd.melt(dfs[key], id_vars='datetime', var_name='code', value_name=key)
        dfs[key].set_index(['datetime', 'code'], inplace=True)
    df = pd.concat(list(dfs.values()), axis=1)
    df.reset_index(inplace=True)
    df['residual_volatility'] = (df['hist_sigma'] + df['daily_std'] + df['cumulative_range']) / 3
    df['volatility'] = (df['beta'] + df['residual_volatility']) / 2

    old_df = pd.read_parquet('../barra_factors.parquet')
    old_df = old_df.merge(df, how='left', on=['datetime', 'code'])
    old_df.to_parquet('../barra_factors.parquet')

    # data = BaseDataLoader.load_data('./volatility.parquet')
    # data = data.to_dataframes()
    # for key in data:
    #     data[key] = (data[key] - data[key].mean(axis=1).values.reshape(-1, 1)) / data[key].std(axis=1).values.reshape(-1, 1)
    # data = dfs_to_df(data)
    # data['residual_volatility'] = (data['hist_sigma'] + data['daily_std'] + data['cumulative_range']) / 3
    # data['volatility'] = (data['residual_volatility'] + data['beta']) / 2
    #
    # data = data[['code', 'datetime', 'residual_volatility', 'volatility']]
    # old_df = pd.read_parquet('./volatility.parquet')
    # old_df = old_df.drop(columns=['residual_volatility', 'volatility'])
    # old_df = old_df.merge(data, how='left', on=['datetime', 'code'])
    # old_df.to_parquet('./volatility.parquet')
