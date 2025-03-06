from base_data_loader import BaseDataLoader
from report_data_loader import ReportDataLoader
import pandas as pd
from barra.functions import cal_quarter_data, dfs_to_df
from barra.run_ols import RunOLS
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import time



if __name__ == '__main__':
    cap = BaseDataLoader.load_data('../../数据/capital.parquet', fields=['market_cap', 'pb_ratio',
                                                                         'pe_ratio', 'pcf_ratio'])
    income = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                         fields=['operating_revenue', 'operating_cost'],
                                         lag='4Q', codes=cap.codes)

    revenue, cost = [], []
    for i in range(4):
        flag = income.season_data.values == 1
        revenue.append(cal_quarter_data(income.data, flag, 2 * i, 2, income.season_data.values))
        cost.append(cal_quarter_data(income.data, flag, 2 * i + 1, 2, income.season_data.values))
        income.season_data = income.season_data - 1
        income.season_data = income.season_data.replace(0, 4)
    cap.data[:, 0, :] = 4 * (np.nanmean(revenue, axis=0) - np.nanmean(cost, axis=0)) / cap.data[:, 0, :]
    for i in range(1, 4):
        cap.data[:, i, :] = 1 / cap.data[:, i, :]
    cap.cols = {'enterprise_multiple': 0, 'book_to_price': 1, 'earnings_to_price': 2, 'cash_earnings_to_price': 3}
    start = time.time()
    data = RunOLS(index_name='000300.SH', std_halflife=260, ols_halflife=260,
                  window=1040, cal_std=True, only_alpha=True)
    data()
    long_term_relative_strength = pd.concat(data.daily_std).rolling(11).mean()
    long_term_historical_alpha = pd.concat(data.alpha, axis=1).T.rolling(11).mean()

    dfs = cap.to_dataframes()
    dfs['long_term_relative_strength'] = long_term_relative_strength
    dfs['long_term_historical_alpha'] = long_term_historical_alpha

    df = dfs_to_df(dfs)
    df['earnings_yields'] = (df['enterprise_multiple'] + df['earnings_to_price'] + df['cash_earnings_to_price']) / 3
    df['long_term_reversal'] = (df['long_term_relative_strength'] + df['long_term_historical_alpha']) / 2
    df['value'] = (df['book_to_price'] + df['earnings_yields'] + df['long_term_reversal']) / 3
    df.to_parquet('./value.parquet')
