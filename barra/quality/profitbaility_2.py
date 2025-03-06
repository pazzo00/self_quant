from base_data_loader import BaseDataLoader
from report_data_loader import ReportDataLoader
import pandas as pd
import numpy as np
from barra.functions import cal_quarter_data, dfs_to_df
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
                                            fields=['total_assets'], lag='0Y', start='2018-01-01')
    income = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                        fields=['total_operating_revenue', 'total_operating_cost'], lag='0Y', start='2018-01-01', codes=balance.codes)
    test = income.to_dataframes()
    diff = income.data[:, 0, :] - income.data[:, 1, :]
    gp = diff / balance.data[:, 0, :]
    gpm = diff / income.data[:, 0, :]

    dfs = {'gp': pd.DataFrame(data=gp, index=balance.trade_days, columns=balance.codes),
           'gpm': pd.DataFrame(data=gpm, index=balance.trade_days, columns=balance.codes)}
    df = dfs_to_df(dfs)
    old_df = pd.read_parquet('../barra_factors.parquet')
    old_df = old_df.merge(df, how='left', on=['datetime', 'code'])
    old_df['profitability'] = (old_df['ato'] + old_df['roa'] + old_df['gp'] + old_df['gpm']) / 4
    old_df.to_parquet('../barra_factors.parquet')


