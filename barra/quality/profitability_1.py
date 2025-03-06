from base_data_loader import BaseDataLoader
from report_data_loader import ReportDataLoader
import pandas as pd
import numpy as np
from barra.functions import cal_quarter_data, dfs_to_df
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # 计算TTM
    balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
                                            fields=['total_assets'], lag='0Q')
    income = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                        fields=['net_profit', 'operating_revenue'], lag='4Q')

    sales, earning = [], []
    for i in range(4):
        flag = income.season_data.values == 1
        earning.append(cal_quarter_data(income.data, flag, 2*i, 2, income.season_data.values))
        sales.append(cal_quarter_data(income.data, flag, 2*i+1, 2, income.season_data.values))
        income.season_data = income.season_data - 1
        income.season_data = income.season_data.replace(0, 4)
    sales_ttm = np.nanmean(sales, axis=0) * 4 / balance.data[:, 0, :]
    earning_ttm = np.nanmean(earning, axis=0) * 4 / balance.data[:, 0, :]
    dfs = {'ato': pd.DataFrame(data=earning_ttm, index=income.trade_days, columns=income.codes),
           'roa': pd.DataFrame(data=sales_ttm, index=income.trade_days, columns=income.codes)}
    df = dfs_to_df(dfs)
    old_df = pd.read_parquet('../barra_factors.parquet')
    old_df = old_df.merge(df, how='left', on=['datetime', 'code'])
    old_df.to_parquet('../barra_factors.parquet')

