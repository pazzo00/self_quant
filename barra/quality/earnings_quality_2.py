import numpy as np

from base_data_loader import BaseDataLoader
from report_data_loader import ReportDataLoader
import pandas as pd
from barra.functions import cal_quarter_data
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
                                            fields=['total_assets'],
                                            lag='0Q')
    cashflow = ReportDataLoader.load_data('../../数据/report_cashflow.parquet',
                                          fields=['net_operate_cash_flow', 'net_invest_cash_flow'], lag='1Q')
    income = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                        fields=['net_profit'], lag='1Q')
    cashflow_flag = cashflow.season_data.values == 1
    income_flag = income.season_data.values == 1

    cfo = cal_quarter_data(cashflow.data, cashflow_flag, 0, 2, cashflow.season_data.values)
    cfi = cal_quarter_data(cashflow.data, cashflow_flag, 1, 2, cashflow.season_data.values)
    ni = cal_quarter_data(income.data, income_flag, 0, 1, income.season_data.values)

    da = BaseDataLoader.load_data('../barra_intermediate_factors.parquet')
    da.data = np.nan_to_num(da.data, 0)

    accr_cf = -1 * (ni - (cfo + cfi) + da.data[:, 0, :]) / balance.data[:, 0, :]
    accr_cf = pd.DataFrame(data=accr_cf, index=cashflow.trade_days, columns=cashflow.codes)
    accr_cf.index.name = 'datetime'
    accr_cf = pd.melt(accr_cf.reset_index(), id_vars='datetime', var_name='code', value_name='accr_cf')

    old_df = pd.read_parquet('./quality.parquet')
    old_df = old_df.drop(columns=['accr_cf'])
    old_df = old_df.merge(accr_cf, how='left', on=['datetime', 'code'])
    old_df.to_parquet('./quality.parquet')
