from base_data_loader import BaseDataLoader
from report_data_loader import ReportDataLoader
import numpy as np
import pandas as pd
from barra.functions import cal_quarter_data
import multiprocessing as mp
from functools import partial
import warnings
warnings.filterwarnings('ignore')




if __name__ == '__main__':
    # # 折旧和摊销
    # cashflow = ReportDataLoader.load_data('../../数据/report_cashflow.parquet',
    #                                       fields=['defferred_expense_amortization',
    #                                               'fixed_assets_depreciation', 'intangible_assets_amortization'],
    #                                       lag='1Q')
    # flag = cashflow.season_data.values == 1
    # da_list = []
    # for i in range(3):
    #     da_list.append(cal_quarter_data(cashflow.data, flag, i, 3, cashflow.season_data.values))
    #
    # da = np.nansum(da_list, axis=0)
    # da[np.all(np.isnan(da_list), axis=0)] = np.nan
    # da = pd.DataFrame(data=da, index=cashflow.trade_days, columns=cashflow.codes)
    # da.index.name = 'datetime'
    # da = pd.melt(da.reset_index(), id_vars='datetime', var_name='code', value_name='da')
    # da.to_parquet('../barra_intermediate_factors.parquet')
    # df = pd.read_parquet('../barra_intermediate_factors.parquet')

    da = BaseDataLoader.load_data('../barra_intermediate_factors.parquet')
    da.data = np.nan_to_num(da.data, 0)
    balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
                                            fields=['total_assets', 'total_liability', 'shortterm_loan',
                                                    'non_current_liability_in_one_year', 'total_non_current_liability'],
                                            lag='1Q', codes=da.codes)
    cashflow = ReportDataLoader.load_data('../../数据/report_cashflow.parquet',
                                          fields=['cash_and_equivalents_at_end'], lag='1Q', codes=da.codes)
    noa_lag0 = ((balance.data[:, 0, :] - cashflow.data[:, 0, :]) -
           (balance.data[:, 1, :] - np.nansum(balance.data[:, 2:5, :])))
    noa_lag1 = ((balance.data[:, 5, :] - cashflow.data[:, 1, :]) -
           (balance.data[:, 6, :] - np.nansum(balance.data[:, 7:, :])))
    accr_bs = -1 * (noa_lag0 - noa_lag1 - da.data[:, 0, :]) / balance.data[:, 0, :]
    accr_bs = pd.DataFrame(data=accr_bs, index=cashflow.trade_days, columns=cashflow.codes)
    accr_bs.index.name = 'datetime'
    accr_bs = pd.melt(accr_bs.reset_index(), id_vars='datetime', var_name='code', value_name='accr_bs')

    old_df = pd.read_parquet('./quality.parquet')
    old_df = old_df.drop(columns=['accr_bs'])
    old_df = old_df.merge(accr_bs, how='left', on=['datetime', 'code'])
    old_df.to_parquet('./quality.parquet')
