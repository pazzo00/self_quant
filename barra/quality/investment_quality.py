from functools import lru_cache
import multiprocessing as mp
import pandas as pd
from barra.functions import dfs_to_df
from report_data_loader import ReportDataLoader
from barra.five_years_ols import FiveYearsOLS
import numpy as np
from base_data_loader import BaseDataLoader
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
                                         fields=['longterm_account_payable', 'specific_account_payable'], lag='4Y')
    cashflow = ReportDataLoader.load_data('../../数据/report_cashflow.parquet',
                                          fields=['fix_intan_other_asset_acqui_cash'], lag='4Y')
    for i in range(5):
        cashflow.data[:, i, :] -= np.nansum(balance.data[:, 2*i:2*i+1, :], axis=1)
    capital_expenditure_growth = FiveYearsOLS(cashflow).run()

    # balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
    #                                      fields=['total_assets'], lag='4Y')
    # capital = ReportDataLoader.load_data('../../数据/capital_change.parquet',
    #                                     fields=['share_total'], lag='4Y')

    # balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
    #                                      fields=['longterm_account_payable', 'specific_account_payable'], lag='4Y')
    # cashflow = ReportDataLoader.load_data('../../数据/report_cashflow.parquet',
    #                                       fields=['fix_intan_other_asset_acqui_cash'], lag='4Y')
    # for i in range(5):
    #     cashflow.data[:, i, :] -= np.nansum(balance.data[:, 2*i:2*i+1, :], axis=1)
    # capital_expenditure_growth = FiveYearsOLS(cashflow).run()
    #
    # capital = ReportDataLoader.load_data('../../数据/capital_change.parquet',
    #                                     fields=['share_total'], lag='4Y')
    # issuance_growth = FiveYearsOLS(capital).run()
    #
    # balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
    #                                      fields=['total_assets'], lag='4Y')
    # total_assets_growth_rate = FiveYearsOLS(balance).run()
    #
    # dfs = {'capital_expenditure_growth': capital_expenditure_growth,
    #        'issuance_growth': issuance_growth,
    #        'total_assets_growth_rate': total_assets_growth_rate}
    # df = dfs_to_df(dfs)
    # df['investment_quality'] = (df['capital_expenditure_growth'] + df['issuance_growth']
    #                             + df['total_assets_growth_rate']) / 3
    # old_df = pd.read_parquet('../barra_factors.parquet')
    # old_df = old_df.merge(df, how='left', on=['datetime', 'code'])
    # old_df['quality'] = (old_df['leverage'] + old_df['earnings_variability'] + old_df['earnings_quality'] +
    #                      old_df['profitability'] + old_df['investment_quality']) / 5
    # old_df.to_parquet('../barra_factors.parquet')

    data = BaseDataLoader.load_data('./quality.parquet')
    data = data.to_dataframes()
    for key in data:
        data[key] = data[key].replace([np.inf, -np.inf], 0)
        data[key] = (data[key] - data[key].mean(axis=1).values.reshape(-1, 1)) / data[key].std(axis=1).values.reshape(-1, 1)
    data = dfs_to_df(data)
    data['earnings_quality'] = (data['accr_cf'] + data['accr_bs']) / 2
    data['leverage'] = (data['market_leverage'] + data['book_leverage'] + data['debt_to_asset_ratio']) / 3
    data['earnings_variability'] = (data['variation_in_sales'] +
                                  data['variation_in_earnings'] + data['variation_in_cashflows']) / 3
    data['profitability'] = (data['ato'] + data['roa'] + data['gp'] + data['gpm']) / 4
    test = pd.pivot_table(data, index='datetime', columns='code',values='profitability')
    data['investment_quality'] = (data['capital_expenditure_growth'] + data['issuance_growth']
                                  + data['total_assets_growth_rate']) / 3
    data['quality'] = (data['leverage'] + data['earnings_quality'] + data['earnings_variability']
                       + data['profitability'] + data['investment_quality']) / 5
    data = data[['code', 'datetime', 'leverage', 'earnings_variability', 'earnings_quality', 'profitability',
                 'investment_quality', 'quality']]
    old_df = pd.read_parquet('./quality.parquet')
    old_df = old_df.drop(columns=['leverage', 'earnings_variability', 'earnings_quality', 'profitability', 'investment_quality', 'quality'])
    old_df = old_df.merge(data, how='left', on=['datetime', 'code'])
    old_df.to_parquet('./quality.parquet')

    df1 = pd.read_parquet('./quality.parquet')
    df2 = pd.read_parquet('quality.parquet')
    test = 1
