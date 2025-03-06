from report_data_loader import ReportDataLoader
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    income = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                        fields=['operating_revenue', 'net_profit'],
                                        lag='4Y')
    cashflow = ReportDataLoader.load_data('../../数据/report_cashflow.parquet',
                                          fields=['cash_equivalent_increase'],
                                          lag='4Y')
    var_in_sales = np.nanmean(income.data[:, 0:5, :], axis=1) / (np.nanstd(income.data[:, 0:5, :], axis=1))
    var_in_earnings = np.nanmean(income.data[:, 5:, :], axis=1) / (np.nanstd(income.data[:, 5:, :], axis=1))
    var_in_cashflow = np.nanmean(cashflow.data, axis=1) / (np.nanstd(cashflow.data, axis=1))
    dfs = {
        'variation_in_sales': pd.DataFrame(data=var_in_sales, index=income.trade_days, columns=income.codes),
        'variation_in_earnings': pd.DataFrame(data=var_in_earnings, index=income.trade_days, columns=income.codes),
        'variation_in_cashflows': pd.DataFrame(data=var_in_cashflow, index=income.trade_days, columns=income.codes)
    }

    for key in dfs:
        dfs[key].replace([np.inf, -np.inf], np.nan, inplace=True)
        dfs[key].index.name = 'datetime'
        dfs[key].reset_index(inplace=True)
        dfs[key] = pd.melt(dfs[key], id_vars='datetime', var_name='code', value_name=key)
        dfs[key].set_index(['datetime', 'code'], inplace=True)
    df = pd.concat(list(dfs.values()), axis=1)
    df.reset_index(inplace=True)
    # df['earnings_variability'] = (df['variation_in_sales'] +
    #                               df['variation_in_earnings'] + df['variation_in_cashflows']) / 3

    old_df = pd.read_parquet('./quality.parquet')
    old_df = old_df.drop(columns=['variation_in_sales', 'variation_in_earnings', 'variation_in_cashflows'])
    old_df = old_df.merge(df, how='left', on=['datetime', 'code'])

    old_df.to_parquet('./quality_new.parquet')


