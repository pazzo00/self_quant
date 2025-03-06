from barra.five_years_ols import FiveYearsOLS
from barra.functions import dfs_to_df
import numpy as np
from report_data_loader import ReportDataLoader


if __name__ == '__main__':
    net_profit = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                            fields=['net_profit'], lag='4Y')
    capital = ReportDataLoader.load_data('../../数据/capital_change.parquet',
                                        fields=['share_total'], lag='4Y', codes=net_profit.codes)
    operating_revenue = ReportDataLoader.load_data('../../数据/report_income.parquet',
                                            fields=['operating_revenue'], lag='4Y', codes=net_profit.codes)
    for i in range(4):
        net_profit.data[:, i, :] /=  capital.data[:, i, :]
        operating_revenue.data[:, i, :] /= capital.data[:, i, :]
    dfs = {'historical_earnings_per_share_growth_rate' :
               FiveYearsOLS(net_profit).run() / np.nanmean(net_profit.data, axis=1),
           'historical_sales_per_share_growth_rate':
               FiveYearsOLS(operating_revenue).run() / np.nanmean(operating_revenue.data, axis=1)}
    df = dfs_to_df(dfs)
    df['growth'] = (df['historical_earnings_per_share_growth_rate'] + df['historical_sales_per_share_growth_rate']) / 2
    df.to_parquet('./growth.parquet')
