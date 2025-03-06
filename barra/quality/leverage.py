from base_data_loader import BaseDataLoader
from report_data_loader import ReportDataLoader
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    cap = BaseDataLoader.load_data('../../数据/capital.parquet', fields=['market_cap', 'pb_ratio'])
    balance = ReportDataLoader.load_data('../../数据/report_balance.parquet',
                                         fields=['total_assets', 'total_liability', 'total_non_current_liability'],
                                         lag='0Y', codes=cap.codes)
    cap.data[:, 1, :] = 1 / cap.data[:, 1, :]
    mlev = (cap.data[:, 0, :] + balance.data[:, 2, :]) / cap.data[:, 0, :]
    blev = (cap.data[:, 1, :] + balance.data[:, 2, :]) / cap.data[:, 1, :]
    dtoa = balance.data[:, 1, :] / balance.data[:, 0, :]
    dfs = {
        'market_leverage': pd.DataFrame(data=mlev, index=cap.trade_days, columns=cap.codes),
        'book_leverage': pd.DataFrame(data=blev, index=cap.trade_days, columns=cap.codes),
        'debt_to_asset_ratio': pd.DataFrame(data=dtoa, index=cap.trade_days, columns=cap.codes)
    }

    for key in dfs:
        dfs[key].index.name = 'datetime'
        dfs[key].reset_index(inplace=True)
        dfs[key] = pd.melt(dfs[key], id_vars='datetime', var_name='code', value_name=key)
        dfs[key].set_index(['datetime', 'code'], inplace=True)
    df = pd.concat(list(dfs.values()), axis=1)
    df.reset_index(inplace=True)
    df['leverage'] = (df['market_leverage'] + df['book_leverage'] + df['debt_to_asset_ratio']) / 3

    old_df = pd.read_parquet('../barra_factors.parquet')
    old_df = old_df.merge(df, how='left', on=['datetime', 'code'])
    old_df.to_parquet('../barra_factors.parquet')


