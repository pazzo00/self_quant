import pandas as pd
from barra.functions import dfs_to_df
import numpy as np
from datetime import timedelta
from base_data_loader import BaseDataLoader


if __name__ == '__main__':
    close = BaseDataLoader.load_data('../../数据/stock_bar_1day.parquet', fields=['close'])
    cash_div_ratio = BaseDataLoader.load_data('../../数据/cash_div_allo.parquet', fields=['cash_dividend_ratio'],
                                              codes=close.codes, start=close.trade_days[0], end=close.trade_days[-1])

    cash_div_ratio = cash_div_ratio.to_dataframe('cash_dividend_ratio')
    cash_div_ratio.index = pd.to_datetime(cash_div_ratio.index) + timedelta(hours=15)
    cash_div_ratio = cash_div_ratio.reindex(close.trade_days)
    cash_div_ratio = cash_div_ratio.fillna(0).rolling(252).sum()

    close = close.to_dataframe('close')
    close.sort_index(inplace=True)
    flag = close.index.month
    flag = np.append(flag[:-1] != flag[1:], False)
    flag = np.where(flag, 1, np.nan)
    close = (close * flag.reshape(-1, 1)).ffill()
    df = dfs_to_df({'dividend_yield': cash_div_ratio / close})
    df.to_parquet('./dividend_yield.parquet')
