from base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from barra.functions import dfs_to_df



if __name__ == '__main__':
    data = BaseDataLoader.load_data('./liquidity.parquet')
    data.data[np.isinf(data.data)] = np.nan
    for i in range(data.data.shape[1]):
        data.data[:, i, :] = (data.data[:, i, :] - np.nanmean(data.data[:, i, :], axis=1)[:, np.newaxis]) /np.nanstd(data.data[:, i, :], axis=1)[:, np.newaxis]
    data.data[:, 4, :] = np.sum(data.data[:, 0:4, :], axis=1)
    data = data.to_dataframe('liquidity')
    data = dfs_to_df({'liquidity':data})
    df = pd.read_parquet('./liquidity.parquet')
    df = df.drop(columns='liquidity')
    df = df.merge(data, how='left', on=['datetime', 'code'])
    df.to_parquet('./liquidity.parquet')
    # data = BaseDataLoader.load_data('../../数据/capital.parquet', fields=['turnover_ratio'])
    # data = data.to_dataframe('turnover_ratio')
    # res = {}
    # res['monthly_share_turnover'] = np.log(data.rolling(21).sum())
    # res['quarterly_share_turnover'] = np.log(data.rolling(63).sum() / 3)
    # res['annual_share_turnover'] = np.log(data.rolling(252).sum() / 12)
    # annualized_traded_value_ration = []
    # for i in tqdm(range(252, len(data) + 1)):
    #     tmp = data.iloc[i-252:i]
    #     tmp = tmp.ewm(halflife=63).mean()
    #     annualized_traded_value_ration.append(tmp.iloc[[-1]])
    # res['annualized_traded_value_ration'] = pd.concat(annualized_traded_value_ration)
    # res = dfs_to_df(res)
    # res['liquidity'] = res.eval(
    #     '(monthly_share_turnover + quarterly_share_turnover + annual_share_turnover + annualized_traded_value_ration) / 4')
    #
    # df = pd.read_parquet('../barra_factors.parquet')
    # df = df.merge(res, how='left', on=['datetime', 'code'])
    # res.to_parquet('./liquidity.parquet')
