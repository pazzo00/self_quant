from functools import lru_cache
import multiprocessing as mp
import pandas as pd
import numpy as np
import warnings
from barra.functions import dfs_to_df
from base_data_loader import BaseDataLoader

warnings.filterwarnings('ignore')


class FiveYearsOLS:
    def __init__(self, data):
        self.data = data

    def run(self):
        with mp.Pool(4) as p:
            res = p.map(self.run_ols, list(range(len(self.data.data))))
        res = pd.concat(res, axis=1).T
        res.index = self.data.trade_days
        return res

    def run_ols(self, idx):
        ols_data = pd.DataFrame(data=self.data.data[idx, :, :], columns=self.data.codes)
        ols_data = ols_data.to_json(orient='records')
        return self.__run_ols__(ols_data)

    @lru_cache
    def __run_ols__(self, ols_data):
        ols_data = pd.read_json(ols_data)
        mu = ols_data.mean(axis=0)
        ols_data['t'] = [1, 2, 3, 4, 5]
        Y_full = ols_data.dropna(axis=1).drop(columns='t')
        idx_full, Y_full = Y_full.columns, Y_full.values
        X_full = ols_data.loc[:, 't'].values.reshape(-1, 1)
        beta_full = np.linalg.pinv(X_full.T @ X_full) @ X_full.T @ Y_full
        beta_full = pd.Series(beta_full[0], index=idx_full)

        beta_lack, hist_sigma_lack, alpha_lack = {}, {}, {}
        for c in set(ols_data.columns) - set(idx_full) - set('t'):
            tmp_ = ols_data.loc[:, [c, 't']].copy()
            tmp_ = tmp_.dropna()
            if len(tmp_) < 3:
                beta_lack[c] = np.nan
                continue
            X_lack = tmp_['t'].values.reshape(-1, 1)
            Y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(X_lack.T @ X_lack) @ X_lack.T @ Y_lack
            beta_lack[c] = beta_tmp[0]
        beta_lack = pd.Series(beta_lack)
        beta = -1 * pd.concat([beta_full, beta_lack]).sort_index() / mu
        return beta


if __name__ == '__main__':
    # df = None
    # for key in ['dividend_yield', 'growth', 'liquidity', 'momentum', 'quality', 'size', 'value', 'volatility']:
    #     tmp = pd.read_parquet('./{}/{}.parquet'.format(key, key), columns=['datetime', 'code'] + [key])
    #     if df is None:
    #         df = tmp
    #     else:
    #         df = df.merge(tmp, how='outer', on=['datetime', 'code'])
    # df.to_parquet('./barra_factors.parquet')
    data = BaseDataLoader.load_data('./liquidity/liquidity.parquet', fields=['liquidity'])
    mu = np.nanmean(data.data, axis=2)[:, :, np.newaxis]
    sigma = np.nanstd(data.data, axis=2)[:, :, np.newaxis]
    data.data = (data.data - mu) / sigma
    data = data.to_dataframes()
    data = dfs_to_df(data)
    df = pd.read_parquet('./barra_factors.parquet')
    df = df.drop(columns=['liquidity'])
    df = df.merge(data, how='left', on=['code', 'datetime'])
    df.to_parquet('./barra_factors.parquet')
