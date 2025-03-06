import pandas as pd
import numpy as np


def dfs_to_df(dfs):
    for key in dfs:
        dfs[key].index.name = 'datetime'
        dfs[key].reset_index(inplace=True)
        dfs[key] = pd.melt(dfs[key], id_vars='datetime', var_name='code', value_name=key)
        dfs[key].set_index(['datetime', 'code'], inplace=True)
    df = pd.concat(list(dfs.values()), axis=1)
    df.reset_index(inplace=True)
    return df


def cal_quarter_data(data, flag, idx, gap, season):
    tmp = data[:, idx, :] - data[:, idx+gap, :]
    tmp_flag = np.isnan(tmp)
    tmp[tmp_flag] = data[:, idx, :][tmp_flag] / season[tmp_flag]
    tmp[flag] = data[:, idx, :][flag]
    return tmp


def gen_weights(halflife, window):
    w = [0.5 ** (1 / halflife)]
    for i in range(1, window):
        w.append(w[-1] * w[0])
    return np.array(w).reshape(-1, 1)


if __name__ == '__main__':
    df = pd.read_parquet('./barra_factors.parquet')


from base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
import multiprocessing as mp


class RunOLS:
    def __init__(self, start=None, end=None, index_name='000300.SH', std_halflife=5, ols_halflife=5,
                 window=252, cal_std=True, only_alpha=False):
        self.index_name = index_name
        self.std_halflife = std_halflife
        self.ols_halflife = ols_halflife
        self.window = window
        self.cal_std = cal_std
        self.only_alpha = only_alpha
        self.ret = RunOLS.get_ret_data(start, end, index_name)
        self.daily_std, self.alpha, self.beta , self.hist_sigma = [], [], [], []

    @staticmethod
    def get_ret_data(start, end, index_name):
        data = BaseDataLoader.load_data('../../数据/stock_bar_1day.parquet', fields=['close'],
                                        start=start, end=end)
        end = data.trade_days[-1]
        data.data = data.data[1:] / data.data[:-1] - 1
        data.trade_days = data.trade_days[1:]

        index_data = BaseDataLoader.load_data('../../数据/index_bar_1day.parquet', fields=['close'],
                                              start=start, end=end, codes=[index_name])
        index_data.data = index_data.data[1:] / index_data.data[:-1] - 1
        index_data.trade_days = index_data.trade_days[1:]

        start = data.trade_days.index(index_data.trade_days[0])
        data.data = data.data[start:]
        data.trade_days = data.trade_days[start:]
        return pd.concat([data.to_dataframe('close'), index_data.to_dataframe('close')], axis=1)

    def __call__(self):
        with mp.Pool(processes=4) as pool:
            pool.map(self.ols, [253])


    def ols(self, idx):
        tmp = self.ret.iloc[idx - self.window:idx]
        self.daily_std.append(tmp.ewm(halflife=self.std_halflife).std().iloc[[-1]].drop(columns=[self.index_name]))



if __name__ == '__main__':
    df = pd.read_parquet('./barra_factors.parquet')
    test = 1
