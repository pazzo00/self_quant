from base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
from barra.functions import gen_weights
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
        self.daily_std = self.alpha = self.beta = self.hist_sigma = None
        # self.daily_std, self.alpha, self.beta, self.hist_sigma = [], [], [], []

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
        with (mp.Manager() as manager):
            beta, hist_sigma, alpha, daily_std = manager.list(), manager.list(), manager.list(), manager.list()  # 创建共享的列表
            with mp.Pool(processes=4) as pool:
                # 使用 pool.map 调用 ols 函数
                pool.starmap(self.ols, [(i, beta, hist_sigma, alpha, daily_std) for i in range(
                    self.window, len(self.ret) + 1)])

            self.daily_std, self.alpha, self.beta , self.hist_sigma = (list(daily_std), list(alpha),
                                                                       list(beta), list(hist_sigma))

    def ols(self, idx, beta, hist_sigma, alpha, daily_std):
        tmp = self.ret.iloc[idx - self.window:idx]
        if self.cal_std:
            daily_std.append(tmp.ewm(halflife=self.std_halflife).std().iloc[[-1]].drop(columns=[self.index_name]))

        w = gen_weights(self.ols_halflife, self.window).reshape(-1)
        W_full = np.diag(w)
        Y_full = tmp.dropna(axis=1).drop(columns=self.index_name)
        idx_full, Y_full = Y_full.columns, Y_full.values
        X_full = np.c_[np.ones((self.window, 1)), tmp.loc[:, self.index_name].values]
        beta_full = np.linalg.pinv(X_full.T @ W_full @ X_full) @ X_full.T @ W_full @ Y_full
        alpha_full = pd.Series(beta_full[0], index=idx_full, name=tmp.index[-1])
        if not self.only_alpha:
            hist_sigma_full = pd.Series(np.std(Y_full - X_full @ beta_full, axis=0), index=idx_full, name=tmp.index[-1])
            beta_full = pd.Series(beta_full[1], index=idx_full, name=tmp.index[-1])

        beta_lack, hist_sigma_lack, alpha_lack = {}, {}, {}
        for c in set(tmp.columns) - set(idx_full) - {self.index_name}:
            tmp_ = tmp.loc[:, [c, self.index_name]].copy()
            tmp_.loc[:, 'W'] = w
            tmp_ = tmp_.dropna()
            W_lack = np.diag(tmp_['W'])
            if len(tmp_) < 63:
                if not self.only_alpha:
                    beta_lack[c] = np.nan
                    hist_sigma_lack[c] = np.nan
                alpha_lack[c] = np.nan
                continue
            X_lack = np.c_[np.ones(len(tmp_)), tmp_[self.index_name].values]
            Y_lack = tmp_[c].values
            beta_tmp = np.linalg.pinv(X_lack.T @ W_lack @ X_lack) @ X_lack.T @ W_lack @ Y_lack
            if not self.only_alpha:
                hist_sigma_lack[c] = np.std(Y_lack - X_lack @ beta_tmp)
                beta_lack[c] = beta_tmp[1]
            alpha_lack[c] = beta_tmp[0]

        alpha_lack = pd.Series(alpha_lack, name=tmp.index[-1])
        if not self.only_alpha:
            beta_lack = pd.Series(beta_lack, name=tmp.index[-1])
            hist_sigma_lack = pd.Series(hist_sigma_lack, name=tmp.index[-1])
            beta.append(pd.concat([beta_full, beta_lack]).sort_index())
            hist_sigma.append(pd.concat([hist_sigma_full, hist_sigma_lack]).sort_index())
        alpha.append(pd.concat([alpha_full, alpha_lack]).sort_index())
