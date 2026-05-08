import seaborn as sns
from matplotlib import pyplot as plt
from base_data_loader import BaseDataLoader
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CorrAnalysis:
    def __init__(self, rtn_pth, pth, start, end, fields=None, mode='norm', weight_data=None):
        if fields is None:
            data = BaseDataLoader.load_data(pth, start=start, end=end)
        else:
            data = BaseDataLoader.load_data(pth, start=start, end=end, fields=fields)
        self.lens = len(data.trade_days)
        self.data = data.to_dataframes()
        self.start = start
        rtn = BaseDataLoader.load_data(rtn_pth, fields=['close', 'factor'], codes=data.codes).to_dataframes()
        rtn = ((rtn['close'] * rtn['factor']).pct_change(1)).shift(-1)
        self.rtn = rtn[rtn.index >= pd.to_datetime(start)]
        self.mode = mode
        self.weight_data = weight_data
        if weight_data is not None:
            self.weight_data = weight_data[weight_data.index >= pd.to_datetime(start)]
            self.weight_data = self.weight_data.reindex(columns=data.codes)
        self.cap = self.weight_mode = None

    def handle_signal(self, mode):
        for key in self.data:
            if mode == 'mean':
                self.data[key] = self.data[key].rolling(21, min_periods=2).mean()
            if mode == 'av_diff':
                self.data[key] = self.data[key] - self.data[key].rolling(21, min_periods=2).mean()
            if mode == 'rank_mean':
                self.data[key] = self.data[key].rank(pct=True, axis=1).rolling(21).mean()
            if mode == 'abs_mean':
                self.data[key] = self.data[key].abs().rolling(21).mean()
            if mode == 'std':
                self.data[key] = self.data[key].rolling(21, min_periods=2).std()
            if mode == 'mean/std':
                self.data[key] = self.data[key].rolling(21, min_periods=2).mean() / self.data[key].rolling(21, min_periods=2).std()
            if mode == 'rank_std':
                self.data[key] = self.data[key].rank(pct=True, axis=1).rolling(21).std()
            if mode == 'mean+std':
                self.data[key] = self.data[key].rolling(21).std() + self.data[key].rolling(21).mean()
            if mode == 'anomaly':
                self.data[key] = self.data[key].rolling(21).mean() / self.data[key].rolling(252).mean()
            if mode == 'rank_anomaly':
                self.data[key] = self.data[key].rank(pct=True, axis=1).rolling(21).mean() / self.data[key].rank(pct=True, axis=1).rolling(
                    252).mean()
            if mode == 'zscore_std':
                self.data[key] = (self.data[key] - self.data[key].mean(axis=1).values.reshape(-1, 1)) / self.data[key].std(axis=1).values.reshape(-1, 1)
                self.data[key] = self.data[key].rolling(21, min_periods=2).std()
            if mode == 'zscore_mean':
                self.data[key] = (self.data[key] - self.data[key].mean(axis=1).values.reshape(-1, 1)) / self.data[key].std(axis=1).values.reshape(-1, 1)
                self.data[key] = self.data[key].rolling(21, min_periods=2).std()
            if mode == 'rank(std)+rank(mean)':
                mu = self.data[key].rolling(21, min_periods=2).mean().rank(pct=True, axis=1)
                sigma = self.data[key].rolling(21, min_periods=2).std().rank(pct=True, axis=1)
                self.data[key] = mu + sigma
            if mode == 'abs':
                self.data[key] = self.data[key].abs()

    def __call__(self, mode=None, weight_mode=None):
        if self.mode == 'premium' and mode is not None:
            res = self.premium_corr_analysis(mode, weight_mode)
        else:
            res = self.norm_corr_analysis('spearman')

        sns.heatmap(res, annot=True, cmap='coolwarm', fmt='.2f')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig('./heatmap.png')

    def norm_corr_analysis(self, method='spearman'):
        res = {key: [] for key in self.data}
        for key in self.data:
            for _ in self.data:
                corr = self.data[key].corrwith(self.data[_], axis=1, method=method)
                res[key].append(np.nanmean(corr))
        return pd.DataFrame(res, index=list(res.keys()))

    def premium_corr_analysis(self, mode, weight_mode):
        if mode == 'wq':
            premium = self.cal_premium_wq()
        elif mode == 'fama':
            premium = self.cal_premium_fama(weight_mode)
        else:
            premium = self.cal_premium_regression()

        return premium.corr()

    def cal_premium_regression(self):
        res = Parallel(n_jobs=16, verbose=10)(
            delayed(self.__cal_premium_regression__)(idx)
            for idx in range(self.lens))
        res = pd.DataFrame(res, columns=list(self.data.keys()))
        return res

    def __cal_premium_regression__(self, idx):
        res = []
        rtn = self.rtn.iloc[idx]
        if self.weight_data is not None:
            wgt = self.weight_data.iloc[idx]
        else:
            wgt = pd.Series(1, index=self.rtn.columns)
        for key in self.data:
            tmp = pd.concat([self.data[key].iloc[idx], rtn, wgt], axis=1).dropna()
            x = tmp.iloc[:, 0].values.reshape(-1, 1)
            w = np.diag(tmp.iloc[:, -1].values)
            y = tmp.iloc[:, 1].values.reshape(-1, 1)
            beta = np.linalg.pinv(x.T @ w @ x) @ x.T @ w @ y
            res.append(beta[0][0])
        return res

    def cal_premium_fama(self, pth, weight_mode):
        self.cap = BaseDataLoader.load_data(pth,
                                       fields=['market_cap'], start=self.start
                                       ).to_dataframe('market_cap')
        self.weight_mode = weight_mode

        res = Parallel(n_jobs=16, verbose=10)(
            delayed(self.__cal_premium_fama__)(idx)
            for idx in range(self.lens))
        res = pd.DataFrame(res, columns=list(self.data.keys()))
        return res

    def __cal_premium_fama__(self, idx):
        rtn = self.rtn.iloc[idx]
        cap = self.cap.iloc[idx]

        res = []
        for key in self.data:
            tmp = pd.concat([self.data[key].iloc[idx], rtn, cap], axis=1).dropna()
            tmp.columns = ['factor', 'rtn', 'market_cap']
            tmp = tmp.sort_values(by='market_cap')
            l1 = len(tmp) // 2
            s = tmp.iloc[:l1, :].sort_values(by='factor')
            b = tmp.iloc[l1:, :].sort_values(by='factor')

            start = 0
            s_premium, b_premium = [], []
            for q in [0.3, 0.7, 1]:
                end = int(l1 * q)
                s_premium.append(self.__cal_rtn__(s.iloc[start:end]))
                b_premium.append(self.__cal_rtn__(b.iloc[start:end]))
                start = end
            hml = (s_premium[-1] + b_premium[-1]) * 0.5 - (s_premium[0] + b_premium[0]) * 0.5
            res.append(hml)
        return res

    def __cal_rtn__(self, data):
        if self.weight_mode != 'equal':
            data['market_cap'] = data['market_cap'] / data['market_cap'].sum()
            return (data['rtn'] * data['market_cap']).sum()
        return data['rtn'].mean()

    def cal_premium_wq(self):
        print('premium_wq')
        res = Parallel(n_jobs=16, verbose=10)(
            delayed(self.__cal_premium_wq__)(idx)
            for idx in range(self.lens))
        res = pd.DataFrame(res, columns=list(self.data.keys()))
        return res

    def __cal_premium_wq__(self, idx):
        res = []
        rtn = self.rtn.iloc[idx]
        for key in self.data:
            tmp = pd.concat([self.data[key].iloc[idx], rtn], axis=1).dropna()
            tmp.iloc[:, 0] = tmp.iloc[:, 0] - tmp.iloc[:, 0].mean()
            tmp.iloc[:, 0] = tmp.iloc[:, 0] / tmp.iloc[:, 0].abs().sum()
            res.append(np.sum(tmp.iloc[:, 0] * tmp.iloc[:, 1]))
        return res


if __name__ == '__main__':
    CorrAnalysis('./data/stock_bar_1day.parquet', './hf_factors/修正振幅/modify_amp.parquet',
                     '2018-01-01', '2025-01-01', mode='premium')('wq')
