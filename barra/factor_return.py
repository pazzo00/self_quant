import pandas as pd
from base_data_loader import BaseDataLoader
import numpy as np


class BarraFactorReturn:
    def __init__(self):
        self.barra = BaseDataLoader.load_data('./barra_factors.parquet')
        self.size = BaseDataLoader.load_data('../数据/capital.parquet', codes=self.barra.codes,
                                             fields=['circulating_market_cap'])
        self.ind_data = BaseDataLoader.load_data('../数据/sw_industry.parquet', codes=self.barra.codes,
                                                 fields=['sw_l1_code'])
        self.close = BaseDataLoader.load_data('../数据/stock_bar_1day.parquet', codes=self.barra.codes,
                                              fields=['close'])
        self.ind_numerical = self.ind_unique = None
        self.factor_ret = []
        self.spec_ret = []

    def pre_transformer(self):
        self.close.data = self.close.data[1:] / self.close.data[:-1] - 1
        self.close.trade_days = self.close.trade_days[:-1]

        ind_unique = self.ind_data.data[:, self.ind_data.cols['sw_l1_code'], :].astype('<U32')
        ind_numerical = ind_unique[:, np.newaxis, :]
        self.ind_unique = np.unique(ind_unique[~np.isin(ind_unique, ['None', np.nan, ''])])
        x = []
        for tag in self.ind_unique:
            x.append(np.nan_to_num(ind_numerical == tag, nan=0))
        self.ind_numerical = np.concatenate(x, axis=1)

    def __call__(self):
        self.pre_transformer()
        for i in range(len(self.ind_data.trade_days) - 1):
            self.cal_factor_return(i)
        factor_ret = pd.concat(self.factor_ret, axis=1).T
        factor_ret.index = self.ind_data.trade_days[:-1]
        factor_ret.to_parquet('./barra_factor_return.parquet')
        spec_ret = pd.concat(self.spec_ret, axis=1).T
        spec_ret.index = self.ind_data.trade_days[1:]
        spec_ret.to_parquet('./barra_spec_return.parquet')

    def cal_factor_return(self, idx):
        barra = self.get_data_by_day('barra', idx)
        weight = self.get_data_by_day('size', idx).dropna(axis=0)
        ret = self.get_data_by_day('close', idx)
        barra_no_nan = barra.dropna(axis=0, how='all')
        ret_no_nan = ret.dropna(axis=0)

        ind_data = self.get_data_by_day('ind_data', idx)

        codes = barra_no_nan.index.intersection(ret_no_nan.index.intersection(weight.index))

        w = np.sqrt(weight.reindex(codes))
        w = w / w.sum()
        w = np.diag(w.values.reshape(-1))

        weights = np.sum(weight.reindex(codes).values * ind_data.reindex(codes).values, axis=0)
        weights = pd.Series(data=weights, index=ind_data.columns)
        ind_weights = weights[weights > 0]
        ind_data = ind_data[ind_weights.index.tolist()]

        exposure = barra.join(ind_data)
        exposure.sort_index(axis=1, inplace=True)
        exposure.insert(0, 'country', 1)
        exposure = exposure.fillna(0)
        col = exposure.columns
        x = exposure.reindex(codes).values.astype(np.float64)

        # 计算c矩阵
        c = np.eye(x.shape[1] - 1, x.shape[1] - 1)
        ind_weights = ind_weights / ind_weights.iloc[-1]
        ind_weights = ind_weights.values[:-1]
        new_line = np.zeros(x.shape[1] - 1)
        new_line[1:len(ind_weights) + 1] = ind_weights
        c = np.insert(c, len(ind_weights) + 1, new_line, axis=0)

        omega = c @ np.linalg.pinv(c.T @ x.T @ w @ x @ c) @ c.T @ x.T @ w
        factor_ret = omega @ ret_no_nan.reindex(codes)
        factor_ret.index = col
        self.factor_ret.append(factor_ret['close'])

        # 计算特质性收益率
        spec_ret = ret - exposure.values @ factor_ret.values.reshape(-1, 1)
        self.spec_ret.append(spec_ret['close'])

    def get_data_by_day(self, attr_name, idx):
        data = self.__getattribute__(attr_name)
        if attr_name == 'ind_data':
            return pd.DataFrame(data=self.ind_numerical[idx].T, index=data.codes, columns=self.ind_unique)
        df = pd.DataFrame(data=data.data[idx].T, index=data.codes, columns=list(data.cols.keys()))
        return df


if __name__ == '__main__':
    BarraFactorReturn()()

