import pandas as pd
import numpy as np
from tqdm import tqdm
from base_data_loader import BaseDataLoader
import warnings
warnings.filterwarnings('ignore')


class SpecRisk:
    def __init__(self):
        self.spec_ret = BaseDataLoader.load_data('./barra_spec_return.parquet', start='2013-01-01')
        self.market_cap = BaseDataLoader.load_data('../数据/capital.parquet', start='2013-01-01',
                                                   end=self.spec_ret.trade_days[-1], codes=self.spec_ret.codes)
        self.exposure = BaseDataLoader.load_data('./barra_factors.parquet', start='2013-01-01',
                                                   end=self.spec_ret.trade_days[-1], codes=self.spec_ret.codes)

    def __call__(self):
        res = []
        for i in tqdm(range(252, len(self.spec_ret.trade_days))):
            res.append(self.cal_spec_risk(i))
        res = pd.concat(res, axis=1).T
        res.to_parquet('./barra_spec_risk.parquet')

    def cal_spec_risk(self, idx):
        df = self.spec_ret.get_window_df_by_idx('spec_ret', 252, idx)
        f_raw = self.cal_f_raw(df)
        f_nw = f_raw + self.cal_newey_west(df)

        # 结构化调整
        market_cap = self.market_cap.get_window_df_by_idx('market_cap', 1, idx).iloc[0]
        market_cap.name = 'market_cap'
        f_n = self.struct(f_nw, df, market_cap, idx)

        # 贝叶斯调整
        f_sh = self.bayes(f_n, market_cap)

        # 波动率偏误调整
        f_vra = self.cal_f_vra(df, f_sh, market_cap)
        f_vra = f_vra.reindex(self.spec_ret.codes)
        f_vra.name = self.spec_ret.trade_days[idx-1]
        return f_vra

    @staticmethod
    def cal_f_raw(df):
        mu = df.ewm(halflife=90).mean().iloc[-1]
        df = df.sub(mu, axis=1)
        df = np.square(df)
        res = df.ewm(halflife=90).mean().iloc[-1]
        res.index.name = 'code'
        return res

    @staticmethod
    def cal_newey_west(df):
        omega = 0
        for i in range(1, 6):
            tmp = df.shift(i)
            tmp = tmp * df
            tmp = 2 * tmp.ewm(halflife=90).mean().iloc[-1]
            omega += (1 - i / 6) * tmp
        omega.index.name = 'code'
        return omega

    def struct(self, f_nw, spret, market_cap, idx):
        v_n = (spret.count(axis=0) - 60) / 120
        v_n = v_n.clip(lower=0, upper=1)

        q = spret.quantile([0.75, 0.25])
        sigma = (q.iloc[0] - q.iloc[1]) / 1.35
        z_n = np.exp(1 - np.abs(spret.std(axis=0) / sigma - 1))
        z_n = z_n.clip(upper=1)

        gama_n = v_n * z_n

        b_k = self.cal_b_k(gama_n, f_nw, market_cap, idx)
        x_nk = self.gen_x_nk(self.spec_ret.codes, idx)
        sigma_str = pd.Series(data=1.05 * np.exp(x_nk @ b_k), index=self.spec_ret.codes)
        return gama_n * f_nw + (1 - gama_n) * sigma_str

    def cal_b_k(self, gama_n, f_nw, market_cap, idx):
        codes = gama_n[gama_n == 1].index.tolist()
        y = f_nw[codes]
        y.name = 'f_nw'
        data = [y[y > 0], market_cap[codes]]

        for col in self.exposure.cols:
            tmp = self.exposure.get_window_df_by_idx(col, 1, idx).iloc[0]
            tmp.name = col
            data.append(tmp)
        data = pd.concat(data, axis=1)
        data = data.dropna()
        x_nk = data.drop(columns=['f_nw', 'market_cap']).values.astype('float64')
        w = np.diag(data['market_cap'].values.astype('float64'))
        y = np.log(data['f_nw'].values.astype('float64'))
        b_k = np.linalg.pinv(x_nk.T @ w @ x_nk) @ x_nk.T @ w @ y
        return b_k

    def gen_x_nk(self, code, idx):
        codes = self.exposure.codes.tolist()
        code_index = [codes.index(_) for _ in code]
        x_nk = self.exposure.data[idx - 1, :, np.array(code_index)]
        return x_nk

    def bayes(self, f_n, market_cap):
        market_cap.sort_values(inplace=True)
        f_sh = []
        n = len(market_cap) // 10
        for i in range(1, 10):
            s_n = market_cap.iloc[(i - 1) * n:i * n]
            f_sh.append(self.cal_f_sh(s_n, f_n))
        f_sh.append(self.cal_f_sh(market_cap.iloc[9 * n:], f_n))
        return pd.concat(f_sh)

    @staticmethod
    def cal_f_sh(s_n, f_n):
        w_n = s_n / s_n.sum()

        sigma = f_n[w_n.index]
        sigma_n = (sigma * w_n).sum()

        diff = sigma - sigma_n
        delta_sigma_n = np.square(diff).mean()

        v_n = np.abs(diff)
        v_n = v_n / (delta_sigma_n + v_n)

        f_sh = v_n * sigma_n + (1 - v_n) * sigma
        return f_sh

    @staticmethod
    def cal_f_vra(df, f_sh, market_cap):
        sigma = df.ewm(halflife=90).std().iloc[-1]
        b_s = np.square(df.div(sigma, axis=1))
        w = market_cap / market_cap.sum()
        b_s = (b_s * w).sum(axis=1)
        lambda_s = b_s.ewm(halflife=42).sum().iloc[-1]
        return np.sqrt(lambda_s) * f_sh


if __name__ == '__main__':
    SpecRisk()()
