import pandas as pd
import numpy as np
import multiprocessing as mp


class BarraCovariance:
    def __init__(self):
        self.factor_ret = 100 * pd.read_parquet('./barra_factor_return.parquet',
                                          columns=['dividend_yield', 'growth', 'liquidity', 'momentum', 'quality',
                                                   'size', 'value', 'volatility'])
        self.factor_ret = self.factor_ret[self.factor_ret.index >= pd.to_datetime('2013-01-01')]
        self.trade_days = self.factor_ret.index.tolist()

    def __call__(self):
        with mp.Pool(16) as p:
            res = p.map(self.cal_covariance, list(range(252, len(self.trade_days))))
        res = pd.concat(res)
        res.to_parquet('./barra_covariance.parquet')

    def cal_covariance(self, idx):
        df = self.factor_ret.iloc[idx - 252:idx]
        f_raw = self.cal_f_raw(df)
        f_nw = 21 * (f_raw + self.cal_newey_west(df))
        f_eigen = self.cal_f_eigen(f_nw)
        f_vra = self.cal_f_vra(f_eigen, df)
        f_vra = pd.DataFrame(index=f_raw.index, columns=f_raw.columns, data=f_vra)
        f_vra.reset_index(inplace=True)
        f_vra['datetime'] = self.trade_days[idx - 1]
        return f_vra

    @staticmethod
    def cal_f_raw(df):
        mu = df.ewm(halflife=90, adjust=True).mean().iloc[-1]
        df = df.sub(mu, axis=1)
        res = []
        for col in df.columns:
            tmp = df.mul(df[col], axis=0).ewm(halflife=90).mean().iloc[-1]
            tmp.name = col
            res.append(tmp)
        res = pd.concat(res, axis=1)
        res.index.name = 'code'
        return res

    @staticmethod
    def cal_newey_west(df):
        omega_1d, omega_2d = [], []
        factor_return_1d = df.shift(1)
        factor_return_2d = df.shift(2)
        for col in df.columns:
            tmp_1d = factor_return_1d.mul(df[col], axis=0).ewm(halflife=90).mean().iloc[-1]
            tmp_2d = factor_return_2d.mul(df[col], axis=0).ewm(halflife=90).mean().iloc[-1]
            tmp_1d.name = col
            tmp_2d.name = col
            omega_1d.append(tmp_1d)
            omega_2d.append(tmp_2d)

        omega_1d = pd.concat(omega_1d, axis=1)
        omega_2d = pd.concat(omega_2d, axis=1)

        omega = 2 / 3 * (omega_1d + omega_1d.T) + 1 / 3 * (omega_2d + omega_2d.T)
        omega.index.name = 'code'
        return omega

    def cal_f_eigen(self, f_nw):
        eigenvalues, eigenvectors = np.linalg.eig(f_nw)
        gama = self.monte_carlo(eigenvalues, eigenvectors, f_nw.values)
        eigenvalues = np.diag(np.square(gama)) * np.diag(eigenvalues)
        f_eigen = eigenvectors @ eigenvalues @ eigenvectors.T
        return f_eigen

    def monte_carlo(self, eigenvalues, eigenvectors, f_nw):
        res = np.zeros(len(eigenvalues))
        for i in range(3000):
            b = []
            for j in range(len(eigenvalues)):
                tmp = np.random.normal(0, np.sqrt(eigenvalues[j]), 252)
                b.append(tmp)
            b = np.array(b)
            f_m = eigenvectors @ b
            f_m = pd.DataFrame(data=f_m.T)
            f_m = f_m - f_m.ewm(halflife=90).mean().iloc[-1]
            f_m = self.cal_f_raw(f_m)
            d_m, u_m = np.linalg.eig(f_m)
            d_m_new = np.diag(u_m.T @ f_nw @ u_m)
            res += d_m_new / d_m / 3000
        res = np.sqrt(res)
        res = 1.5 * (res - 1) + 1
        return res

    @staticmethod
    def cal_f_vra(f_eigen, df):
       sigma = df.ewm(halflife=90).std().iloc[-1]
       b = np.square(df.div(sigma, axis=1)).mean(axis=1)
       b = b.ewm(halflife=42).sum().iloc[-1]
       return np.sqrt(b) * f_eigen


if __name__ == '__main__':
    BarraCovariance()()
