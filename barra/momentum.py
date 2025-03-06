import bisect
from base_data_loader import BaseDataLoader
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from datetime import timedelta
from tqdm import tqdm
from barra.functions import gen_weights



if __name__ == '__main__':
    data = BaseDataLoader.load_data('../../数据/stock_bar_1day.parquet', fields=['close'])
    df = data.to_dataframe('close')
    ret_1 = df / df.shift(1) - 1
    ret_21 = df.shift(-21) / df - 1

    # 短期反转
    res = {'short_term_reversal': []}
    w_5 = gen_weights(5, 21)
    for i in range(21, len(ret_1) + 1):
        tmp = np.log(1 + ret_1.iloc[i-21:i]) * w_5
        name = tmp.index[-1]
        tmp = 21 * tmp.mean()
        tmp.name = name
        res['short_term_reversal'].append(tmp)
    res['short_term_reversal'] = pd.concat(res['short_term_reversal'], axis=1).T

    # 季节因子
    res['seasonality'] = []
    for i in range(1000, len(data.trade_days)):
        cur = data.trade_days[i]
        name = cur
        tmp_data = []
        for j in range(1, 6):
            if cur.year % 4 == 0 and cur.month > 2:
                cur = cur - timedelta(days=366)
            else:
                cur = cur - timedelta(days=365)
            idx = bisect.bisect_left(data.trade_days, cur)
            tmp_data.append(ret_21.iloc[[idx]])
        tmp_data = pd.concat(tmp_data).mean()
        tmp_data.name = name
        res['seasonality'].append(tmp_data)
    res['seasonality'] = pd.concat(res['seasonality'], axis=1).T

    # 行业动量
    w_21 = gen_weights(21, 126)
    rs = []
    for i in range(126, len(ret_1) + 1):
        tmp = np.log(1 + ret_1.iloc[i - 126:i]) * w_21
        name = tmp.index[-1]
        tmp = 126 * tmp.mean()
        tmp.name = name
        rs.append(tmp)
    rs = pd.concat(rs, axis=1).T

    cap_data = BaseDataLoader.load_data('../../数据/capital.parquet', fields=['circulating_cap'],
                                        start=rs.index[0], end=rs.index[-1], codes=rs.columns.tolist())
    cap_data.data = np.sqrt(cap_data.data)
    rs = rs * cap_data.data[:, 0, :]

    ind_data = BaseDataLoader.load_data('../../数据/sw_industry.parquet', fields=['sw_l1_code'],
                                        start=rs.index[0], end=rs.index[-1], codes=rs.columns.tolist())
    ind_unique = ind_data.data[:, 0, :].astype('<U32')
    ind_unique = np.unique(ind_unique[~np.isin(ind_unique, ['None', np.nan, ''])])
    rs = rs[rs.index >= ind_data.trade_days[0]]
    x = []
    rsi = 0
    for tag in ind_unique:
        tmp = np.nan_to_num(ind_data.data[:, 0, :] == tag, nan=0)
        rsi += np.nansum(rs * tmp, axis=1, keepdims=True) * tmp
    res['industry_momentum'] = -1 * (rs - rsi)

    # 相对强度
    w_126 = gen_weights(126, 252)
    rssi = []
    for i in range(252, len(ret_1) + 1):
        tmp = np.log(1 + ret_1.iloc[i - 252:i]) * w_126
        name = tmp.index[-1]
        tmp = 252 * tmp.mean()
        tmp.name = name
        rssi.append(tmp)
    rssi = pd.concat(rssi, axis=1).T
    res['relative_strength'] = rssi.rolling(11).mean()

    for key in res:
        res[key].index.name = 'datetime'
        res[key].reset_index(inplace=True)
        res[key] = pd.melt(res[key], id_vars='datetime', var_name='code', value_name=key)
        res[key].set_index(['datetime', 'code'], inplace=True)
    df = pd.concat(list(res.values()), axis=1)
    df.reset_index(inplace=True)

    old_df = pd.read_parquet('../barra_factors.parquet')
    old_df = old_df.merge(df, how='left', on=['datetime', 'code'])

    old_df['momentum_2'] = (old_df['relative_strength'] + old_df['alpha']) / 2
    old_df['momentum_1'] = (old_df['momentum_2'] + old_df['industry_momentum']
                            + old_df['seasonality'] + old_df['short_term_reversal']) / 4
    old_df.to_parquet('../barra_factors.parquet')







